# DESIGN: FIX C — MoE dispatcher replay-metadata reuse (hilbert, 2026-08-09)

**Gate:** `BT_MOE_DISPATCH_REPLAY_CACHE=1` (default OFF) · **Verify:**
`BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY=1` (requires the main gate; soaks only) ·
**Files:** `megatron/core/recompute.py` (+112), `megatron/core/transformer/moe/
token_dispatcher.py` (+425) · **Patch:** `fixc.patch` · **Test:**
`test_fixc_replay_cache.py` (Mac-CPU, 49 assertions, ALL PASS) · **Checker:**
`runs/overnight_20260809_dispatcher_hostsync/check_acceptance.py` profile `post-patch-BFC-4mb131k`.

## 0. What and why

Under full recompute, every MoE layer-microbatch runs
`MoEAlltoAllTokenDispatcher.preprocess` twice: the no-grad first pass and the
grad-enabled recompute replay. Each run pays (a) a 16-rank all-gather over
tp_ep (`:558` upstream) and (b) a side-stream D2H batch + deferred
`d2h_event.synchronize()` (`:959` upstream). Post-B+F, the replay half is the
dominant remaining host block: 300 replay event-syncs = 23.7 s CPU per
4×131k step (runahead-inflated; ATTRIBUTION addendum §3) plus 300 replay
all-gathers. The replay recomputes `routing_map` **bitwise-identically**
(same inputs, RNG state restored by the checkpoint), so every value derived
from it is identical too. FIX C stores the first pass's split metadata and
reuses it in the replay: replay all-gather, D2H copies, and event sync are
skipped entirely. Realistic wall win today ~1–3 s/step; it is also the
precondition for un-parking FIX A (A's exposure is throttled by exactly this
replay sync — PATCH_NOTES dependency chain) and matters more once
a2a↔compute overlap lands.

## 1. The keying problem and the design

Replay recomputes `routing_map` as a NEW tensor (same values) — no identity
key exists, and a value key needs a sync (self-defeating). The key must come
from the **pass context**, not the data.

### 1.1 Pass classification: checkpoint-pass frames (recompute.py)

Full recompute drives each chunk through `checkpointed_forward`
(`recompute.py`), which hands a `custom_forward` closure (`cf`) to either
TE's `te_checkpoint` (FP8 path — the ship config) or mcore's
`tensor_parallel.checkpoint` → `CheckpointFunction`. Patching either
checkpoint *implementation* would miss the other (and patching TE means
patching site-packages — fragile). Instead, FIX C wraps `cf` **before** it
reaches either checkpoint:

```python
cf = _wrap_checkpoint_chunk_pass(cf, packed_seq_params)   # in chunk_runner
```

`_CheckpointChunkPassMarker` is a plain callable (NOT an
`autograd.Function`). The checkpoint calls it in both passes, and because it
is ordinary code executed *by* the Function — not a Function itself —
`torch.is_grad_enabled()` inside it truthfully discriminates:

| pass | called by | grad mode |
|---|---|---|
| first pass | `Function.forward` | **False** — always; PyTorch runs every autograd `Function.forward` under no-grad (the same torch invariant that made v1's FIX A gate inert — here it is load-bearing in our favor) |
| replay | `Function.backward` | **True** — both mcore and TE re-run `run_function` under `torch.enable_grad()` (required for grads to flow) |

Each call pushes a `CheckpointPassFrame(is_replay, key_obj, scratch)` onto a
**thread-local** stack for the duration of the pass. Thread-local is
mandatory: the first pass runs on the main thread while another
microbatch's replay runs on the autograd worker thread (the trace-proven
"bwd-mb1 ∥ fwd-mb2" overlap) — a global flag would cross the streams.

Gate-off behavior: `_wrap_checkpoint_chunk_pass` returns `cf` unchanged, no
frames exist, and the dispatcher never queries — byte-identical path.

### 1.2 Microbatch key: the packed_seq_params carrier

The frame's `key_obj` is the microbatch's `packed_seq_params` — the same
per-microbatch carrier FIX B uses (`_dsa_cp_layout_cache`), closure-captured
by `custom_forward`, therefore the **identical object** in both passes.
(FIX B's shipped patch proves the pattern in this tree.) When
`packed_seq_params is None` (non-packed configs), the marker instance itself
is the key (held alive by the checkpoint context from first pass through
backward).

Entries are stored **on the key_obj** (`_moe_dispatch_replay_cache` attr,
dict keyed by `id(dispatcher)` — one per MoE layer per microbatch) and
**popped on first replay use**. Consequences:

- **No cross-step / grad-fence staleness by construction.** The carrier dies
  with the microbatch; an entry whose replay never runs (exception mid-step)
  is GC'd with it. There is no ordering assumption to break.
- **4+ datums in flight: correct.** Each replay looks up *its own*
  microbatch's carrier; interleavings (fwd mb2 on main thread ∥ replay mb1
  on autograd thread) touch different carriers and different thread-local
  frame stacks. (Tested: 4 mb × 2 layers, fwd-then-bwd, bitwise a2a log.)
- **Eval/inference/non-recompute training: untouched.** `checkpointed_forward`
  only runs in training full-recompute; eval forwards never see a frame, so
  they never store and never pollute (tested).
- **A-v3 composition (PATCH_NOTES §FIX A v3):** A-v3 stashes
  `(event, pinned_buf)` on the same carrier keyed by `layer_number` at the
  dsa.py level (where the carrier is in scope and `is_grad_enabled` already
  discriminates). C's carrier dict is a separate attr — no collision; C's
  frame machinery is reusable if A-v3 ever needs it at a non-carrier site.
  Do not start A-v3 before C lands (dependency chain).

### 1.3 What is cached (per layer-microbatch)

Captured at the first pass's sync point (all host values valid, device
values final), restoring the exact instance-state transitions of the
status-quo replay:

- device form (installed by replay `preprocess`): `input_splits`,
  `output_splits`, `output_splits_tp`, `num_out_tokens` (int in the dropless
  path), `num_tokens_per_local_expert` (the `preprocess` return),
  `num_global_tokens_per_local_expert` (consumed on-device by the fused
  `sort_chunks_by_idxs`);
- host form (installed at the replay's DtoH point): the numpy/CPU versions
  of the above plus the host `tokens_per_expert`. The
  `num_global_tokens_per_local_expert` host form is recorded only when the
  status quo would have produced one (`num_local_experts > 1 and not
  moe_permute_fusion`) — the ship config is fused, so only the device form
  is carried (tested both ways).

Memory: a few hundred bytes per entry × 75 MoE layers × in-flight
microbatches — negligible. Device tensors are held by the entry until the
replay pops it; the caching allocator cannot recycle them while referenced.

## 2. Control flow

```
first pass (main thread, per MoE layer-mb)          replay (autograd thread)
------------------------------------------          ------------------------
marker: push frame(is_replay=False, carrier)        marker: push frame(is_replay=True, carrier)
preprocess:                                         preprocess:
  _replay_pass_begin -> rec="store"                   _replay_pass_begin:
  [status quo: sum, all-gather, splits]                 pop carrier[id(self)]
  capture device_vals -> rec.scratch                    miss/shape-mismatch -> rec="fallback"
_maybe_dtoh @ dtoh_point: [status quo copies]           + WARNING; else rec="hit", restore device
_maybe_dtoh @ sync_point:                               attrs, return cached device tpe
  d2h_event.synchronize()                             (skips sum + all-gather)
  rec="store" -> _store_replay_entry:                 _maybe_dtoh @ dtoh_point:
    build entry from device_vals + host attrs           install cached host values;
    stash on carrier[id(self)]; stats stores+=1         d2h_event=None; return cached host tpe
... rest of forward unchanged ...                     _maybe_dtoh @ sync_point: NO-OP (no sync)
                                                      ... rest of replay identical (a2a re-runs
                                                      on cached host splits; permute re-runs) ...
                                                      combine_finish clears instance attrs
VERIFY=1 (soak): replay runs the FULL status-quo body (all-gather + D2H +
sync) and then asserts the fresh values bitwise-equal the popped entry
(device compares at preprocess end, host compares after the sync).
Mismatch -> RuntimeError (training stops loudly). stats verifies+=1.
```

Fallbacks (all counted + WARNING): replay key miss; routing_map shape
mismatch. Gate off / no frame: `_replay_pass_begin` returns None before any
state change. `drop_and_pad` path: untouched (no all-gather, no D2H — nothing
to cache).

## 3. Telemetry (WARNING level; the v1 lesson)

- one-time gate-state line (`ACTIVE` / `present but DISABLED`) at first use;
  same for VERIFY (plus "ignored" if the main gate is off);
- one-time `armed — first replay-metadata entry stored`;
- one-time `first replay cache hit — replay all-gather, D2H copies, and
  d2h_event.synchronize() skipped`;
- immediate WARNING on every fallback (miss / shape mismatch);
- per-window counters every 300 replay lookups (~1 step at 75×4) for the
  first 100 windows: `stores / hits / misses / shape_mismatches / verifies`.
  A present-but-inert patch shows up as `hits=0` + misses; a gate that never
  wraps shows no `armed` line. Expected steady state at 4 mb: `stores=300,
  hits=300, misses=0` per step.

## 4. Risk register

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| Replay routing diverges from fwd (ULP nondeterminism in router/GEMM) | very low (deterministic by design: same inputs, restored RNG) | silent wrong splits → corrupt a2a | VERIFY soak asserts bitwise equality on every replay before any timed run; loss-canary bar (bitwise vs gates-off) |
| TE checkpoint grad-mode semantics differ from mcore | very low (Function.forward no-grad is a torch invariant; replay must enable_grad for grads to flow) | frames misclassified → patch inert (never hits) | telemetry (`armed`/`first hit`/window counters) + on-box activation check; CPU test drives the real mcore CheckpointFunction |
| Marker wrap perturbs te_checkpoint (introspection of `cf`) | very low (TE calls `forward_func(*args,**kwargs)`; a functor is transparent) | startup failure | gate-off returns `cf` untouched; boot smoke in the recipe |
| Cross-thread instance-attr races (main fwd ∥ autograd replay on the same dispatcher) | pre-existing (status quo has the same write pattern; C adds no new instance channel — pass state lives on the thread-local frame) | as today | no regression vs status quo by construction; noted for the record |
| Stale entry served | ~impossible by construction (entries live on the per-mb carrier, popped on use, GC'd with the carrier) | — | shape guard + verify mode as defense-in-depth |
| Nested recompute (recompute_mlp inside full) | not the ship config | double dispatcher pass per frame | idempotent re-restore on repeat `_replay_pass_begin` within a frame (covered in code; not the ship path) |
| CUDA-graph capture paths | not the ship config (checkpoint is skipped during capture) | none | no frames during capture → status quo |
| Collective-skip divergence (some rank replays, some don't) | ~impossible (all ranks replay the same layers in the same order under full recompute) | hang | all ranks skip the same replay all-gathers; verify soak runs the full model |

## 5. What FIX C deliberately does NOT do

- **FIX D (batched D2H): not folded in.** The remaining fwd copies are
  µs-scale CPU each (~0.2–0.5 s/step); packing 5 tensors into one staged
  pinned buffer needs an on-device `cat` (extra launches in the fwd launch
  pipeline we just decompressed) plus special-casing the python-int
  `num_out_tokens`. Not trivial, not worth the risk slot. Revisit if the
  fwd D2H shows in a post-C capture.
- **The fwd event sync (300/step, ~9.9 s CPU runahead-inflated): stays.**
  The fwd sync is load-bearing for the real forward (the a2a needs host
  splits). It shrinks naturally as the launch pipeline keeps decompressing.

## 6. Acceptance

- Trace (`check_acceptance.py --profile post-patch-BFC-4mb131k`):
  `eventsync_dispatcher_replay_calls` 300→~0; `eventsync_dispatcher_fwd_calls`
  300 unchanged; `dispatcher_allgather_replay_calls` 300→~0 (calibrate the
  allgather metric on a baseline capture first — see measure() comment);
  dtoh-pinned and memcpy rows drop by the replay half.
- Log: `armed` + `first replay cache hit` at boot; window counters
  `hits=300, misses=0, shape_mismatches=0` per step.
- Loss canary: bitwise vs gates-off at identical `--warmup-datums` (the
  rng-stream rule); drift > 5e-3 = stop.
- No window-1 profiled captures (graph-capture/compile contamination) —
  steady windows only.

## 7. On-box validation recipe

See `ONBOX_VALIDATION.md` (10 lines).
