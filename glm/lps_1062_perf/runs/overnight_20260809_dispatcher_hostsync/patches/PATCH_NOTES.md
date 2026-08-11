# PATCHES: LPS-1062 host-sync elimination (ramanujan, 2026-08-09)

Three env-gated patches (all default OFF), developed against the Mac checkout
(trainers ef4ea4a8, vendored mcore 57efae08b). The three touched files are
byte-identical between that pin and the on-box pin (mcore d3932e757c), so the
patches apply cleanly to the shared on-box checkout. Parity tests:
`runs/overnight_20260809_dispatcher_hostsync/tests/` (Mac-CPU green as of this writing; GPU sections
auto-enable on the box).

Apply order: independent; any subset may be applied/toggled for bisection.

## ⚠️ v1 → v2 correction (2026-08-09, post-sign-off)

**0001 v1 was present-but-inert on-box** (gibbs's capture: 312/312 drains
present, zero probes). Root cause: v1 gated the probe kick on
`torch.is_grad_enabled()` *inside* `FusedSparseAttentionFunc.forward` — and
PyTorch runs every autograd `Function.forward` under no-grad (verified
empirically on torch 2.11; behavior since ~1.10), so the gate was always
False, in every window, on every config. The v1 file is archived as
`quarantine/0001-fix-a-dsa-bwd-async-nonempty-v1-INERT.patch` for provenance (it is what
produced the BF-only measurement).

**v1→v2 delta (for laplace's delta review — complete list):**
1. Removed the `torch.is_grad_enabled()` check from
   `_maybe_kick_nonempty_probe` (THE fix; the discarded no-grad first pass now
   also kicks — harmless: its ctx is dropped, nothing waits on its event).
2. Header comment rewritten accordingly (documents why grad mode cannot gate
   inside Function.forward).
3. Loud telemetry (pascal's requirement, all at WARNING because logger.info
   from megatron.core does not reach trainer_srun.log): one-time gate-state
   line per gate ("ACTIVE" / "present but DISABLED") at first use; one-time
   "armed — first async nonempty probe kicked" on first successful kick;
   per-window counter now reports `reads/fallbacks/probes_missing` (a
   present-but-inert patch shows probes_missing=156/156).
4. `_read_nonempty_probe` with gate ON and no probe on ctx now counts
   `probes_missing` (+ one-time WARNING) instead of silently falling back.
5. NO changes to the fast-path/slow-path dataflow, the ordering spec, or the
   arange substitution — numerics-relevant code untouched.

0002/0003 gained only the one-time armed/disabled gate line (no logic change).

**Test gap that let v1 through, now closed:** the GPU probe test called
`_maybe_kick_nonempty_probe` directly (outside any autograd Function), where
grad mode is truthful — it never exercised the Function.forward no-grad
bubble. v2's test suite adds `_checkpoint_replay_test`: the Function under
`torch.utils.checkpoint` (no-grad first pass + enable_grad replay, matching
mcore's structure) with a kick spy — asserts the probe fires in both passes
and backward finds it (`probes_missing == 0`), GPU liveness + everywhere
bitwise. Plus a CPU-checkable source guard: the kick must not contain an
`is_grad_enabled` gate statement. This test fails on v1.

**On-box swap (gibbs)** — verified in a scratch worktree against the exact
current on-box state (v1+0002+0003 applied to d3932e757c):

```
cd /root/.cache/user_artifacts/trainers_main/server/vendor/megatron-bridge/3rdparty/Megatron-LM
git apply -R 0001-fix-a-dsa-bwd-async-nonempty-v1-INERT.patch   # reverse v1
git apply    0001-fix-a-dsa-bwd-async-nonempty.patch            # apply v2
# 0002/0003 stay applied as-is (or re-apply the regenerated ones — they only
# gained the armed log line; the swap for them is identical apply -R + apply)
```

**Activation check (post-swap, definitive):** with
`BT_DSA_BWD_ASYNC_NONEMPTY=1`, the first gated forward logs
`BT_DSA_BWD_ASYNC_NONEMPTY=1: ... ACTIVE` then `...: armed — first async
nonempty probe kicked` (WARNING → trainer_srun.log); the per-window line shows
`probes_missing=0 fallbacks=0`. Trace counters on the next capture:
`aten::all` +312, `cudaEventSynchronize` +312 all µs-scale (REVIEW_FIXA §2c #4:
no new slice >1ms), `aten::nonzero` slices >5ms = 0.

## 0001 — FIX A: `BT_DSA_BWD_ASYNC_NONEMPTY` (dsa_cudnn_kernels.py)

**Target:** the 156/step `torch.nonzero(topk_length > 0)` pipeline drains in
the DSA sparse-attention backward (`_run_sparse_attention_backward`), 19.2s
CPU on the exp05d trace (93% of all nonzero CPU).

**Mechanism:** `FusedSparseAttentionFunc.forward` kicks an async 1-byte D2H of
`(topk_length_flat > 0).all()` on a module-level side stream (ordering: `.all()`
on the compute stream, THEN `side_stream.wait_stream(current)`, then pinned
copy + `record_event`) and stashes `(event, pinned_buf, flag_gpu)` on ctx.
Backward does `event.synchronize()` and, when all rows are non-empty,
substitutes `torch.arange(N)` for the nonzero — **bitwise identical by
construction** (nonzero of an all-true mask is exactly row-major arange,
int64), keeping every other op of the compaction dataflow (dummy-row cat,
index_selects, wrapper call) unchanged. Any empty row → original syncing
nonzero (status-quo cost).

**Why the bwd event-sync is ~µs and deadlock-free (required argument):**
under full recompute, the probe fires during the *replay* (the ctx the
backward actually reads; the discarded no-grad first pass also kicks —
harmless, see the v1→v2 note). The backward runs tens of ms later, after the
replayed layer forward has finished on the compute stream; the event waits
only on the side-stream 1-byte copy — ordered after replay's `topk_length`
production, long complete. The wait is `cudaEventSynchronize` on a completed
event: it never touches or drains the compute stream. No circular wait:
backward holds nothing the side stream waits on. Correct under
`CUDA_DEVICE_MAX_CONNECTIONS=1` (wait_stream is a device-side ordering
snapshot, not a host block) — and the trainer leaves that var unset anyway
(verified in the captured /proc environ). CUDA-graph capture → original path.
Per-call pinned buffer + device flag referenced from ctx (no ring buffer) so
neither recycles while the D2H is in flight.

**Telemetry (REVIEW_FIXA 2b, binding):** per-window fallback counter logged
every 156 flag-reads (≈1 step at 78 layers × 2 mb) for the first 100 windows,
plus an immediate WARNING on any fallback. Expected: 0/156 per step on bench
and customer shapes; routine fallbacks void the win claim — stop and report.

## 0002 — FIX B: `BT_DSA_CP_LAYOUT_CACHE` (dsa.py)

**Target:** the 26,520/step boolean-mask-index nonzeros (85 per layer-mb-pass =
17 × 5) in `build_packed_allgather_cp_query_positions_and_key_reorder`,
recomputed identically in all 78 layers and the recompute replay; ~1.3s CPU +
~150k kernel launches per step.

**Mechanism:** cache the builder results on the `packed_seq_params` carrier —
the same per-microbatch lifetime pattern as the existing
`_dsa_index_share_topk_holder`. All three call sites covered (main
`cp_size > 1` branch, the sequence-parallel-TP branch, `_build_kv_reorder_idx`).
Entries key on the full argument tuple and hit only on **object identity** of
the input cu_seqlens tensors (refs held in the entry), so stale reads are
impossible. Cached tensors are the identical objects from the first
computation → **bitwise exact**. Downstream consumers are read-only
(`index_select`, mask construction via `build_dsattention_forward_mask`).

## 0003 — FIX F: `BT_THD_ROPE_HOST_CACHE` (rope_utils.py) — **SITE CORRECTION**

**Correction to laplace's ATTRIBUTION §3 (he flagged ±1 line uncertainty):**
the 15.0s `cudaMemcpyAsync` class is **not** `sort_chunks_by_idxs` — the TE
fused path is active (`te_moe::chunk_sort_fwd` ×600 in the trace, so no
unfused tolists run there). The blocking pageable D2H is the **THD RoPE
bookkeeping** in `_apply_rotary_pos_emb_thd` (`rope_utils.py:222-239`):
`((cu_seqlens[1:] - cu_seqlens[:-1]) // cp_size).tolist()` + per-sequence
`.item()`s, called 2×/layer (q/k rope, absorbed_mla.py:626/636; MLA-style
interleaving forces the unfused path). Evidence: 616 `aten::to` slices >1ms
(14.8s) are direct children of CheckpointFunction{,Backward} at 2/layer-mb/pass
(308 ≈ 78×2×2 fwd), and the op motif around them is `sub → floor_divide →
blocking to(D2H) → item×2 → cat×3 → cos/sin` (rotary). Hence the honest gate
name (pascal's `BT_MOE_SORT_CHUNKS_D2H_BATCH` would be a misnomer — there is
no sort_chunks D2H problem to batch).

**Mechanism:** cu_seqlens is a per-microbatch constant shared by all layers;
take one host copy per unique tensor object (module-level FIFO cache, 64
entries, strong refs so ids can't recycle) and run all host reads on it.
Identical integer values and arithmetic → **bitwise exact**; steady state has
zero device syncs in this function (624 → ~4 blocking D2H per step). Gate off
→ byte-unchanged path. **Sign-off hardening (pascal/laplace, 2026-08-09):**
each entry also records the tensor's `_version` counter and hits only while it
matches — an in-place mutation of a cached cu_seqlens forces a fresh copy
instead of silently serving stale seqlens (covered by a dedicated
mutation-miss case in the parity test).

## Parity tests (`runs/overnight_20260809_dispatcher_hostsync/tests/`)

Standalone (no trainer deps); import the real patched modules from the vendored
tree. On the box, point `BT_TEST_MCORE_PATH` at the patched on-box mcore:

```
python test_dsa_bwd_async_nonempty_parity.py   # FIX A: arange==nonzero,
    # backward bitwise (both branches), FORCED-FALLBACK case (zeroed rows),
    # gating matrix, function-level fwd+bwd parity; GPU: real probe semantics
python test_dsa_cp_layout_cache_parity.py      # FIX B: cached vs direct
    # bitwise across cp sizes/ranks/docs/padding; miss/hit/identity safety
python test_thd_rope_host_cache_parity.py      # FIX F: gate off/on bitwise
    # through the real _apply_rotary_pos_emb_thd, both freqs formats,
    # both interleave modes; cache identity + FIFO bound
```

Mac-CPU results 2026-08-09 (v2): A 23 PASS (+2 GPU-only skips; includes the
checkpoint-replay regression test that fails on v1), B 191 PASS, F 167 PASS.
All three re-verified against the post-swap on-box state (v2+0002+0003 applied
to d3932e757c) in a scratch worktree.

## On-box results (2026-08-09, gibbs A/B + laplace trace check)

**v2 activation: proven.** Predicted counters matched exactly (eventsync 912,
`probes_missing=0`, nonzero >5ms drains gone).

**Package (A+B+F) at 4×131K: +11–14% (634 → ~720 tok/s/GPU steady).** The win
is launch-pipeline decompression, not comm de-staggering: SendRecv totals and
per-call p50/p90 flat; GPU-union idle −6.7s (8.52s → 1.82s) against a 7.3s
profiled wall win. B+F did the heavy lifting (RoPE tolist copies 13.9s CPU +
53k layout-builder nonzeros + ~300k extra kernel launches had starved the
launch pipeline at 4mb). The earlier 0.7–2s win bound was derived at
2mb/256K where the blocks were ~fully overlapped; it was shape-specific and
did not transfer — recorded as a scaling-law lesson (op counts scale with M;
host-block wait totals are ~M-invariant; GPU starvation share is
shape-dependent; see ATTRIBUTION addendum §2/§5).

**FIX A's marginal contribution: ≈0 to −2% — mechanically perfect, throttled
upstream.** A's flag events are µs-scale as designed (312 calls, 0.46s CPU,
p50 5.5µs; GPU idle inside their windows 0.012s; the 312 backward drains are
gone). It buys ~nothing today because the dispatcher's replay
`d2h_event.synchronize()` (p50 77ms) fires upstream of A's read point in every
layer-backward and throttles the autograd thread to GPU-lockstep — A's event
and the original nonzero are both cheap at that point regardless. The
`cudaEventSynchronize` CPU balloon to 33.56s is the **dispatcher's** class
(2.22s pre-patch), inflated by the decompression itself (host runs ahead →
waits lengthen in wall terms while overlapping GPU work; benign: GPU idle
1.82s) — NOT A's flag events. **Correction for the record:** the intermediate
"FIX A's drain relocated into its own event waits" theory (carried here from
the pre-trace diagnostic) was refuted by laplace's trace check — A's events
are µs; the §5 review's stream model was right about A. What nobody's model
caught was the *dispatcher-throttle* interaction (A's exposure is a sequela of
FIX C) and the shape-dependence of the win bound. **Decision: ship B+F; FIX A
stays default-off (parked).**

### Future work: FIX A v3 (sketch, not scheduled)

What v3 would take: compute the flag in the **no-grad first pass** (seconds
before the backward, GPU queue shallow) and thread it to the backward, instead
of kicking in the replay. The first pass's ctx is discarded, so the probe must
live on a per-(layer, microbatch) cache — the same keying problem flagged for
FIX C. Feasible concretely: at the `dsa.py` level (a plain function, where
`torch.is_grad_enabled()` *does* discriminate the passes — False in the first
pass, True in the replay), kick the probe on `topk_length` in the first pass
and stash `(event, pinned_buf)` on the `packed_seq_params` carrier keyed by
`layer_number` (the carrier lives exactly one microbatch, fwd through bwd; the
DSA top-k holder is the precedent); in the replay, fetch the stash and pass it
through `run_fused_absorbed_sparse_attention` into
`FusedSparseAttentionFunc.apply` so the backward's ctx references the
*first-pass* probe — an event that completed long ago. This matters in the
post-FIX-C regime specifically: today A's replay-anchored events measure µs
(p50 5.5µs) only because the dispatcher's 77ms replay syncs throttle the
autograd thread to GPU-lockstep upstream; once FIX C removes that throttle and
the CPU runs ahead, a replay-anchored event would wait on replay-fwd GPU
progress (the relocation returns) — first-pass anchoring keeps it µs
regardless. Risks: (i) the flag describes the first pass's `topk_length`
while the backward consumes the replay's — requires replay identity; the
load-bearing mitigation is that row-emptiness is *structural* (mask bounds),
not score-dependent, so even ULP-level replay nondeterminism in the indexer
cannot flip a row's emptiness — the residual risk is a genuine kernel bug,
which the loss canary covers; (ii) plumbing through the Function's signature
(backward return arity); (iii) 156 in-flight probes per step (fine).

**Dependency chain (corrected per laplace's ATTRIBUTION addendum §3–4):** A's
exposure is a sequela of **FIX C**, not of comm improvements per se. The
dispatcher's replay `d2h_event.synchronize()` (p50 77ms, 23.7s CPU across the
300 replay reads — runahead-inflated, now the dominant host block) fires
*upstream* of A's read point in every layer-backward and throttles the
autograd thread back to GPU-lockstep, so A's event (and the original nonzero)
are both cheap at that point regardless — comm wins alone will not expose A
while that dispatcher throttle stands. The order is therefore: **FIX C first**
(replay-metadata reuse; removes the 23.7s replay eventSync class; keying
problem as scoped in CANDIDATE_FIXES §FIX C) → **then A-v3 becomes worth
measuring** → comm wins expose both. Do not start v3 before FIX C lands (or a
customer shape shows A's block on the critical path).

## On-box acceptance (from REVIEW_FIXA §2c, gibbs runs)

- `aten::nonzero` ≤ ~200 calls / ≤ 0.2s CPU (from 26,684 / 20.6s); zero slices
  >5ms; layout-builder nonzeros ~170 (FIX B first-layers only).
- `cudaStreamSynchronize` ≤ ~500 / ≤ 0.6s; blocking pageable-DtoH copies >1ms
  = 0 (FIX F); `Memcpy DtoH (Device -> Pinned)` ≤ ~3k.
- `cudaEventSynchronize` ~456 / ≤ 4.5s and **no new slice >1ms** (proves the
  FIX A replay argument).
- Fallback counter: 0/156 per step, first 100 steps.
- Loss canary: identical `--warmup-datums` across boots, drift ≤ 5e-3.
- Throughput non-regression ≥ baseline −2%; expected +1–4% (a "win" > +6% is
  a measurement artifact per laplace — re-measure).
