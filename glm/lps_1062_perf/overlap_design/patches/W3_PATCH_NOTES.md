# W3 — BT_MOE_LOOKAHEAD_RECOMPUTE patch notes (helmholtz, 2026-08-09)

## What

`w3-lookahead-recompute.patch` — cross-layer lookahead recompute for full
activation checkpointing: at the boundary between chunk L's recompute and
chunk L's backward, the PREVIOUS chunk's recompute is issued on a side
stream, overlapping recompute(L−1)'s compute (and its MoE all-to-alls) with
bwd(L). Targets the backward phase, where 2/3 of A2A time lives. Design:
`../DESIGN_helmholtz.md` §7 (W3); preconditions V2b/V4 PASS (recorded in
§12). Expected win: −8…−12 s/step at 60–80 % capture (with CDMC unset —
NOT touched by this patch; the unset experiment is a separate box run).

## Base (apply over)

- Vendored mcore @ `57efae08b` + FIX A/B/F + hilbert's FIX C
  (`dispatcher_opt/fixc/fixc.patch`; the recompute.py hunk has FIX C's
  pass-marker wrap as context). Independent of W1/W2 (orthogonal lever).
- Files: `megatron/core/lookahead_checkpoint.py` (NEW, 450 lines) +
  `megatron/core/recompute.py` (chunk_runner branch + import).
- Apply: `cd <mcore> && patch -p1 < w3-lookahead-recompute.patch` (or
  `git apply`). Verified: applies clean, reproduces the author's tree
  byte-for-byte.

## Gate + telemetry (WARNING level)

- `BT_MOE_LOOKAHEAD_RECOMPUTE=1`, default OFF. Gate off / no carrier
  (packed_seq_params None) / CUDA-graph capture-warmup ⇒ status-quo
  checkpoint behavior (the Function degrades to inline recompute; capture
  passes through like `random.py:642-649`).
- One-time gate-state line; one-time `armed — first checkpointed chunk
  registered`; per-window counters every 300 chunk backwards (~1 step):
  `{kicks, stash_hits, stash_misses, sweeps, fallbacks}`. Expected steady
  state per microbatch: kicks = hits = chunks−1, misses = 1 (the last chunk
  is never kick-targeted — it recomputes inline by design).
- Gate-on config assert (loud, once): `hidden_dropout == 0 and
  attention_dropout == 0` (`lookahead_check_config_once`).

## Correctness invariants (review checklist)

- **Bitwise-identical scheduling**: same run_function, same detached saved
  inputs, same restored RNG states (fork/set/restore host-atomic on the
  autograd thread). The kicked graph's leaves are the kick's own detached
  inputs; the chunk's backward collects their `.grad` (value-identical to
  the inline path's).
- **No state on pass-through tensors** (the BUG-A lesson): all lookahead
  state lives on the carrier's registry (ctx ref, rng states, fp8 snapshot,
  kicked inputs/outputs, done-event). The Function returns run_function's
  fresh outputs only.
- **Stream/allocator safety**: each kick starts with
  `side_stream.wait_stream(current)` (orders the side stream after the
  compute stream's backlog — blocks freed during earlier backwards can never
  be clobbered while still read; costs ~nothing because the next backward
  hasn't been pushed yet). The consuming backward waits the kick's
  done-event and `record_stream()`s the stashed outputs.
- **Eviction discipline (fibonacci's hardening)**: entries popped on
  consume; registry swept when the last remaining chunk's backward completes
  (a non-empty sweep logs loud — structural bug); the registry lives on the
  per-microbatch carrier, so anything left dies with the microbatch.
- **FIX C composition**: the kick runs the (possibly pass-marker-wrapped)
  run_function under enable_grad → classified as a replay → FIX C
  store/hit pattern is unchanged (first pass stores, kicked replay hits,
  no second lookup).
- **fp8**: mirrors CheckpointWithoutOutputFunction's fp8 snapshot +
  `activation_recompute_forward` first-pass/recompute-phase contexts
  (random.py:666-682, 758-782).

## Tests (Mac CPU, ALL PASS)

`../tests/test_w3_lookahead_checkpoint.py` (monkeypatched CPU RNG stubs —
the real helpers touch CUDA RNG; the fork/restore LOGIC under test is
identical and the CUDA path is box-covered):
1. **RNG-isolation with dropout=0.5** (fibonacci's mechanism proof): 3-chunk
   chain, kicked vs unkicked order must be bitwise-identical on outputs AND
   input grads — any fork/restore leak flips a dropout mask. PASS ×3 seeds.
2. Kick/skip counters: kicks=2, stash_hits=2, stash_misses=1, fallbacks=0,
   sweeps=0 for a 3-chunk microbatch.
3. Eviction: registry empty at microbatch end; two fresh-carrier
   microbatches bitwise-identical (no cross-datum stash leak).
4. Fallback: carrier=None → status-quo bitwise + fallbacks counted.
5. RNG save/restore round-trip; gate telemetry.

## Known limits / follow-ups

- With `CUDA_DEVICE_MAX_CONNECTIONS=1` (devbox benches) the two op sequences
  head-of-line block at stream waits — capture ratio degrades (the memo §7e
  analysis). The CDMC-unset experiment (V2, prod parity; V2b now PASS) is a
  separate box run; this patch does not touch CDMC anywhere.
- The kick's host pushes interleave the dispatcher's replay D2H event syncs
  (~14 ms each) — pair with `BT_MOE_DISPATCH_REPLAY_CACHE=1` (FIX C) to
  remove them from the replay path (the memo §7c synergy).
- Memory: +1 layer of live recompute activations (~2–4 GiB at 131k; ~8–12
  GiB at 16k×d32 — measure before enabling there; per-shape gate if tight).
- The registry key is the chunk's global layer index; MTP layers and
  extract_layer_indices flow through the same chunk_runner and are covered
  by the same ordered-registry mechanics.
