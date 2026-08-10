# W3 — BT_MOE_LOOKAHEAD_RECOMPUTE patch notes (helmholtz, 2026-08-09)

## v3 (2026-08-10, fermi) — input-dependency-only kick ordering

- **Why:** the v2 canary measured kicks 99.9 % serialized (0 wall for +25.8
  GiB). boltzmann's SM-slack measurement (curie-confirmed) refuted the "no SM
  headroom" reading: 83.5 % of bwd-phase kernel time runs at <10 % occupancy
  (~35.9 s slack/step), SendRecv windows at ~0 % occ. The serialization was
  ORDERING: v2's `side_stream.wait_stream(current)` at kick time chained each
  kick behind the whole compute backlog — including, transitively, the
  previous kick's consume-wait sitting on the compute stream's tail.
- **Change (lookahead_checkpoint.py only; recompute.py hunk byte-identical to
  v2):** the kick now waits on exactly two events — (a) its own chunk's
  `input_event` (recorded at first-pass registration; covers the saved-input
  producers and, for a microbatch's first kick, transitively the previous
  microbatch's whole backward) and (b) the registry's `last_bwd_end` event
  (recorded at the end of each chunk backward; covers the allocator reuse
  edge — side-pool blocks freed from the graph consumed two backwards ago may
  still be read by its queued kernels). `wait_stream` is banned from the
  module (AST-guarded). The distribute-saved-activations gather moved INSIDE
  the side-stream context (the two waits don't cover a compute-stream gather
  pushed at kick time). Full design + risk audit: `../DESIGN_W3V3.md`.
- **Also in v3:** telemetry window now rolls on CHUNK BACKWARDS (the v2
  events-based window + "chunk backwards" label caused the false 150/mb
  reading; counter semantics unchanged — the v2 bars kicks==hits==78/mb,
  misses==1/mb, sweeps==0, fallbacks==0 carry over verbatim); per-kick
  CUDA-event duration stats (`kick_ms_avg/max` on the window line — the
  in-log dilation signal); `BT_MOE_LOOKAHEAD_TRIM_EVERY=N` allocator-trim
  knob (default OFF; the 16k-enablement knob; sync + churn cost is
  canary-measured before any reliance).
- **Identity:** `w3-lookahead-recompute-v3.patch`, md5
  **05dda37f68f3113747a812e357d9b552**, 638 lines (new file 588 + recompute.py
  hunk unchanged). Verified: `git apply --check` clean vs the v2 base
  (57efae08b + recompute.py blob d43d8621b); applied tree reproduces the Mac
  tree byte-for-byte (lookahead_checkpoint.py md5 7304202b…, recompute.py
  0ec487cd…). CPU suite 40/40 green (incl. the dropout RNG-isolation proof
  and the new sec8 ordering guards).
- **v2 patch hygiene note (found at v3 regeneration):** v2's recompute.py
  `index` line post-image hash (3c45f3695) is STALE — it names the v1-era
  keyword-form blob; the v2 hunk CONTENT is the correct all-positional form
  (true post-image 3c9babb34). Cosmetic (plain `git apply` ignores it); the
  v3 patch carries correct hashes. Recorded so no one "verifies" v2 by its
  index line.
- **16k×d32 stays HARD OFF** (memory model there is genuinely borderline:
  MoE/MLP terms ×4 ≈ 13–14 GiB/chunk intrinsic + retention; separate
  justification + measurement required — see DESIGN_W3V3.md §5).

## v2 (2026-08-09, minkowski) — boot-blocking call-site fix

- **Defect:** the recompute.py `chunk_runner` call site passed
  `chunk_key=`/`carrier=` as keywords BEFORE `*args`; the 6 positional args
  fill signature slots 3–4 (`chunk_key`, `carrier`) and collide with the
  keywords → `TypeError: got multiple values` on all 16 ranks, 0.1 s into
  forward (box log `w3_boot_FAIL_chunk_key.log`, md5 dd58a4df). v1 never
  booted.
- **Fix (fourier's proposal, confirmed correct):** all-positional in
  signature order — `lookahead_checkpoint(cf,
  self.config.distribute_saved_activations, start + layer_offset,
  packed_seq_params, *args)` — matching the sibling `te_checkpoint` call's
  `self.config.` convention (the v1 text already used
  `self.config.distribute_saved_activations`; no discrepancy there).
  Keywords-after-`*args` is NOT an alternative: `chunk_key`/`carrier` are
  positional-or-keyword params, so the positional fill still collides.
- **Verification:** v2 hunk applied to the FIX-C-base recompute.py blob
  (`d43d8621b`) reproduces the fixed Mac tree byte-for-byte; full patch
  `git apply --check` clean vs the same base. New md5:
  **6f08c5dc08b6720c26102a47a9706b85** (509 lines), supersedes v1.
- **Epistemics (third instance tonight of tested-in-isolation /
  dead-at-integration, after the two T2 harness artifacts):** the Mac suite
  drives `lookahead_checkpoint()` directly and never executes the
  integration call site. Regression net added:
  `tests/test_w3_lookahead_checkpoint.py` now drives the call with the
  integration arity (6 trailing positional args) AND statically guards the
  recompute.py call site against keyword-before-`*args` (CPU, no CUDA).

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
