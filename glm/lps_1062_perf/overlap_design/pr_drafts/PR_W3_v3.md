# PRE-DRAFTED PR BODY — W3 v3: lookahead recompute, input-dependency-only ordering (mcore)

**Repo:** basetenlabs/Megatron-LM · **Branch:** to be cut from the v3 patch
(`w3-lookahead-recompute-v3.patch`, md5 05dda37f68f3113747a812e357d9b552)
on the ship stack · **Base:** per stack state at open (W2/A-v3 verdicts
permitting)
**OPEN CONDITION (helmholtz):** only after the W3-v3 canary PASSES the
frozen acceptance frame
(~/perf_profiles/lps-1062/W3V3_CANARY_ACCEPTANCE_FRAME.md, curie). Do NOT
open before helmholtz confirms. (Skeleton — fill the canary numbers at open
time.)

---

## What

`BT_MOE_LOOKAHEAD_RECOMPUTE=1` (default OFF): cross-layer lookahead recompute
for full activation checkpointing. A checkpointed chunk's recompute replay
depends only on its own saved inputs (produced in the forward phase) — not on
the current chunk's backward — so at the boundary between chunk L's recompute
and chunk L's backward, chunk L−1's recompute is **kicked early on a side
stream**, overlapping its compute (and its MoE A2As) with chunk L's backward.
2/3 of A2A time lives in the backward phase; that is the territory.

**v3 vs v2 (the whole story):** v2's mechanism was proven on-box and its win
refuted *as implemented* — kicks 99.9 % serialized, +25.8 GiB for 0 wall.
The gate measurement then proved the bwd phase carries ~35.9 s/step of SM
slack (83.5 % of kernel time <10 % occupancy; the ~27.5 s/rank/step of
SendRecv comm windows at ~0 % occupancy). The serialization was **ordering**:
v2's `side_stream.wait_stream(current)` chained each kick behind the whole
compute backlog — transitively behind the previous kick's consume-wait on
the compute tail. v3 replaces it with **input-dependency-only ordering**: the
kick waits on exactly two events — its own chunk's `input_event` (recorded
at first-pass registration) and the previous chunk-backward's `last_bwd_end`
event (the allocator reuse edge). Both complete long before kick time in
steady state, so the kick co-runs with the current chunk's backward window.

## Why (win model, with the honest open term)

Per layer-bwd window (300/step at 131k×d4): serial ≈ 105–110 ms today
(recompute ~41 + bwd ~65); v3's floor ≈ max(comm ~46, compute ~50) + tails ≈
55–65 ms → **−8…−12 s/step at 60–80 % capture**. The capture risk is
**HBM-bandwidth contention** (recompute into comm windows competes for BW) —
measured in the canary frame (per-occupancy-bucket bwd-kernel dilation +
in-log `kick_ms` telemetry), not pre-promised. Break-even: the win goes
negative only if co-running nearly doubles bwd compute time; the realistic
risk is a smaller win.

**Canary result:** *fills at open time — the frozen frame's L/T/M/N bars +
the verdict rule.*

## Numerics / memory

- Pure scheduling: same kernels, same inputs, same RNG (fork/set/restore
  host-atomic on the autograd thread; dropout=0 asserted at gate-on), same
  reduction orders. In-process gates hard-bitwise (CPU suite 40/40 incl. the
  dropout=0.5 RNG-isolation mechanism proof); ship verdicts through the
  house band (≤2e-3 / >5e-3, matched `--warmup-datums`, 20-step drift
  cover).
- Memory at 131k: ~11.5 GiB intrinsic (two ~11.4 GiB live chunk graphs at
  the mid-backward peak; depth bounded at 2 by construction) + ~14 GiB
  allocator retention on the side-stream pool — **re-measured under v3 by
  the canary (bar ≤ +28 GiB vs baseline)**, not inherited from v2's
  serialized-regime number. `BT_MOE_LOOKAHEAD_TRIM_EVERY=N` releases the
  retention at microbatch boundaries (default OFF at 131k; the
  16k-enablement knob). **16k×d32 stays HARD OFF** (corrected model ≈ 13–14
  GiB/chunk intrinsic + retention — genuinely borderline; separate
  justification + measurement required).

## Evidence / tests

- Design (ordering rule, E1–E8 cross-stream allocator audit, depth bound,
  win model, canary bars): [overlap_design/DESIGN_W3V3.md](https://github.com/JackRao123/experiment-artefacts/blob/main/glm/lps_1062_perf/overlap_design/DESIGN_W3V3.md)
- v2 lineage + verdict record: [patches/W3_PATCH_NOTES.md](https://github.com/JackRao123/experiment-artefacts/blob/main/glm/lps_1062_perf/overlap_design/patches/W3_PATCH_NOTES.md)
  · [ESTATE_NOTES_minkowski.md](https://github.com/JackRao123/experiment-artefacts/blob/main/glm/lps_1062_perf/overlap_design/ESTATE_NOTES_minkowski.md)
  · [ESTATE_NOTES_fermi.md](https://github.com/JackRao123/experiment-artefacts/blob/main/glm/lps_1062_perf/overlap_design/ESTATE_NOTES_fermi.md)
- SM-slack gate measurement: ~/perf_profiles/lps-1062/round3/SM_SLACK_RESULT_318g61w.md
  (boltzmann's analyzers: dispatcher_opt/analyzers/)
- Tests: [tests/test_w3_lookahead_checkpoint.py](https://github.com/JackRao123/experiment-artefacts/blob/main/glm/lps_1062_perf/overlap_design/tests/test_w3_lookahead_checkpoint.py)
  — 40/40 Mac-CPU: bitwise parity under dropout=0.5 (RNG-isolation proof),
  kick/eviction/fallback counters, integration-arity + AST call-site guard,
  and the v3 ordering guards (no `wait_stream` anywhere; exactly the two
  `wait_event` edges; gather-inside-stream-context; both event producers
  pinned).
- Telemetry: WARNING-level gate/armed lines; per-window counters every 300
  true chunk backwards — kicks == stash_hits == 78/mb, misses == 1/mb,
  sweeps == 0 (halt), fallbacks == 0 — plus `kick_ms_avg/max` dilation
  stats. (The v2 events-window mislabel that produced the false 150/mb
  reading is dead at `_lookahead_note_chunk_backward`.)

## Status

Prepared for review — **do not merge** (ship go/no-go is Jack's). Opens only
on a canary PASS under the frozen frame.
