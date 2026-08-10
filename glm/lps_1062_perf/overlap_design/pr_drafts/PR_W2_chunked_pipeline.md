# PRE-DRAFTED PR BODY — W2 chunked MoE A2A pipeline (mcore)

**Repo:** basetenlabs/Megatron-LM · **Branch:** `jackrao/lps-1062-ship-w2`
(to be cut at the W2-v3 commit 09c1c5cff, stacked on `jackrao/lps-1062-ship-w1`)
· **Base:** `jackrao/lps-1062-ship-w1` (stacked PR #28)
**OPEN CONDITION (helmholtz):** only after the W2 timed arm (T3 canary slot)
PASSES. The gate is already clean; the wall/mechanism verdict is pending.
Do NOT open before helmholtz confirms.

> **⚠ STATUS CAVEAT (2026-08-10 — PR STAYS CLOSED):** the W2 timed arm
> **FAILED BY HANG** in the first window's backward (warmup0, box 3) — NCCL
> collective-timeout after ~7 min. First-pass analysis:
> [W2_ARM_HANG_ANALYSIS_20260810.md](../W2_ARM_HANG_ANALYSIS_20260810.md).
> Headline: (i) the failure is a stack-composition class, not a
> gate-regression — the T2 gate passed at its scale and the W2 backward's
> collective order is structural (not data-dependent) at code level; (ii) the
> #28-class topology suspect (unguarded W1 `new_group` on the stack branch)
> is expected dead — the reference arm ran clean with W1 ACTIVE and warmup0's
> forward completed (900 probs A2As on the W1 comm) before the backward hang;
> (iii) front-runner: a FIX-C replay-restore divergence at full shape (ranks
> disagree on chunk plans ⇒ mismatched A2A sizes ⇒ collective hang in the
> first checkpoint backward's recompute) — the discriminating experiment is
> the W2+C′ VERIFY=1 full-shape soak (morning item, needs a box slot). **Do
> not open this PR until the hang is root-caused and a re-arm passes.**

---

## What

`BT_MOE_A2A_PIPELINE=2` (default OFF): intra-MoE-layer all-to-all ⇄ compute
pipelining for the alltoall dispatcher. The 16 local experts split into K=2
groups of 8 (group g = local experts [8g, 8g+8) on EVERY rank); dispatch and
combine A2As become K list-form `torch.distributed.all_to_all` calls over
per-peer contiguous **views** of the existing buffers — zero copies, zero
layout changes — so chunk 1's dispatch overlaps chunk 0's expert GEMMs and
chunk 0's combine overlaps chunk 1's. The final unpermute sees a
byte-identical buffer in byte-identical order (the combine writes land at
today's per-src offsets).

Variant note (from the design memo): row-chunking the permuted matrix
(variant (i)) is provably unsound — expert-major layout makes row chunks
destination-rank-contiguous; expert-group chunking (ii) is the only sound
intra-layer decomposition, and list-A2A views make it free.

## Why (win model)

- v1 (this patch): fwd+replay pipelining ≈ **−3.0 s/step** at 131k×d4
  (−5.1 ms/layer-pass × 600); bwd ≈ neutral (issue+wait backward).
- The v2 backward seq-bump extension (−1.2…−1.5 s more) is **not in this
  patch** — its disposition is pre-registered in
  [W2V2_DECISION_stub.md](https://github.com/JackRao123/experiment-artefacts/blob/main/glm/lps_1062_perf/overlap_design/W2V2_DECISION_stub.md)
  (un-parked, sequenced behind W3-v3 per the subsumption note).
- De-skew upside (dispatch p50 10.7 ms vs 4.3 ms floor) is on top of the
  static math, not booked.
- Timed-arm result: *fills at open time (T3 canary slot, house band +
  20-step drift cover).*

## Numerics (the gate story — read this first)

- **T2 gate ALL PASS** on the canonical gate `d88d8b7d` (2026-08-09/10):
  outputs AND input grads **bitwise** across all four routing cases
  (balanced / imbalance / zero-expert / zero-peer-group), checkpointing
  on/off, and FIX-C composition variants. The zero-padded grouped-GEMM dgrad
  schedule question (R1 — the only residual non-bitwise risk, GEMM kernel
  schedule vs `num_gemms` 16→8) resolved **NEGATIVE** at gate scale.
- **The 0626 "gate FAIL" was two harness artifacts, both root-caused and
  sealed** (full post-mortem in
  [ESTATE_NOTES_minkowski.md](https://github.com/JackRao123/experiment-artefacts/blob/main/glm/lps_1062_perf/overlap_design/ESTATE_NOTES_minkowski.md),
  2026-08-09 T2 entries): (1) `craft_routing` collapsed the engineered cases
  below topk selections/token (bool-mask duplicates) → permute pads to the
  static T·topk → A2A split mismatch crash — the gate had never run cases
  2–4 before; (2) `run_once` drew `grad_out` from the global RNG per call,
  so the A/B/C comparison backwarded against different upstream grads — the
  grad assertion was **vacuous from authoring** (an assertion that never ran
  is indistinguishable from one that passed). The fixed harness adds
  `_delta_stats` mismatch diagnostics so a future grad mismatch produces the
  noise-class-vs-systematic evidence in-run.
- **Fidelity caveat (stated, not hidden):** the ALL-PASS re-run ran at
  BT_TEST_HIDDEN 2048 / seq 8192 gate scale. The 6144 full-fidelity
  confirmation is queued on an idle box window (belt-and-suspenders under
  Jack's variance-class bar — the ship bar is "optimized-vs-default
  difference ≈ run-to-run variance of default", not bitwise; bitwise stays
  the in-process diagnostic gold tier).
- FP8: per-expert `Fp8Padding` alignment is unchanged per expert; every
  128-row block boundary lands on the same rows → quantized inputs bitwise →
  GEMM bitwise (memo §5b).

## Memory / host cost

≈ 0 GiB: recv chunks are halves of today's recv footprint; all per-peer
tensors are views; events are µs-scale. Host: per-(group,peer) offset
template precomputed at init; per-pass only counts fill (<100 µs/pass
budget, binding — measured in the canary).

## Composition

- **W1** armed: the single full probs A2A rides the second communicator
  concurrent with the token chunks; without W1 it rides the EP comm issued
  FIRST. Per-group probs selected by one fused `sort_chunks` gather
  (`probs_keep_idxs`, precomputed at arm time).
- **FIX C**: the two count matrices ride the SAME single D2H batch + single
  event as today (no new syncs) and are cached across the recompute replay
  via two W2-gated slots on FIX C's entry, verified in
  `_verify_host_metadata`.
- Guards (loud `armed=NO — <reason>` fallbacks to the status-quo path):
  K ∤ local experts; EP=1; drop_and_pad; expert-TP>1; moe_permute_fusion
  off; CUDA graphs; fused TEGroupedMLP impl; activation offloading; paged
  stash; moe_act recompute; overlap with wgrad; probs-on-input; unfrozen
  experts.

## Evidence / tests

- Design: [DESIGN_helmholtz.md](https://github.com/JackRao123/experiment-artefacts/blob/main/glm/lps_1062_perf/overlap_design/DESIGN_helmholtz.md)
  §2–§3 (decomposition, autograd structure, streams/events, memory) +
  §5 (per-tensor bitwise argument) · patch notes:
  [W2_PATCH_NOTES.md](https://github.com/JackRao123/experiment-artefacts/blob/main/glm/lps_1062_perf/overlap_design/patches/W2_PATCH_NOTES.md)
- CPU suites:
  [test_w2_chunk_plan.py](https://github.com/JackRao123/experiment-artefacts/blob/main/glm/lps_1062_perf/overlap_design/tests/test_w2_chunk_plan.py)
  (T1 index math: per-group splits/offsets exact row-permutation of today's
  buffer; combine reassembly byte-exact),
  [test_w2_chunked_a2a_functions.py](https://github.com/JackRao123/experiment-artefacts/blob/main/glm/lps_1062_perf/overlap_design/tests/test_w2_chunked_a2a_functions.py),
  [test_w2_probs_selection_backward.py](https://github.com/JackRao123/experiment-artefacts/blob/main/glm/lps_1062_perf/overlap_design/tests/test_w2_probs_selection_backward.py)
- On-box gate: `t2_w2_numerics_gate.py` (canonical `d88d8b7d`) — re-run log
  `lps1062_bench/wxlgv5w/t2_w2_gate_rerun_d88d8b7d.log` (box 2).
- Telemetry: WARNING-level gate-state/armed lines + per-window
  `{dispatch_issues, combine_issues, waits, fallback_passes}` counters (a
  gate that can't fire is loud by absence).

## Status

Gate-clean; opens only on a timed-arm PASS. Prepared for review — **do not
merge** (ship go/no-go is Jack's).
