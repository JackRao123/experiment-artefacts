# PRE-REGISTRATION — SM-Slack Measurement (W3 gate on B300)
Author: curie (verification lane, ex-boltzmann) · Date: 2026-08-10 · Status: REGISTERED before any derivation

## Question
Does the B300 backward phase have SM-occupancy headroom for concurrent recompute
kernels, or do bwd-phase kernels run full-width? (Baseline kernel concurrency
measured at 1.017 on the anchor trace suggests the latter.)

Gate: NO HEADROOM => W3-on-B300 CLOSES permanently.
      HEADROOM     => W3-v3 design (priority-stream / per-kick sync) stays alive.
Scope (helmholtz update): this measurement gates W3-v3 vs W3-closure ONLY.
It does NOT gate option-6; result is shared with fermi as W3-v3 context.

## Subject data (EXISTING traces, no new boots)
- Primary:   anchor-318g61w/rank0_131k-d4-steady.pt.trace.json  (199,804 kernels)
- Confirm:   anchor-318g61w/rank8_131k-d4-steady.pt.trace.json
- Device (from trace header): "NVIDIA L20D", compute 10.3 (Blackwell Ultra / B300),
  148 SMs, 2048 max threads/SM, 64K regs/SM, 233472 B smem/SM, 288 GB.
- Per-kernel args present on 100% of kernel slices: grid[0..2], block[0..2],
  registers per thread, shared memory, blocks per SM, warps per SM,
  est. achieved occupancy %, stream.

## Definitions
- Analysis window: ProfilerStep#0 (49,713 ms steady-state step on rank0).
- BWD WINDOW (primary): [min(ts), max(ts+dur)] over all user_annotation slices
  named 'CheckpointFunctionBackward' (n=312 on rank0). A kernel slice belongs to
  the bwd window iff its ts falls inside the window.
- Sensitivity definition S1: kernel belongs iff its [ts, ts+dur] intersects the
  UNION of the 312 CheckpointFunctionBackward intervals (excludes pre/post-bwd
  stragglers: loss, embedding bwd, final grad-sync tail).
- Kernel classes: NCCL = name LIKE 'ncclDevKernel%'; COMPUTE = all others.
  gpu_memcpy/gpu_memset excluded from occupancy metrics (no SM occupancy args
  semantics), reported separately if material.

## Metrics (computed identically on rank0 and rank8)
1. occ_w   — duration-weighted mean of `est. achieved occupancy %` over bwd-window
             kernels (ALL classes, and COMPUTE-only cut).
2. f_low50 — fraction of bwd kernel-duration with occupancy < 50%.
3. f_low25 — fraction of bwd kernel-duration with occupancy < 25%.
4. f_gap   — fraction of bwd-window WALL time with zero kernels resident
             (temporal gaps), computed via interval union of kernel slices.
5. conc    — mean kernel concurrency in window = SUM(dur) / covered-wall-time
             (sanity vs the 1.017 baseline figure).
6. Occupancy decile histogram of bwd kernel-duration (0-10%, ..., 90-100%).
7. grid_w  — fraction of bwd kernel-duration with grid[0]*grid[1]*grid[2] >= 148
             (enough blocks for >=1 wave over 148 SMs; width cross-check
             independent of the occupancy arg's provenance).

## Decision rule (fixed; no post-hoc adjustment)
- HEADROOM     if occ_w <= 60%  OR  f_gap >= 10%  OR  f_low50 >= 25%.
- NO HEADROOM  if occ_w >= 75%  AND f_gap <  5%   AND f_low50 <= 10%.
- Otherwise MARGINAL: report full distribution; verdict = insufficient evidence;
  escalate to helmholtz with numbers (no self-authorized re-thresholding).
- Rank-8 must land on the same side of the marginal band as rank-0, else verdict
  = INCONCLUSIVE (cross-rank divergence); report both.
- NCCL-class kernels are INCLUDED in the primary all-kernel metrics (a recompute
  kernel co-scheduled into a comm window is exactly the W3-v3 win), and ALSO
  reported as a COMPUTE-only cut for interpretation. The gate reads the
  all-kernel numbers; the compute-only cut is context, not a second gate.

## Invalidation (INVALID != FAIL; INVALID => re-derive, no verdict)
- Trace fails to load, or kernel count < 100k, or zero CheckpointFunctionBackward
  slices, or occupancy args missing on >5% of kernels => INVALID for that rank;
  fall back to the other rank; if both invalid => INVALID overall, report to
  helmholtz, no W3 verdict from this artifact set.

## Queries (exact SQL to be run, trace_processor v57.2 local)
Q1 window:      SELECT MIN(ts), MAX(ts+dur) FROM slice
                WHERE category='user_annotation' AND name='CheckpointFunctionBackward';
Q2 per-kernel:  SELECT s.name, s.ts, s.dur,
                  (SELECT double_value FROM args a WHERE a.arg_set_id=s.arg_set_id
                     AND a.key='args.est. achieved occupancy %') AS occ,
                  (SELECT int_value FROM args a WHERE a.arg_set_id=s.arg_set_id
                     AND a.key='args.grid[0]') AS g0,
                  (SELECT int_value FROM args a WHERE a.arg_set_id=s.arg_set_id
                     AND a.key='args.grid[1]') AS g1,
                  (SELECT int_value FROM args a WHERE a.arg_set_id=s.arg_set_id
                     AND a.key='args.grid[2]') AS g2
                FROM slice s WHERE s.category='kernel' AND s.dur>0
                  AND s.ts BETWEEN :w0 AND :w1;
  (post-processed in Python for occ_w, f_low50, f_low25, deciles, grid_w, conc,
   f_gap via sorted-interval union; deterministic, no sampling)
Q3 sensitivity: same as Q2 with membership by interval-union intersection.

## Pre-registered outputs
One results table (rank0 + rank8 side by side), verdict line, and the decile
histogram. To helmholtz (gate) and fermi (W3-v3 context; NOT an option-6 gate).
