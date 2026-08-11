# SM-SLACK RESULT — W3 gate on B300 (anchor-318g61w)
Author: curie · Date: 2026-08-10 · Rule: SM_SLACK_PREREG_318g61w.md (filed pre-derivation)
Toolchain: trace_processor v57.2 local; extraction SQL + /tmp/sm_slack.py; no sampling.

## VERDICT: HEADROOM — W3-v3 STAYS ALIVE (both ranks, same side of band)
Decision-rule evaluation (primary ALL-kernel cut):
| metric | rank0 | rank8 | HEADROOM threshold | fired |
|---|---|---|---|---|
| occ_w | 15.16% | 15.96% | <= 60% | YES |
| f_low50 | 84.26% | 83.47% | >= 25% | YES |
| f_gap | 3.27% | 1.02% | >= 10% | no (not needed; OR-rule) |
Two independent conditions fire on both ranks => HEADROOM, not MARGINAL, not INCONCLUSIVE.

## CONFIRM vs boltzmann (independent confirmation pass)
| quantity | boltzmann (via kepler) | curie rank0 | curie rank8 | verdict |
|---|---|---|---|---|
| kernel-time <10% occ | 83.5% | 83.14% | 82.38% | MATCH (~1pp) |
| SendRecv/NCCL occupancy | 0.0% | 0.0% (all 4,010 NCCL kernels) | 0.0% | MATCH |
| bwd SM slack / step | 35.9 s | 37.3 s (time@<10%occ); 38.0 s (Σ dur·(1−occ)) | 37.8 / 38.5 s | MATCH (same magnitude; small delta from window-def) |
| 1.017 = dependency serialization | yes | conc 1.026 with f_gap 1–3%: serial back-to-back, not gaps | same | CONSISTENT |
=> CONFIRM. No conflict to escalate.

## Full numbers
Window: 45,178.7 ms (rank0) / 45,181.0 ms (rank8) — span of 312 CheckpointFunctionBackward cpu_ops.
Kernels in window: 180,941 (r0) / 180,947 (r8); occupancy arg present on 100%.

ALL kernels:  kernel-time 44.82s (r0) / 45.86s (r8); occ_w 15.2/16.0%;
  deciles (r0): [0-10%: 83.1% of time] … [90-100%: 13.3%].
COMPUTE cut:  17.15s (r0) / 18.57s (r8); occ_w 39.6/39.4%; f_lt10 55.9/56.5%;
  bimodal: 34.7% of compute time in 90-100% decile (GEMM/attention full-width),
  56% at <10% occ (elementwise/copy/small kernels). grid_w: 95.6% of compute
  time launches >=148 blocks — low occ is regs/smem-limited blocks/SM, not
  undersized grids.
NCCL cut:     27.67s (r0) / 27.29s (r8) = 61.7/59.5% of bwd kernel-time;
  occ_w 0.0%, all grids <148 blocks. Comm windows are SM-empty.

## Interpretation for W3-v3 (context, not gate)
- The bwd phase is comm-dominated (≈60% NCCL-time at 0% occupancy) and the
  compute fraction is bimodal. Co-schedulable SM slots exist both in comm
  windows and in the low-occupancy compute tail.
- 1.017-1.026 concurrency with tiny gaps = wait_stream dependency serialization,
  not SM saturation: kernels reserve few SM slots but are serialized by stream
  deps. A priority-stream / per-kick-sync design attacks exactly this.
- Caveat (fit-for-purpose note): `est. achieved occupancy %` is a launch-config
  theoretical-occupancy estimate — it measures SLOT availability, which is the
  binding constraint for co-residency. It does not measure HBM bandwidth
  contention; a co-scheduled recompute kernel still competes for bandwidth.
  => HBM-bandwidth contention watch belongs in the W3-v3 canary frame (as
  helmholtz already scoped).
- Provenance: occupancy arg stored as int_value; 0 NULL rows; extraction
  deterministic; raw CSVs at /tmp/bwd_kernels_rank{0,8}.csv.

## Next duty (per helmholtz)
W3-v3 canary acceptance frame — pre-register once fermi's design spec exists:
kick ordering by input-dependency, overlap capture %, HBM-bandwidth contention
watch, memory bars incl. the ~14 GiB allocator-retention knob.
