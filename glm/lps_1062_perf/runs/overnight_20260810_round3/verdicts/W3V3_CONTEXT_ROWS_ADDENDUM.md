# W3-v3 CANARY — CONTEXT-ROWS ADDENDUM (T5 dilation + memory decomposition)
Author: curie · Date: 2026-08-10 · Closes the reported-under-every-outcome rows.
Artifacts: incoming/w3v3_rank0.pt.trace.json (md5 8acd46a6…, verified);
incoming/w3v3_memsnap/memory.rank{0..15}.pickle (16/16, md5-matched box↔Mac).

## T5 — bwd compute-kernel dilation per occupancy bucket (arm vs anchor-318g61w)
Method: arm bwd window = span of 312 LookaheadCheckpointFunctionBackward cpu_ops
(180,927 kernels); matched to anchor bwd-window kernels BY NAME (392 compute
names, 17.15s anchor time covered); bucketed by ANCHOR occupancy decile;
dilation = per-name mean-duration ratio, duration-weighted.
MAIN streams (first-pass compute): TOTAL 1.0175. By bucket (weight = anchor
seconds): 0-10% occ: 1.0146 (9.56s); 90-100%: 1.0123 (4.52s); mid buckets
noisy at tiny weights (20-30%: 1.25 on 0.066s; 30-40%: 0.33 on 0.043s).
Side-stream-moved kernels (the kicked recompute): TOTAL 1.0190.
READING: at capture = 0.9% nothing co-runs, so true HBM-contention dilation is
definitionally ~nil; the residual ~1.7-1.9% is stack/cross-boot noise class
(arm = C′-ON+W3-v3 vs anchor = B+F — different stacks) and shows NO
occupancy-bucket concentration (the win model's full-width-block contention
signature is absent, as expected without co-running). The HBM term d at real
capture remains UNMEASURED — it is a v4 measurement, not estimable from this
arm. Quick-check cross-validation (all matched compute, no windowing):
1.0021 duration-weighted; the two material per-kernel outliers
(_permute_kernel 0.37, _sort_chunks_by_map 2.07) are MoE-pipeline kernels
whose instances mix phases differently across the two stacks — phase-mixing
artifact class, not a contention signal.

## MEMORY DECOMPOSITION (16-rank snapshot, minkowski-style segment walk)
Fleet aggregates (16/16 ranks):
- final reserved:  mean 208.85 GiB (min 202.53, max 215.05)
- final active:    mean 123.83 GiB (uniform 123.83-123.85 — params/grads/
                   optimizer/persistent; 122.1 GiB of it is pre-recording
                   blocks with no stack, as expected)
- final free-cached: mean 85.02 GiB (78.70-91.22, per-rank varying)
Peak-live reconstruction (device_traces replay; 100k-event cap noted):
- peak allocated: rank max 23.14 GiB (rank0 is the peak rank; ranks whose
  capped window missed the bwd peak read lower — min 0.00, artifact of the
  cap, not a zero working set)
- live-at-peak by call site (rank0, sums to ~22.9 GiB): forward 8.05,
  sort_chunks_by_map 5.24, glu 2.36, silu 1.21, qkv_up_proj_and_rope 1.13,
  flash_mla_sparse_fwd 1.00, get_emb 1.00, _sort_valid_topk 0.88,
  permute_with_mask_map 0.75, ... — MoE-PIPELINE SAVES DOMINATE.
READING vs the v2 structure (11.5 intrinsic + ~14 retention):
- INTRINSIC CONFIRMED: ~23.1 GiB live at peak = two ~11.4-11.5 GiB chunk
  graphs — the in-flight depth bound of 2 (spec §4) holds under v3's real
  concurrency regime, as designed. MoE-pipeline saves dominate, as in v2.
- RETENTION: total free-cached 85.0 GiB mean is ALL-POOLS (the segment stream
  fields are raw pointers; a side-pool-only split is not recoverable from
  these artifacts) and measured at the memsnap windows (later than the canary
  drive). Not comparable to v2's side-pool-only 26-30 GiB figure; the
  poller-delta (+24.9 GiB, method-matched) remains the M1 bar number.
  trims==0 confirmed (knob OFF at 131k).
- Poller-vs-snapshot non-identity stands as recorded (different windows;
  ~9.4 GiB = driver-used vs allocator-reserved overhead class).

## ROW LEDGER (final)
T1 FAIL (0.9% >= 50% bar) — verdict-carrying. T2 PASS (99.5% in-window).
T3 NOT-MEASURABLE on this artifact (no GPU-track annotations; tooling need
logged for the morning capture recipe). T5 reported above (d ~ noise at
capture 0.9%; no bucket concentration). T6 reported separately (the
v4-or-close discriminator). L1 PASS-by-re-derivation (74+1/mb exact).
L2 PASS exact. L3 PASS (sweeps==0, fallbacks==0 — grothendieck's boot-41
line). L4 PASS-by-intent (steady band 43.5-47.8ms; cumulative 59.7<60 at w7;
2947ms recurring max = structural wait). M1 PASS (+24.9 <= +28 GiB; headroom
20.0 >= 10). M2 PASS (trims==0). N1 PASS (w0 +0.1e-3, mains <=0.7e-3).
N2 respected (131k only).
