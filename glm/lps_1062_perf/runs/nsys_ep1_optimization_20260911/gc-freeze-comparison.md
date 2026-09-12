# Matched Nsight comparison

Left: devbox-ep1-te-repeat; right: devbox-ep1-te-gcfreeze.

Headline TPS comes from unprofiled controls. Trace categories below are attribution, not headline timing.

| Case | Controls | Mean FB seconds | TPS/GPU | Peak allocated GiB |
|---|---:|---:|---:|---:|
| devbox-ep1-te-repeat | 5 | 11.8373 | 1384.1 | 248.10 |
| devbox-ep1-te-gcfreeze | 5 | 11.7847 | 1390.3 | 248.10 |

Source-change caveat: Baseline predates the grouped-MM asynchronous offset fix. The GC case contains that fix, but both cases use TE (grouped-MM disabled), so the changed kernel path is inactive.
This is not a same-revision A/B; instrumentation overhead is a possible confound.

Runtime ablation (gc_freeze_after_warmup): Enable existing GC freezing after the first warmup optimizer; TE and FSDP execution settings otherwise unchanged.

## Per-rank reconciliation

All deltas are right minus left, milliseconds per profiled FB.

Exclusive categories + mixed-category overlap + GPU idle exactly reconcile to traced step time.

| Rank | Step delta | Expert GEMM exclusive delta | Dispatcher exclusive delta | Other exclusive delta | Mixed overlap delta | Idle delta | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 124.74 | -47.91 | -77.69 | -335.67 | -19.11 | 605.12 | 0.00 |
| 1 | 76.00 | 1.27 | 0.20 | -165.67 | 2.73 | 237.47 | 0.00 |
| 2 | 85.54 | 71.71 | 79.02 | 109.11 | 42.96 | -217.26 | -0.00 |
| 3 | 94.54 | 114.41 | 134.46 | 70.06 | 61.73 | -286.12 | -0.00 |
| 4 | 58.12 | 44.35 | 26.42 | 39.47 | -6.57 | -45.56 | -0.00 |
| 5 | 111.40 | 15.89 | 0.17 | 94.03 | 31.98 | -30.67 | 0.00 |
| 6 | 97.24 | -30.19 | 0.13 | 525.12 | 125.34 | -523.17 | 0.00 |
| 7 | 108.59 | 67.29 | 79.68 | 40.42 | 4.05 | -82.85 | 0.00 |

## Expert execution and attention (overlap included)

Do not infer equal GEMM execution time from equal exclusive time: FSDP can overlap a slower GEMM.

| Rank | Left expert GPU ms | Right expert GPU ms | Left attention GPU ms | Right attention GPU ms | Left idle in expert host scopes | Right idle in expert host scopes |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2119.51 | 2176.82 | 4413.63 | 4415.34 | 202.14 | 355.15 |
| 1 | 2167.70 | 2163.36 | 4587.00 | 4590.26 | 268.07 | 219.70 |
| 2 | 2140.78 | 2118.97 | 4574.29 | 4579.36 | 189.03 | 96.16 |
| 3 | 2149.41 | 2105.34 | 4558.56 | 4565.20 | 314.54 | 134.12 |
| 4 | 2181.74 | 2175.14 | 4553.08 | 4566.84 | 319.56 | 292.22 |
| 5 | 2131.95 | 2142.52 | 4546.23 | 4555.25 | 182.15 | 158.63 |
| 6 | 2200.65 | 2137.48 | 4534.86 | 4531.35 | 460.77 | 167.11 |
| 7 | 2154.75 | 2146.13 | 4599.05 | 4593.88 | 163.11 | 114.99 |

## Interpretation limits

- This reconciliation is an observed wall-time partition, not a causal estimate of removable time.
- Expert GEMM path includes kernel-side helpers; CPU preparation is reflected in scopes and idle, not counted as GEMM GPU work.
- A matched TE/grouped-MM ablation is needed to measure the benefit of the proposed GEMM change.
- FSDP EP8 on eight GPUs has expert-DP size one, whereas EP1 has expert-DP size eight.
- Compare loss/gradients and profiler slowdown before accepting a performance conclusion.
