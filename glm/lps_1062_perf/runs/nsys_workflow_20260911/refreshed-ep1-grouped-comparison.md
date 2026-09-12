# Matched Nsight comparison

Left: devbox-ep1-te-repeat; right: devbox-ep1-grouped.

Headline TPS comes from unprofiled controls. Trace categories below are attribution, not headline timing.

| Case | Controls | Mean FB seconds | TPS/GPU | Peak allocated GiB |
|---|---:|---:|---:|---:|
| devbox-ep1-te-repeat | 5 | 11.8373 | 1384.1 | 248.10 |
| devbox-ep1-grouped | 5 | 11.7888 | 1389.8 | 247.98 |

## Per-rank reconciliation

All deltas are right minus left, milliseconds per profiled FB.

Exclusive categories + mixed-category overlap + GPU idle exactly reconcile to traced step time.

| Rank | Step delta | Expert GEMM exclusive delta | Dispatcher exclusive delta | Other exclusive delta | Mixed overlap delta | Idle delta | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | -488.20 | -1.50 | 2.59 | -357.62 | 202.81 | -334.49 | 0.00 |
| 1 | -546.56 | 41.74 | 80.93 | -257.84 | 208.98 | -620.37 | 0.00 |
| 2 | -521.25 | 44.93 | 80.48 | -368.41 | 121.16 | -399.41 | -0.00 |
| 3 | -526.28 | 41.49 | 56.58 | -208.53 | 142.13 | -557.96 | -0.00 |
| 4 | -567.53 | -7.65 | -28.90 | -108.37 | 304.86 | -727.48 | -0.00 |
| 5 | -518.39 | 20.43 | 0.88 | -278.88 | 152.19 | -413.01 | 0.00 |
| 6 | -541.75 | 7.93 | 80.33 | 133.92 | 165.01 | -928.95 | 0.00 |
| 7 | -512.20 | -2.39 | -25.27 | -282.05 | 186.54 | -389.04 | 0.00 |

## Expert execution and attention (overlap included)

Do not infer equal GEMM execution time from equal exclusive time: FSDP can overlap a slower GEMM.

| Rank | Left expert GPU ms | Right expert GPU ms | Left attention GPU ms | Right attention GPU ms | Left idle in expert host scopes | Right idle in expert host scopes |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2119.51 | 2295.99 | 4413.63 | 4508.87 | 202.14 | 110.80 |
| 1 | 2167.70 | 2288.37 | 4587.00 | 4708.45 | 268.07 | 109.61 |
| 2 | 2140.78 | 2267.36 | 4574.29 | 4685.12 | 189.03 | 106.86 |
| 3 | 2149.41 | 2315.12 | 4558.56 | 4651.63 | 314.54 | 108.02 |
| 4 | 2181.74 | 2378.27 | 4553.08 | 4637.20 | 319.56 | 107.51 |
| 5 | 2131.95 | 2337.68 | 4546.23 | 4638.78 | 182.15 | 106.70 |
| 6 | 2200.65 | 2260.65 | 4534.86 | 4629.21 | 460.77 | 106.25 |
| 7 | 2154.75 | 2415.78 | 4599.05 | 4724.70 | 163.11 | 105.03 |

## Interpretation limits

- This reconciliation is an observed wall-time partition, not a causal estimate of removable time.
- Expert GEMM path includes kernel-side helpers; CPU preparation is reflected in scopes and idle, not counted as GEMM GPU work.
- A matched TE/grouped-MM ablation is needed to measure the benefit of the proposed GEMM change.
- FSDP EP8 on eight GPUs has expert-DP size one, whereas EP1 has expert-DP size eight.
- Compare loss/gradients and profiler slowdown before accepting a performance conclusion.
