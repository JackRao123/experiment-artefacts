# Matched Nsight comparison

Left: devbox-ep1-grouped; right: devbox-ep8-grouped.

Headline TPS comes from unprofiled controls. Trace categories below are attribution, not headline timing.

| Case | Controls | Mean FB seconds | TPS/GPU | Peak allocated GiB |
|---|---:|---:|---:|---:|
| devbox-ep1-grouped | 5 | 11.7888 | 1389.8 | 247.98 |
| devbox-ep8-grouped | 5 | 11.3280 | 1446.3 | 213.49 |

## Per-rank reconciliation

All deltas are right minus left, milliseconds per profiled FB.

Exclusive categories + mixed-category overlap + GPU idle exactly reconcile to traced step time.

| Rank | Step delta | Expert GEMM exclusive delta | Dispatcher exclusive delta | Other exclusive delta | Mixed overlap delta | Idle delta | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | -598.07 | 231.11 | 2860.34 | 445.43 | -3305.65 | -829.30 | 0.00 |
| 1 | -571.39 | 262.88 | 2727.74 | 646.50 | -3425.88 | -782.63 | 0.00 |
| 2 | -569.51 | 243.10 | 2796.22 | 365.11 | -3309.95 | -664.00 | 0.00 |
| 3 | -571.67 | 335.46 | 2855.98 | 219.37 | -3360.50 | -621.98 | 0.00 |
| 4 | -561.57 | 295.27 | 2876.28 | 329.09 | -3529.93 | -532.28 | -0.00 |
| 5 | -571.69 | 210.81 | 2887.05 | 337.18 | -3429.77 | -576.97 | 0.00 |
| 6 | -551.47 | 474.66 | 2490.76 | 390.26 | -3377.29 | -529.86 | -0.00 |
| 7 | -556.80 | 235.21 | 2864.47 | 380.47 | -3495.75 | -541.20 | -0.00 |

## Expert execution and attention (overlap included)

Do not infer equal GEMM execution time from equal exclusive time: FSDP can overlap a slower GEMM.

| Rank | Left expert GPU ms | Right expert GPU ms | Left attention GPU ms | Right attention GPU ms | Left idle in expert host scopes | Right idle in expert host scopes |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2295.99 | 1418.83 | 4508.87 | 4056.55 | 110.80 | 55.23 |
| 1 | 2288.37 | 1464.92 | 4708.45 | 4252.06 | 109.61 | 56.40 |
| 2 | 2267.36 | 1432.78 | 4685.12 | 4181.23 | 106.86 | 54.70 |
| 3 | 2315.12 | 1472.47 | 4651.63 | 4186.90 | 108.02 | 55.13 |
| 4 | 2378.27 | 1403.52 | 4637.20 | 4248.99 | 107.51 | 53.17 |
| 5 | 2337.68 | 1361.69 | 4638.78 | 4209.55 | 106.70 | 55.59 |
| 6 | 2260.65 | 1648.36 | 4629.21 | 4216.73 | 106.25 | 56.94 |
| 7 | 2415.78 | 1403.56 | 4724.70 | 4238.26 | 105.03 | 54.43 |

## Interpretation limits

- This reconciliation is an observed wall-time partition, not a causal estimate of removable time.
- Expert GEMM path includes kernel-side helpers; CPU preparation is reflected in scopes and idle, not counted as GEMM GPU work.
- A matched TE/grouped-MM ablation is needed to measure the benefit of the proposed GEMM change.
- FSDP EP8 on eight GPUs has expert-DP size one, whereas EP1 has expert-DP size eight.
- Compare loss/gradients and profiler slowdown before accepting a performance conclusion.
