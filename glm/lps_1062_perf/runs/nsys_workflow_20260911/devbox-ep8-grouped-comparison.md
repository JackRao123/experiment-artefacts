# Matched Nsight comparison

Left: devbox-ep8-te; right: devbox-ep8-grouped.

Headline TPS comes from unprofiled controls. Trace categories below are attribution, not headline timing.

| Case | Controls | Mean FB seconds | TPS/GPU | Peak allocated GiB |
|---|---:|---:|---:|---:|
| devbox-ep8-te | 5 | 10.5694 | 1550.1 | 213.62 |
| devbox-ep8-grouped | 5 | 11.3280 | 1446.3 | 213.49 |

## Per-rank reconciliation

All deltas are right minus left, milliseconds per profiled FB.

Exclusive categories + mixed-category overlap + GPU idle exactly reconcile to traced step time.

| Rank | Step delta | Expert GEMM exclusive delta | Dispatcher exclusive delta | Other exclusive delta | Mixed overlap delta | Idle delta | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 661.04 | 174.71 | 283.38 | 148.15 | -3.18 | 57.98 | -0.00 |
| 1 | 656.71 | 164.24 | 299.70 | 142.77 | -4.28 | 54.28 | 0.00 |
| 2 | 657.02 | 157.20 | 320.53 | 139.13 | 1.27 | 38.88 | -0.00 |
| 3 | 656.69 | 183.72 | 281.81 | 140.46 | -0.22 | 50.92 | 0.00 |
| 4 | 656.92 | 160.11 | 303.78 | 167.96 | 2.28 | 22.79 | -0.00 |
| 5 | 652.06 | 135.42 | 338.35 | 134.09 | 0.47 | 43.72 | 0.00 |
| 6 | 640.18 | 204.86 | 266.21 | 165.43 | 3.00 | 0.68 | -0.00 |
| 7 | 659.95 | 147.13 | 321.16 | 138.25 | 3.08 | 50.33 | -0.00 |

## Expert execution and attention (overlap included)

Do not infer equal GEMM execution time from equal exclusive time: FSDP can overlap a slower GEMM.

| Rank | Left expert GPU ms | Right expert GPU ms | Left attention GPU ms | Right attention GPU ms | Left idle in expert host scopes | Right idle in expert host scopes |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1244.12 | 1418.83 | 4019.79 | 4056.55 | 2.44 | 55.23 |
| 1 | 1300.68 | 1464.92 | 4202.02 | 4252.06 | 3.00 | 56.40 |
| 2 | 1275.58 | 1432.78 | 4157.67 | 4181.23 | 4.04 | 54.70 |
| 3 | 1288.75 | 1472.47 | 4155.32 | 4186.90 | 3.62 | 55.13 |
| 4 | 1243.41 | 1403.52 | 4201.76 | 4248.99 | 2.85 | 53.17 |
| 5 | 1226.26 | 1361.69 | 4194.27 | 4209.55 | 2.50 | 55.59 |
| 6 | 1443.50 | 1648.36 | 4183.00 | 4216.73 | 2.61 | 56.94 |
| 7 | 1256.43 | 1403.56 | 4206.46 | 4238.26 | 2.61 | 54.43 |

## Interpretation limits

- This reconciliation is an observed wall-time partition, not a causal estimate of removable time.
- Expert GEMM path includes kernel-side helpers; CPU preparation is reflected in scopes and idle, not counted as GEMM GPU work.
- A matched TE/grouped-MM ablation is needed to measure the benefit of the proposed GEMM change.
- FSDP EP8 on eight GPUs has expert-DP size one, whereas EP1 has expert-DP size eight.
- Compare loss/gradients and profiler slowdown before accepting a performance conclusion.
