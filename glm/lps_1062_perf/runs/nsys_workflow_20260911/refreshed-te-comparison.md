# Matched Nsight comparison

Left: devbox-ep1-te-repeat; right: devbox-ep8-te.

Headline TPS comes from unprofiled controls. Trace categories below are attribution, not headline timing.

| Case | Controls | Mean FB seconds | TPS/GPU | Peak allocated GiB |
|---|---:|---:|---:|---:|
| devbox-ep1-te-repeat | 5 | 11.8373 | 1384.1 | 248.10 |
| devbox-ep8-te | 5 | 10.5694 | 1550.1 | 213.62 |

## Per-rank reconciliation

All deltas are right minus left, milliseconds per profiled FB.

Exclusive categories + mixed-category overlap + GPU idle exactly reconcile to traced step time.

| Rank | Step delta | Expert GEMM exclusive delta | Dispatcher exclusive delta | Other exclusive delta | Mixed overlap delta | Idle delta | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | -1747.31 | 54.90 | 2579.56 | -60.34 | -3099.65 | -1221.77 | 0.00 |
| 1 | -1774.66 | 140.39 | 2508.97 | 245.89 | -3212.63 | -1457.28 | 0.00 |
| 2 | -1747.78 | 130.83 | 2556.17 | -142.44 | -3190.05 | -1102.29 | 0.00 |
| 3 | -1754.64 | 193.23 | 2630.75 | -129.62 | -3218.14 | -1230.86 | -0.00 |
| 4 | -1786.02 | 127.50 | 2543.60 | 52.77 | -3227.35 | -1282.55 | -0.00 |
| 5 | -1742.14 | 95.81 | 2549.57 | -75.78 | -3278.04 | -1033.70 | 0.00 |
| 6 | -1733.41 | 277.73 | 2304.88 | 358.76 | -3215.29 | -1459.49 | 0.00 |
| 7 | -1728.94 | 85.70 | 2518.05 | -39.84 | -3312.29 | -980.56 | 0.00 |

## Expert execution and attention (overlap included)

Do not infer equal GEMM execution time from equal exclusive time: FSDP can overlap a slower GEMM.

| Rank | Left expert GPU ms | Right expert GPU ms | Left attention GPU ms | Right attention GPU ms | Left idle in expert host scopes | Right idle in expert host scopes |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2119.51 | 1244.12 | 4413.63 | 4019.79 | 202.14 | 2.44 |
| 1 | 2167.70 | 1300.68 | 4587.00 | 4202.02 | 268.07 | 3.00 |
| 2 | 2140.78 | 1275.58 | 4574.29 | 4157.67 | 189.03 | 4.04 |
| 3 | 2149.41 | 1288.75 | 4558.56 | 4155.32 | 314.54 | 3.62 |
| 4 | 2181.74 | 1243.41 | 4553.08 | 4201.76 | 319.56 | 2.85 |
| 5 | 2131.95 | 1226.26 | 4546.23 | 4194.27 | 182.15 | 2.50 |
| 6 | 2200.65 | 1443.50 | 4534.86 | 4183.00 | 460.77 | 2.61 |
| 7 | 2154.75 | 1256.43 | 4599.05 | 4206.46 | 163.11 | 2.61 |

## Interpretation limits

- This reconciliation is an observed wall-time partition, not a causal estimate of removable time.
- Expert GEMM path includes kernel-side helpers; CPU preparation is reflected in scopes and idle, not counted as GEMM GPU work.
- A matched TE/grouped-MM ablation is needed to measure the benefit of the proposed GEMM change.
- FSDP EP8 on eight GPUs has expert-DP size one, whereas EP1 has expert-DP size eight.
- Compare loss/gradients and profiler slowdown before accepting a performance conclusion.
