# Matched Nsight comparison

Left: devbox-ep1-grouped-async; right: devbox-ep8-grouped-async.

Headline TPS comes from unprofiled controls. Trace categories below are attribution, not headline timing.

| Case | Controls | Mean FB seconds | TPS/GPU | Peak allocated GiB |
|---|---:|---:|---:|---:|
| devbox-ep1-grouped-async | 5 | 11.8737 | 1379.9 | 247.98 |
| devbox-ep8-grouped-async | 5 | 11.2000 | 1462.9 | 213.49 |

## Per-rank reconciliation

All deltas are right minus left, milliseconds per profiled FB.

Exclusive categories + mixed-category overlap + GPU idle exactly reconcile to traced step time.

| Rank | Step delta | Expert GEMM exclusive delta | Dispatcher exclusive delta | Other exclusive delta | Mixed overlap delta | Idle delta | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | -944.32 | 372.90 | 2942.15 | -116.13 | -3660.11 | -483.14 | 0.00 |
| 1 | -941.43 | 370.36 | 2806.29 | 187.53 | -3723.86 | -581.75 | -0.00 |
| 2 | -953.80 | 287.94 | 2769.31 | -255.11 | -3565.87 | -190.07 | 0.00 |
| 3 | -958.83 | 342.45 | 2756.63 | -316.45 | -3552.17 | -189.29 | 0.00 |
| 4 | -954.32 | 360.87 | 2833.59 | -178.42 | -3765.08 | -205.28 | -0.00 |
| 5 | -949.15 | 335.24 | 2917.33 | -238.97 | -3782.45 | -180.30 | 0.00 |
| 6 | -945.84 | 521.17 | 2477.52 | -106.82 | -3656.66 | -181.05 | 0.00 |
| 7 | -948.41 | 199.85 | 2724.32 | -174.95 | -3514.76 | -182.88 | -0.00 |

## Expert execution and attention (overlap included)

Do not infer equal GEMM execution time from equal exclusive time: FSDP can overlap a slower GEMM.

| Rank | Left expert GPU ms | Right expert GPU ms | Left attention GPU ms | Right attention GPU ms | Left idle in expert host scopes | Right idle in expert host scopes |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2414.53 | 1422.16 | 4697.16 | 4054.83 | 4.11 | 0.04 |
| 1 | 2439.41 | 1464.03 | 4889.64 | 4246.13 | 1.50 | 0.10 |
| 2 | 2351.64 | 1433.46 | 4799.14 | 4177.94 | 0.49 | 0.06 |
| 3 | 2326.47 | 1474.03 | 4791.84 | 4185.52 | 0.41 | 0.14 |
| 4 | 2391.10 | 1409.40 | 4776.23 | 4232.31 | 4.49 | 0.08 |
| 5 | 2396.67 | 1364.58 | 4806.63 | 4228.07 | 3.07 | 0.12 |
| 6 | 2343.16 | 1647.90 | 4722.52 | 4219.34 | 0.43 | 0.17 |
| 7 | 2387.96 | 1405.24 | 4874.87 | 4240.98 | 0.49 | 0.13 |

## Interpretation limits

- This reconciliation is an observed wall-time partition, not a causal estimate of removable time.
- Expert GEMM path includes kernel-side helpers; CPU preparation is reflected in scopes and idle, not counted as GEMM GPU work.
- A matched TE/grouped-MM ablation is needed to measure the benefit of the proposed GEMM change.
- FSDP EP8 on eight GPUs has expert-DP size one, whereas EP1 has expert-DP size eight.
- Compare loss/gradients and profiler slowdown before accepting a performance conclusion.
