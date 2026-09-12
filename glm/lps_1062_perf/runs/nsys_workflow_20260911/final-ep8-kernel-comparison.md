# Matched Nsight comparison

Left: devbox-ep8-te; right: devbox-ep8-grouped-async.

Headline TPS comes from unprofiled controls. Trace categories below are attribution, not headline timing.

| Case | Controls | Mean FB seconds | TPS/GPU | Peak allocated GiB |
|---|---:|---:|---:|---:|
| devbox-ep8-te | 5 | 10.5694 | 1550.1 | 213.62 |
| devbox-ep8-grouped-async | 5 | 11.2000 | 1462.9 | 213.49 |

Source-change caveat: TE versus optional grouped-MM with the committed asynchronous-offset fix; other source behavior and instrumentation unchanged.
This is not a same-revision A/B; instrumentation overhead is a possible confound.

## Per-rank reconciliation

All deltas are right minus left, milliseconds per profiled FB.

Exclusive categories + mixed-category overlap + GPU idle exactly reconcile to traced step time.

| Rank | Step delta | Expert GEMM exclusive delta | Dispatcher exclusive delta | Other exclusive delta | Mixed overlap delta | Idle delta | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 587.62 | 178.03 | 260.97 | 126.46 | -6.65 | 28.81 | -0.00 |
| 1 | 575.61 | 163.35 | 274.39 | 127.50 | -2.64 | 13.02 | 0.00 |
| 2 | 577.80 | 157.88 | 296.28 | 129.66 | 1.32 | -7.34 | 0.00 |
| 3 | 579.54 | 185.28 | 263.72 | 123.88 | -0.36 | 7.01 | 0.00 |
| 4 | 571.08 | 165.99 | 291.81 | 149.18 | -0.12 | -35.78 | -0.00 |
| 5 | 569.99 | 138.31 | 313.81 | 136.72 | -3.20 | -15.64 | 0.00 |
| 6 | 557.53 | 204.40 | 254.54 | 153.33 | 1.78 | -56.52 | -0.00 |
| 7 | 577.03 | 148.81 | 291.99 | 130.46 | -3.32 | 9.09 | -0.00 |

## Expert execution and attention (overlap included)

Do not infer equal GEMM execution time from equal exclusive time: FSDP can overlap a slower GEMM.

| Rank | Left expert GPU ms | Right expert GPU ms | Left attention GPU ms | Right attention GPU ms | Left idle in expert host scopes | Right idle in expert host scopes |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1244.12 | 1422.16 | 4019.79 | 4054.83 | 2.44 | 0.04 |
| 1 | 1300.68 | 1464.03 | 4202.02 | 4246.13 | 3.00 | 0.10 |
| 2 | 1275.58 | 1433.46 | 4157.67 | 4177.94 | 4.04 | 0.06 |
| 3 | 1288.75 | 1474.03 | 4155.32 | 4185.52 | 3.62 | 0.14 |
| 4 | 1243.41 | 1409.40 | 4201.76 | 4232.31 | 2.85 | 0.08 |
| 5 | 1226.26 | 1364.58 | 4194.27 | 4228.07 | 2.50 | 0.12 |
| 6 | 1443.50 | 1647.90 | 4183.00 | 4219.34 | 2.61 | 0.17 |
| 7 | 1256.43 | 1405.24 | 4206.46 | 4240.98 | 2.61 | 0.13 |

## Interpretation limits

- This reconciliation is an observed wall-time partition, not a causal estimate of removable time.
- Expert GEMM path includes kernel-side helpers; CPU preparation is reflected in scopes and idle, not counted as GEMM GPU work.
- A matched TE/grouped-MM ablation is needed to measure the benefit of the proposed GEMM change.
- FSDP EP8 on eight GPUs has expert-DP size one, whereas EP1 has expert-DP size eight.
- Compare loss/gradients and profiler slowdown before accepting a performance conclusion.
