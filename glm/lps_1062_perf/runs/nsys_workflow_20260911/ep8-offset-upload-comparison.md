# Matched Nsight comparison

Left: devbox-ep8-grouped; right: devbox-ep8-grouped-async.

Headline TPS comes from unprofiled controls. Trace categories below are attribution, not headline timing.

| Case | Controls | Mean FB seconds | TPS/GPU | Peak allocated GiB |
|---|---:|---:|---:|---:|
| devbox-ep8-grouped | 5 | 11.3280 | 1446.3 | 213.49 |
| devbox-ep8-grouped-async | 5 | 11.2000 | 1462.9 | 213.49 |

Source-change caveat: Only frozen expert offset upload changed: pinned CPU staging and nonblocking H2D, same GPU cumsum/GEMM/routing. Parent pins updated; NVTX implementation unchanged.
This is not a same-revision A/B; instrumentation overhead is a possible confound.

## Per-rank reconciliation

All deltas are right minus left, milliseconds per profiled FB.

Exclusive categories + mixed-category overlap + GPU idle exactly reconcile to traced step time.

| Rank | Step delta | Expert GEMM exclusive delta | Dispatcher exclusive delta | Other exclusive delta | Mixed overlap delta | Idle delta | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | -73.42 | 3.33 | -22.40 | -21.69 | -3.47 | -29.18 | 0.00 |
| 1 | -81.10 | -0.89 | -25.31 | -15.28 | 1.64 | -41.26 | 0.00 |
| 2 | -79.22 | 0.68 | -24.26 | -9.48 | 0.05 | -46.22 | 0.00 |
| 3 | -77.15 | 1.56 | -18.08 | -16.59 | -0.13 | -43.91 | -0.00 |
| 4 | -85.84 | 5.88 | -11.97 | -18.78 | -2.40 | -58.58 | 0.00 |
| 5 | -82.06 | 2.89 | -24.55 | 2.63 | -3.67 | -59.37 | 0.00 |
| 6 | -82.65 | -0.46 | -11.67 | -12.10 | -1.22 | -57.20 | 0.00 |
| 7 | -82.92 | 1.68 | -29.17 | -7.80 | -6.40 | -41.24 | -0.00 |

## Expert execution and attention (overlap included)

Do not infer equal GEMM execution time from equal exclusive time: FSDP can overlap a slower GEMM.

| Rank | Left expert GPU ms | Right expert GPU ms | Left attention GPU ms | Right attention GPU ms | Left idle in expert host scopes | Right idle in expert host scopes |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1418.83 | 1422.16 | 4056.55 | 4054.83 | 55.23 | 0.04 |
| 1 | 1464.92 | 1464.03 | 4252.06 | 4246.13 | 56.40 | 0.10 |
| 2 | 1432.78 | 1433.46 | 4181.23 | 4177.94 | 54.70 | 0.06 |
| 3 | 1472.47 | 1474.03 | 4186.90 | 4185.52 | 55.13 | 0.14 |
| 4 | 1403.52 | 1409.40 | 4248.99 | 4232.31 | 53.17 | 0.08 |
| 5 | 1361.69 | 1364.58 | 4209.55 | 4228.07 | 55.59 | 0.12 |
| 6 | 1648.36 | 1647.90 | 4216.73 | 4219.34 | 56.94 | 0.17 |
| 7 | 1403.56 | 1405.24 | 4238.26 | 4240.98 | 54.43 | 0.13 |

## Interpretation limits

- This reconciliation is an observed wall-time partition, not a causal estimate of removable time.
- Expert GEMM path includes kernel-side helpers; CPU preparation is reflected in scopes and idle, not counted as GEMM GPU work.
- A matched TE/grouped-MM ablation is needed to measure the benefit of the proposed GEMM change.
- FSDP EP8 on eight GPUs has expert-DP size one, whereas EP1 has expert-DP size eight.
- Compare loss/gradients and profiler slowdown before accepting a performance conclusion.
