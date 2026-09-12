# Matched Nsight comparison

Left: devbox-ep8-te; right: devbox-ep8-te-metadata-v2.

Headline TPS comes from unprofiled controls. Trace categories below are attribution, not headline timing.

| Case | Controls | Mean FB seconds | TPS/GPU | Peak allocated GiB |
|---|---:|---:|---:|---:|
| devbox-ep8-te | 5 | 10.5694 | 1550.1 | 213.62 |
| devbox-ep8-te-metadata-v2 | 5 | 10.6553 | 1537.6 | 213.62 |

Source-change caveat: V2 adds host bookkeeping optimizations and FSDP/GC NVTX scopes; grouped-MM remains disabled. Both cases begin at optimizer step zero with the same input/configuration.
This is not a same-revision A/B; instrumentation overhead is a possible confound.

Runtime ablation (cache_parameter_metadata): Static FSDP metadata cache plus per-bucket wait/release deduplication; TE and other execution options unchanged.

## Per-rank reconciliation

All deltas are right minus left, milliseconds per profiled FB.

Exclusive categories + mixed-category overlap + GPU idle exactly reconcile to traced step time.

| Rank | Step delta | Expert GEMM exclusive delta | Dispatcher exclusive delta | Other exclusive delta | Mixed overlap delta | Idle delta | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | -81.11 | 0.83 | -20.57 | -30.35 | -2.93 | -28.09 | -0.00 |
| 1 | -89.97 | -1.52 | -9.19 | -33.85 | -2.12 | -43.28 | 0.00 |
| 2 | -84.51 | -0.76 | -10.10 | -21.96 | -0.53 | -51.17 | -0.00 |
| 3 | -84.56 | -0.04 | -21.73 | -28.23 | -3.01 | -31.54 | 0.00 |
| 4 | -69.90 | -1.21 | -6.92 | -19.15 | -3.26 | -39.36 | 0.00 |
| 5 | -86.63 | -0.37 | -10.42 | -18.66 | -1.89 | -55.29 | -0.00 |
| 6 | -103.50 | -2.01 | 5.60 | -12.94 | -0.02 | -94.14 | -0.00 |
| 7 | -82.83 | -1.11 | -15.63 | -20.20 | 0.80 | -46.69 | -0.00 |

## Expert execution and attention (overlap included)

Do not infer equal GEMM execution time from equal exclusive time: FSDP can overlap a slower GEMM.

| Rank | Left expert GPU ms | Right expert GPU ms | Left attention GPU ms | Right attention GPU ms | Left idle in expert host scopes | Right idle in expert host scopes |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1244.12 | 1244.95 | 4019.79 | 4014.08 | 2.44 | 2.50 |
| 1 | 1300.68 | 1299.16 | 4202.02 | 4187.90 | 3.00 | 2.92 |
| 2 | 1275.58 | 1274.82 | 4157.67 | 4161.59 | 4.04 | 2.45 |
| 3 | 1288.75 | 1288.71 | 4155.32 | 4155.91 | 3.62 | 3.74 |
| 4 | 1243.41 | 1242.19 | 4201.76 | 4199.04 | 2.85 | 3.95 |
| 5 | 1226.26 | 1225.90 | 4194.27 | 4191.37 | 2.50 | 2.27 |
| 6 | 1443.50 | 1441.49 | 4183.00 | 4176.97 | 2.61 | 2.81 |
| 7 | 1256.43 | 1255.32 | 4206.46 | 4208.68 | 2.61 | 2.45 |

## Interpretation limits

- This reconciliation is an observed wall-time partition, not a causal estimate of removable time.
- Expert GEMM path includes kernel-side helpers; CPU preparation is reflected in scopes and idle, not counted as GEMM GPU work.
- A matched TE/grouped-MM ablation is needed to measure the benefit of the proposed GEMM change.
- FSDP EP8 on eight GPUs has expert-DP size one, whereas EP1 has expert-DP size eight.
- Compare loss/gradients and profiler slowdown before accepting a performance conclusion.
