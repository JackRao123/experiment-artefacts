# Matched Nsight comparison

Left: ep1-te; right: ep8-te.

Headline TPS comes from unprofiled controls. Trace categories below are attribution, not headline timing.

| Case | Controls | Mean FB seconds | TPS/GPU | Peak allocated GiB |
|---|---:|---:|---:|---:|
| ep1-te | 5 | 11.5838 | 1414.4 | 248.10 |
| ep8-te | 5 | 10.5831 | 1548.1 | 213.62 |

## Per-rank reconciliation

All deltas are right minus left, milliseconds per profiled FB.

Exclusive categories + mixed-category overlap + GPU idle exactly reconcile to traced step time.

| Rank | Step delta | Expert GEMM exclusive delta | Dispatcher exclusive delta | Other exclusive delta | Mixed overlap delta | Idle delta | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | -1511.39 | 46.37 | 2637.55 | 179.89 | -3102.30 | -1272.90 | 0.00 |
| 1 | -1513.50 | 122.91 | 2579.59 | 287.62 | -3331.79 | -1171.82 | 0.00 |
| 2 | -1497.03 | 30.48 | 2636.28 | 121.21 | -3316.92 | -968.08 | -0.00 |
| 3 | -1503.46 | 87.05 | 2591.93 | -34.79 | -3273.70 | -873.94 | -0.00 |
| 4 | -1508.40 | 99.20 | 2683.52 | 267.54 | -3270.45 | -1288.20 | 0.00 |
| 5 | -1509.96 | 110.08 | 2657.28 | 67.41 | -3249.44 | -1095.29 | -0.00 |
| 6 | -1500.62 | 236.55 | 2385.63 | -1.52 | -3292.64 | -828.63 | -0.00 |
| 7 | -1509.91 | 38.99 | 2694.56 | 128.50 | -3272.33 | -1099.62 | 0.00 |

## Interpretation limits

- This reconciliation is an observed wall-time partition, not a causal estimate of removable time.
- Expert GEMM path includes kernel-side helpers; CPU preparation is reflected in scopes and idle, not counted as GEMM GPU work.
- A matched TE/grouped-MM ablation is needed to measure the benefit of the proposed GEMM change.
- FSDP EP8 on eight GPUs has expert-DP size one, whereas EP1 has expert-DP size eight.
- Compare loss/gradients and profiler slowdown before accepting a performance conclusion.
