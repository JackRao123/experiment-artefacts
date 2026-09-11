# Matched Nsight comparison

Left: devbox-ep1-te; right: devbox-ep1-grouped.

Headline TPS comes from unprofiled controls. Trace categories below are attribution, not headline timing.

| Case | Controls | Mean FB seconds | TPS/GPU | Peak allocated GiB |
|---|---:|---:|---:|---:|
| devbox-ep1-te | 5 | 11.9001 | 1376.8 | 248.10 |
| devbox-ep1-grouped | 5 | 11.7888 | 1389.8 | 247.98 |

Checkpoint relocation: Exact same HF snapshot restored under node-local cache root after shared-cache cleanup.
Both paths identify the same HF repository and exact snapshot commit; startup is excluded from controls.

## Per-rank reconciliation

All deltas are right minus left, milliseconds per profiled FB.

Exclusive categories + mixed-category overlap + GPU idle exactly reconcile to traced step time.

| Rank | Step delta | Expert GEMM exclusive delta | Dispatcher exclusive delta | Other exclusive delta | Mixed overlap delta | Idle delta | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | -808.47 | 45.00 | 80.44 | -550.57 | 222.48 | -605.82 | -0.00 |
| 1 | -826.75 | -26.65 | 2.65 | 10.12 | 69.50 | -882.37 | 0.00 |
| 2 | -824.34 | 26.96 | 80.55 | -531.65 | 94.11 | -494.31 | -0.00 |
| 3 | -817.44 | 12.43 | 27.40 | -396.67 | 141.22 | -601.82 | 0.00 |
| 4 | -830.83 | 5.81 | -28.91 | -507.65 | 182.73 | -482.81 | -0.00 |
| 5 | -821.38 | -53.41 | -77.11 | -438.21 | 54.83 | -307.48 | 0.00 |
| 6 | -827.76 | 35.57 | 80.10 | -548.44 | 24.95 | -419.94 | -0.00 |
| 7 | -827.49 | -70.24 | -104.09 | -469.81 | 155.38 | -338.73 | 0.00 |

## Expert execution and attention (overlap included)

Do not infer equal GEMM execution time from equal exclusive time: FSDP can overlap a slower GEMM.

| Rank | Left expert GPU ms | Right expert GPU ms | Left attention GPU ms | Right attention GPU ms | Left idle in expert host scopes | Right idle in expert host scopes |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2172.56 | 2295.99 | 4419.98 | 4508.87 | 399.27 | 110.80 |
| 1 | 2141.09 | 2288.37 | 4579.65 | 4708.45 | 168.88 | 109.61 |
| 2 | 2169.27 | 2267.36 | 4563.05 | 4685.12 | 248.15 | 106.86 |
| 3 | 2187.83 | 2315.12 | 4548.15 | 4651.63 | 340.07 | 108.02 |
| 4 | 2159.84 | 2378.27 | 4542.50 | 4637.20 | 200.43 | 107.51 |
| 5 | 2108.24 | 2337.68 | 4540.34 | 4638.78 | 102.36 | 106.70 |
| 6 | 2134.09 | 2260.65 | 4536.35 | 4629.21 | 178.95 | 106.25 |
| 7 | 2140.30 | 2415.78 | 4597.35 | 4724.70 | 108.33 | 105.03 |

## Interpretation limits

- This reconciliation is an observed wall-time partition, not a causal estimate of removable time.
- Expert GEMM path includes kernel-side helpers; CPU preparation is reflected in scopes and idle, not counted as GEMM GPU work.
- A matched TE/grouped-MM ablation is needed to measure the benefit of the proposed GEMM change.
- FSDP EP8 on eight GPUs has expert-DP size one, whereas EP1 has expert-DP size eight.
- Compare loss/gradients and profiler slowdown before accepting a performance conclusion.
