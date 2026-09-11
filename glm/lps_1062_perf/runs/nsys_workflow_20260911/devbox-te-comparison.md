# Matched Nsight comparison

Left: devbox-ep1-te; right: devbox-ep8-te.

Headline TPS comes from unprofiled controls. Trace categories below are attribution, not headline timing.

| Case | Controls | Mean FB seconds | TPS/GPU | Peak allocated GiB |
|---|---:|---:|---:|---:|
| devbox-ep1-te | 5 | 11.9001 | 1376.8 | 248.10 |
| devbox-ep8-te | 5 | 10.5694 | 1550.1 | 213.62 |

Checkpoint relocation: The identical zai-org/GLM-5.3 snapshot 187fb9fff6319062325ff825627ef6db084d9bc6 was restored to node-local disk after shared-cache removal; startup excluded.
Both paths identify the same HF repository and exact snapshot commit; startup is excluded from controls.

## Per-rank reconciliation

All deltas are right minus left, milliseconds per profiled FB.

Exclusive categories + mixed-category overlap + GPU idle exactly reconcile to traced step time.

| Rank | Step delta | Expert GEMM exclusive delta | Dispatcher exclusive delta | Other exclusive delta | Mixed overlap delta | Idle delta | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | -2067.58 | 101.40 | 2657.40 | -253.29 | -3079.99 | -1493.10 | 0.00 |
| 1 | -2054.85 | 72.00 | 2430.68 | 513.85 | -3352.10 | -1719.29 | -0.00 |
| 2 | -2050.87 | 112.86 | 2556.24 | -305.68 | -3217.11 | -1197.18 | -0.00 |
| 3 | -2045.79 | 164.17 | 2601.57 | -317.76 | -3219.05 | -1274.72 | -0.00 |
| 4 | -2049.32 | 140.96 | 2543.60 | -346.51 | -3349.48 | -1037.89 | -0.00 |
| 5 | -2045.13 | 21.98 | 2471.59 | -235.12 | -3375.40 | -928.17 | 0.00 |
| 6 | -2019.41 | 305.38 | 2304.65 | -323.61 | -3355.34 | -950.48 | 0.00 |
| 7 | -2044.23 | 17.84 | 2439.22 | -227.60 | -3343.45 | -930.25 | 0.00 |

## Expert execution and attention (overlap included)

Do not infer equal GEMM execution time from equal exclusive time: FSDP can overlap a slower GEMM.

| Rank | Left expert GPU ms | Right expert GPU ms | Left attention GPU ms | Right attention GPU ms | Left idle in expert host scopes | Right idle in expert host scopes |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2172.56 | 1244.12 | 4419.98 | 4019.79 | 399.27 | 2.44 |
| 1 | 2141.09 | 1300.68 | 4579.65 | 4202.02 | 168.88 | 3.00 |
| 2 | 2169.27 | 1275.58 | 4563.05 | 4157.67 | 248.15 | 4.04 |
| 3 | 2187.83 | 1288.75 | 4548.15 | 4155.32 | 340.07 | 3.62 |
| 4 | 2159.84 | 1243.41 | 4542.50 | 4201.76 | 200.43 | 2.85 |
| 5 | 2108.24 | 1226.26 | 4540.34 | 4194.27 | 102.36 | 2.50 |
| 6 | 2134.09 | 1443.50 | 4536.35 | 4183.00 | 178.95 | 2.61 |
| 7 | 2140.30 | 1256.43 | 4597.35 | 4206.46 | 108.33 | 2.61 |

## Interpretation limits

- This reconciliation is an observed wall-time partition, not a causal estimate of removable time.
- Expert GEMM path includes kernel-side helpers; CPU preparation is reflected in scopes and idle, not counted as GEMM GPU work.
- A matched TE/grouped-MM ablation is needed to measure the benefit of the proposed GEMM change.
- FSDP EP8 on eight GPUs has expert-DP size one, whereas EP1 has expert-DP size eight.
- Compare loss/gradients and profiler slowdown before accepting a performance conclusion.
