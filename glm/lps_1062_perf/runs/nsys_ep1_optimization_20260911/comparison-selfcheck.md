# Matched Nsight comparison

Left: devbox-ep1-te-repeat; right: devbox-ep1-te-repeat.

Headline TPS comes from unprofiled controls. Trace categories below are attribution, not headline timing.

| Case | Controls | Mean FB seconds | TPS/GPU | Peak allocated GiB |
|---|---:|---:|---:|---:|
| devbox-ep1-te-repeat | 5 | 11.8373 | 1384.1 | 248.10 |
| devbox-ep1-te-repeat | 5 | 11.8373 | 1384.1 | 248.10 |

## Per-rank reconciliation

All deltas are right minus left, milliseconds per profiled FB.

Exclusive categories + mixed-category overlap + GPU idle exactly reconcile to traced step time.

| Rank | Step delta | Expert GEMM exclusive delta | Dispatcher exclusive delta | Other exclusive delta | Mixed overlap delta | Idle delta | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 7 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Expert execution and attention (overlap included)

Do not infer equal GEMM execution time from equal exclusive time: FSDP can overlap a slower GEMM.

| Rank | Left expert GPU ms | Right expert GPU ms | Left attention GPU ms | Right attention GPU ms | Left idle in expert host scopes | Right idle in expert host scopes |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2119.51 | 2119.51 | 4413.63 | 4413.63 | 202.14 | 202.14 |
| 1 | 2167.70 | 2167.70 | 4587.00 | 4587.00 | 268.07 | 268.07 |
| 2 | 2140.78 | 2140.78 | 4574.29 | 4574.29 | 189.03 | 189.03 |
| 3 | 2149.41 | 2149.41 | 4558.56 | 4558.56 | 314.54 | 314.54 |
| 4 | 2181.74 | 2181.74 | 4553.08 | 4553.08 | 319.56 | 319.56 |
| 5 | 2131.95 | 2131.95 | 4546.23 | 4546.23 | 182.15 | 182.15 |
| 6 | 2200.65 | 2200.65 | 4534.86 | 4534.86 | 460.77 | 460.77 |
| 7 | 2154.75 | 2154.75 | 4599.05 | 4599.05 | 163.11 | 163.11 |

## Interpretation limits

- This reconciliation is an observed wall-time partition, not a causal estimate of removable time.
- Expert GEMM path includes kernel-side helpers; CPU preparation is reflected in scopes and idle, not counted as GEMM GPU work.
- A matched TE/grouped-MM ablation is needed to measure the benefit of the proposed GEMM change.
- FSDP EP8 on eight GPUs has expert-DP size one, whereas EP1 has expert-DP size eight.
- Compare loss/gradients and profiler slowdown before accepting a performance conclusion.
