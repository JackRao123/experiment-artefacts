# Matched Nsight comparison

Left: devbox-ep1-grouped; right: devbox-ep1-grouped-async.

Headline TPS comes from unprofiled controls. Trace categories below are attribution, not headline timing.

| Case | Controls | Mean FB seconds | TPS/GPU | Peak allocated GiB |
|---|---:|---:|---:|---:|
| devbox-ep1-grouped | 5 | 11.7888 | 1389.8 | 247.98 |
| devbox-ep1-grouped-async | 5 | 11.8737 | 1379.9 | 247.98 |

Source-change caveat: Only frozen expert offset upload changed: pinned CPU staging and nonblocking H2D, same GPU cumsum/GEMM/routing. Parent dependency pins updated; NVTX implementation unchanged.
This is not a same-revision A/B; instrumentation overhead is a possible confound.

## Per-rank reconciliation

All deltas are right minus left, milliseconds per profiled FB.

Exclusive categories + mixed-category overlap + GPU idle exactly reconcile to traced step time.

| Rank | Step delta | Expert GEMM exclusive delta | Dispatcher exclusive delta | Other exclusive delta | Mixed overlap delta | Idle delta | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 272.83 | -138.47 | -104.22 | 539.86 | 350.99 | -375.33 | -0.00 |
| 1 | 288.94 | -108.36 | -103.87 | 443.70 | 299.61 | -242.14 | -0.00 |
| 2 | 305.07 | -44.15 | 2.66 | 610.75 | 255.97 | -520.15 | 0.00 |
| 3 | 310.01 | -5.43 | 81.27 | 519.23 | 191.54 | -476.60 | -0.00 |
| 4 | 306.91 | -59.73 | 30.73 | 488.73 | 232.75 | -385.58 | 0.00 |
| 5 | 295.40 | -121.54 | -54.83 | 578.79 | 349.02 | -456.03 | 0.00 |
| 6 | 311.72 | -46.97 | 1.57 | 484.99 | 278.14 | -406.01 | 0.00 |
| 7 | 308.69 | 37.05 | 110.98 | 547.62 | 12.60 | -399.55 | -0.00 |

## Expert execution and attention (overlap included)

Do not infer equal GEMM execution time from equal exclusive time: FSDP can overlap a slower GEMM.

| Rank | Left expert GPU ms | Right expert GPU ms | Left attention GPU ms | Right attention GPU ms | Left idle in expert host scopes | Right idle in expert host scopes |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2295.99 | 2414.53 | 4508.87 | 4697.16 | 110.80 | 4.11 |
| 1 | 2288.37 | 2439.41 | 4708.45 | 4889.64 | 109.61 | 1.50 |
| 2 | 2267.36 | 2351.64 | 4685.12 | 4799.14 | 106.86 | 0.49 |
| 3 | 2315.12 | 2326.47 | 4651.63 | 4791.84 | 108.02 | 0.41 |
| 4 | 2378.27 | 2391.10 | 4637.20 | 4776.23 | 107.51 | 4.49 |
| 5 | 2337.68 | 2396.67 | 4638.78 | 4806.63 | 106.70 | 3.07 |
| 6 | 2260.65 | 2343.16 | 4629.21 | 4722.52 | 106.25 | 0.43 |
| 7 | 2415.78 | 2387.96 | 4724.70 | 4874.87 | 105.03 | 0.49 |

## Interpretation limits

- This reconciliation is an observed wall-time partition, not a causal estimate of removable time.
- Expert GEMM path includes kernel-side helpers; CPU preparation is reflected in scopes and idle, not counted as GEMM GPU work.
- A matched TE/grouped-MM ablation is needed to measure the benefit of the proposed GEMM change.
- FSDP EP8 on eight GPUs has expert-DP size one, whereas EP1 has expert-DP size eight.
- Compare loss/gradients and profiler slowdown before accepting a performance conclusion.
