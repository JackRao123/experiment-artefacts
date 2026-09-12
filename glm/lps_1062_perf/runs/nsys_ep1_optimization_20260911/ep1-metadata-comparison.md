# Matched Nsight comparison

Left: devbox-ep1-te-repeat; right: devbox-ep1-te-metadata.

Headline TPS comes from unprofiled controls. Trace categories below are attribution, not headline timing.

| Case | Controls | Mean FB seconds | TPS/GPU | Peak allocated GiB |
|---|---:|---:|---:|---:|
| devbox-ep1-te-repeat | 5 | 11.8373 | 1384.1 | 248.10 |
| devbox-ep1-te-metadata | 5 | 11.3441 | 1444.3 | 248.10 |

Source-change caveat: Candidate adds the metadata-cache implementation plus GC/FSDP NVTX scopes and contains the inactive grouped-MM offset fix. Both cases use TE. New instrumentation is a possible confound; an annotated same-revision baseline is planned.
This is not a same-revision A/B; instrumentation overhead is a possible confound.

Runtime ablation (cache_parameter_metadata): Cache static FSDP parameter owners and persistent frozen BF16 buffer views. Same TE kernels, precision, routing, recompute and collectives; GC freezing off.

## Per-rank reconciliation

All deltas are right minus left, milliseconds per profiled FB.

Exclusive categories + mixed-category overlap + GPU idle exactly reconcile to traced step time.

| Rank | Step delta | Expert GEMM exclusive delta | Dispatcher exclusive delta | Other exclusive delta | Mixed overlap delta | Idle delta | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | -976.20 | -42.49 | -77.63 | -416.73 | -7.91 | -431.44 | 0.00 |
| 1 | -1006.35 | 69.39 | 78.70 | -277.92 | 9.77 | -886.29 | 0.00 |
| 2 | -986.62 | -26.01 | -55.22 | -598.23 | 120.65 | -427.81 | -0.00 |
| 3 | -989.02 | 65.97 | 55.55 | -580.32 | 9.87 | -540.09 | -0.00 |
| 4 | -1025.92 | 91.38 | 104.24 | -420.67 | 84.19 | -885.07 | -0.00 |
| 5 | -981.81 | 74.96 | 78.37 | -543.15 | -1.85 | -590.14 | 0.00 |
| 6 | -1002.10 | -19.38 | 0.11 | -117.76 | 78.18 | -943.24 | 0.00 |
| 7 | -960.51 | 76.56 | 79.38 | -573.81 | 9.99 | -552.63 | -0.00 |

## Expert execution and attention (overlap included)

Do not infer equal GEMM execution time from equal exclusive time: FSDP can overlap a slower GEMM.

| Rank | Left expert GPU ms | Right expert GPU ms | Left attention GPU ms | Right attention GPU ms | Left idle in expert host scopes | Right idle in expert host scopes |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2119.51 | 2180.27 | 4413.63 | 4427.84 | 202.14 | 343.48 |
| 1 | 2167.70 | 2141.35 | 4587.00 | 4612.12 | 268.07 | 152.95 |
| 2 | 2140.78 | 2180.67 | 4574.29 | 4604.10 | 189.03 | 257.87 |
| 3 | 2149.41 | 2160.68 | 4558.56 | 4588.01 | 314.54 | 252.75 |
| 4 | 2181.74 | 2109.14 | 4553.08 | 4594.21 | 319.56 | 91.99 |
| 5 | 2131.95 | 2113.47 | 4546.23 | 4587.20 | 182.15 | 99.27 |
| 6 | 2200.65 | 2140.56 | 4534.86 | 4562.58 | 460.77 | 184.68 |
| 7 | 2154.75 | 2144.56 | 4599.05 | 4643.32 | 163.11 | 107.74 |

## Interpretation limits

- This reconciliation is an observed wall-time partition, not a causal estimate of removable time.
- Expert GEMM path includes kernel-side helpers; CPU preparation is reflected in scopes and idle, not counted as GEMM GPU work.
- A matched TE/grouped-MM ablation is needed to measure the benefit of the proposed GEMM change.
- FSDP EP8 on eight GPUs has expert-DP size one, whereas EP1 has expert-DP size eight.
- Compare loss/gradients and profiler slowdown before accepting a performance conclusion.
