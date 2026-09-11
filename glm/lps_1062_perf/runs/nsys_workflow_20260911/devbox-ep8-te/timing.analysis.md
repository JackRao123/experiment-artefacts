# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `devbox-ep8-te/timing.sqlite`
- Ranks: 0, 3, 1, 2, 7, 5, 4, 6; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: 0.7131290291236381%.
- Hardware metrics present: False; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 10423.55 | 10191.57 | 231.98 | 3575.09 | 371.53 | 3411.02 | 3328.59 | 3031.41 |
| 3 | 1 | 10416.73 | 10184.28 | 232.44 | 3385.91 | 339.86 | 3219.88 | 3138.91 | 2875.87 |
| 1 | 1 | 10420.58 | 10182.94 | 237.63 | 3282.41 | 310.52 | 3110.42 | 3030.30 | 2798.82 |
| 2 | 1 | 10418.39 | 10169.21 | 249.18 | 3371.21 | 341.94 | 3185.48 | 3107.68 | 2852.90 |
| 7 | 1 | 10417.42 | 10181.20 | 236.22 | 3299.21 | 298.11 | 3123.76 | 3048.75 | 2832.90 |
| 5 | 1 | 10418.30 | 10173.40 | 244.90 | 3356.73 | 311.04 | 3175.79 | 3097.83 | 2868.31 |
| 4 | 1 | 10417.85 | 10151.60 | 266.26 | 3328.38 | 317.90 | 3123.92 | 3048.03 | 2830.10 |
| 6 | 1 | 10431.95 | 10141.51 | 290.44 | 3121.45 | 353.94 | 2894.97 | 2816.91 | 2568.26 |

### Capture diagnostics

- 25× on other/helper processes: Not all NVTX events might have been collected.
- 25× on other/helper processes: No NVTX events collected. Does the process use NVTX?
- 8× on captured trainer ranks: Not all NVTX events might have been collected.
- 8× on other/helper processes: CUDA profiling might have not been started correctly.
- 25× on other/helper processes: No CUDA events collected. Does the process use CUDA?
- 17× on other/helper processes: Not all CUDA events might have been collected.
- 8× on captured trainer ranks: Not all CUDA events might have been collected.

Capture completeness is qualified by these warnings. Complete step anchors and runtime links do not prove zero event loss.

## Expert-weight prefetch arrivals

Arrival relative to preceding-block compute completion; not a zero-communication speedup estimate.

| Rank | Step | Attributed gathers | Ready by preceding block end | Status |
|---|---:|---:|---|---|
| 0 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 3 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 1 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 2 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 7 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 5 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 4 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 6 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3951.76 | 4019.79 |
| expert_dispatch_combine | 2796.33 | 2949.63 |
| expert_gemm_path | 1244.12 | 1244.12 |
| other_compute | 702.40 | 714.73 |
| other_gemm | 556.41 | 562.18 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4088.23 | 4155.32 |
| expert_dispatch_combine | 2714.25 | 2865.13 |
| expert_gemm_path | 1288.75 | 1288.75 |
| other_compute | 707.95 | 719.70 |
| other_gemm | 551.90 | 557.46 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4136.03 | 4202.02 |
| expert_dispatch_combine | 2647.64 | 2799.87 |
| expert_gemm_path | 1300.68 | 1300.68 |
| other_compute | 746.85 | 759.50 |
| other_gemm | 569.38 | 574.59 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4093.21 | 4157.67 |
| expert_dispatch_combine | 2694.83 | 2845.97 |
| expert_gemm_path | 1275.58 | 1275.58 |
| other_compute | 703.64 | 715.63 |
| other_gemm | 559.65 | 564.93 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4142.42 | 4206.46 |
| expert_dispatch_combine | 2657.93 | 2811.08 |
| expert_gemm_path | 1256.43 | 1256.43 |
| other_compute | 736.34 | 746.08 |
| other_gemm | 572.98 | 578.02 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4130.64 | 4194.27 |
| expert_dispatch_combine | 2688.20 | 2841.37 |
| expert_gemm_path | 1226.26 | 1226.26 |
| other_compute | 734.67 | 747.50 |
| other_gemm | 556.55 | 561.90 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4139.80 | 4201.76 |
| expert_dispatch_combine | 2656.33 | 2808.68 |
| expert_gemm_path | 1243.41 | 1243.41 |
| other_compute | 739.35 | 751.78 |
| other_gemm | 555.03 | 560.34 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4119.82 | 4183.00 |
| expert_dispatch_combine | 2443.62 | 2596.13 |
| expert_gemm_path | 1443.50 | 1443.50 |
| other_compute | 763.71 | 777.04 |
| other_gemm | 555.69 | 561.13 |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
