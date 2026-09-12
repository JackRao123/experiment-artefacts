# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/nsys_workflow_20260911/devbox-ep1-te-gcfreeze/timing.sqlite`
- Ranks: 0, 5, 4, 3, 2, 7, 6, 1; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: 6.453996008706819%.
- Hardware metrics present: False; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 12295.60 | 10236.72 | 2058.88 | 3844.87 | 2370.68 | 4843.42 | 1767.15 | 438.39 |
| 5 | 1 | 12271.85 | 11023.92 | 1247.93 | 3601.39 | 2120.43 | 5459.99 | 2335.83 | 510.52 |
| 4 | 1 | 12261.99 | 10758.74 | 1503.25 | 3539.03 | 2249.66 | 5186.50 | 2018.02 | 380.87 |
| 3 | 1 | 12265.90 | 11088.72 | 1177.19 | 3656.72 | 2019.43 | 5485.29 | 2460.82 | 640.74 |
| 2 | 1 | 12251.71 | 11117.50 | 1134.21 | 3625.49 | 1985.51 | 5471.79 | 2472.84 | 626.53 |
| 7 | 1 | 12254.95 | 11121.02 | 1133.93 | 3540.17 | 1995.80 | 5425.24 | 2388.30 | 629.53 |
| 6 | 1 | 12262.59 | 11035.83 | 1226.76 | 3602.28 | 2080.84 | 5495.57 | 2357.82 | 543.31 |
| 1 | 1 | 12271.24 | 10338.85 | 1932.39 | 3583.83 | 2181.62 | 4764.84 | 1632.83 | 411.83 |

### Capture diagnostics

- Profile median is outside the observed control range: instrumentation, remaining warmup, or workload drift may affect attribution.
- 25× on other/helper processes: Not all NVTX events might have been collected.
- 25× on other/helper processes: No NVTX events collected. Does the process use NVTX?
- 8× on captured trainer ranks: Not all NVTX events might have been collected.
- 17× on other/helper processes: Not all CUDA events might have been collected.
- 25× on other/helper processes: No CUDA events collected. Does the process use CUDA?
- 8× on captured trainer ranks: Not all CUDA events might have been collected.
- 8× on other/helper processes: CUDA profiling might have not been started correctly.

Capture completeness is qualified by these warnings. Complete step anchors and runtime links do not prove zero event loss.

## Expert-weight prefetch arrivals

Arrival relative to preceding-block compute completion; not a zero-communication speedup estimate.

| Rank | Step | Attributed gathers | Ready by preceding block end | Status |
|---|---:|---:|---|---|
| 0 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 5 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 4 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 3 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 2 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 7 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 6 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 1 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |

## Forward expert-wrapper CUDA API costs

Original + recomputed forwards, averaged per FB. API residence can overlap GPU work; it is not additive or automatically recoverable time.

| Rank | API | Calls/FB | Host API ms/FB |
|---|---|---:|---:|
| 0 | cuLaunchKernelEx | 74736 | 334.24 |
| 0 | cudaLaunchKernelExC_v11060 | 4292 | 17.12 |
| 0 | cuKernelGetAttribute | 149472 | 10.17 |
| 5 | cuLaunchKernelEx | 74532 | 249.31 |
| 5 | cudaLaunchKernelExC_v11060 | 4414 | 13.55 |
| 5 | cuKernelGetAttribute | 149064 | 10.26 |
| 4 | cuLaunchKernelEx | 74612 | 361.06 |
| 4 | cudaLaunchKernelExC_v11060 | 4552 | 19.39 |
| 4 | cuKernelGetAttribute | 149224 | 10.01 |
| 3 | cuLaunchKernelEx | 74520 | 273.89 |
| 3 | cudaLaunchKernelExC_v11060 | 4540 | 16.74 |
| 3 | cuKernelGetAttribute | 149040 | 10.05 |
| 2 | cuLaunchKernelEx | 74484 | 268.18 |
| 2 | cudaLaunchKernelExC_v11060 | 4364 | 14.41 |
| 2 | cuKernelGetAttribute | 148968 | 9.85 |
| 7 | cuLaunchKernelEx | 74400 | 253.36 |
| 7 | cudaLaunchKernelExC_v11060 | 4460 | 14.35 |
| 7 | cuKernelGetAttribute | 148800 | 9.98 |
| 6 | cuLaunchKernelEx | 74520 | 246.93 |
| 6 | cudaLaunchKernelExC_v11060 | 4572 | 14.19 |
| 6 | cuKernelGetAttribute | 149040 | 10.19 |
| 1 | cuLaunchKernelEx | 74576 | 255.69 |
| 1 | cudaLaunchKernelExC_v11060 | 4344 | 14.93 |
| 1 | cuKernelGetAttribute | 149152 | 10.15 |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 2999.03 | 4415.34 |
| gpu_idle | 2058.88 | 2058.88 |
| expert_gemm_path | 1141.31 | 2176.82 |
| fsdp_gather:expert | 927.05 | 4219.15 |
| other_compute | 522.47 | 862.84 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3125.39 | 4555.25 |
| gpu_idle | 1247.93 | 1247.93 |
| expert_gemm_path | 1146.34 | 2142.52 |
| fsdp_gather:expert | 886.12 | 4408.44 |
| cp_communication | 638.24 | 882.34 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3128.18 | 4566.84 |
| gpu_idle | 1503.25 | 1503.25 |
| expert_gemm_path | 1160.26 | 2175.14 |
| fsdp_gather:expert | 794.73 | 4222.83 |
| cp_communication | 644.32 | 761.57 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3134.55 | 4565.20 |
| expert_gemm_path | 1209.93 | 2105.34 |
| gpu_idle | 1177.19 | 1177.19 |
| fsdp_gather:expert | 875.84 | 4362.73 |
| cp_communication | 642.85 | 858.01 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3139.41 | 4579.36 |
| expert_gemm_path | 1216.46 | 2118.97 |
| gpu_idle | 1134.21 | 1134.21 |
| fsdp_gather:expert | 925.78 | 4366.30 |
| cp_communication | 649.08 | 844.65 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3150.00 | 4593.88 |
| expert_gemm_path | 1238.02 | 2146.13 |
| gpu_idle | 1133.93 | 1133.93 |
| fsdp_gather:expert | 818.60 | 4346.88 |
| cp_communication | 616.35 | 852.53 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3104.88 | 4531.35 |
| gpu_idle | 1226.76 | 1226.76 |
| expert_gemm_path | 1135.58 | 2137.48 |
| fsdp_gather:expert | 873.04 | 4419.15 |
| cp_communication | 660.91 | 915.04 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3147.52 | 4590.26 |
| gpu_idle | 1932.39 | 1932.39 |
| expert_gemm_path | 1161.56 | 2163.36 |
| fsdp_gather:expert | 858.77 | 4288.81 |
| other_compute | 529.33 | 928.93 |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
