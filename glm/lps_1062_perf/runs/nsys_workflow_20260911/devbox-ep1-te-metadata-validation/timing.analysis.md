# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/nsys_workflow_20260911/devbox-ep1-te-metadata-validation/timing.sqlite`
- Ranks: 0, 4, 3, 6, 2, 7, 5, 1; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: 2.720226982765528%.
- Hardware metrics present: False; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 11191.02 | 10278.67 | 912.35 | 2803.02 | 1480.23 | 4891.77 | 1871.98 | 510.24 |
| 4 | 1 | 11188.96 | 10596.59 | 592.37 | 2663.23 | 1112.03 | 5014.22 | 2053.00 | 694.49 |
| 3 | 1 | 11180.16 | 10343.13 | 837.02 | 2615.03 | 938.42 | 4821.18 | 1759.33 | 507.01 |
| 6 | 1 | 11184.65 | 10477.08 | 707.57 | 2652.88 | 1182.87 | 4978.94 | 1927.33 | 587.10 |
| 2 | 1 | 11187.83 | 10292.53 | 895.31 | 2516.13 | 1203.82 | 4794.04 | 1602.14 | 415.02 |
| 7 | 1 | 11175.32 | 10599.65 | 575.67 | 2364.51 | 766.92 | 4831.63 | 1770.59 | 617.13 |
| 5 | 1 | 11186.08 | 10540.61 | 645.47 | 2661.10 | 1055.72 | 4966.80 | 1997.84 | 732.63 |
| 1 | 1 | 11176.34 | 10446.19 | 730.16 | 2622.77 | 1235.19 | 4838.99 | 1873.91 | 579.92 |

### Capture diagnostics

- 25× on other/helper processes: Not all NVTX events might have been collected.
- 25× on other/helper processes: No NVTX events collected. Does the process use NVTX?
- 8× on captured trainer ranks: Not all NVTX events might have been collected.
- 17× on other/helper processes: Not all CUDA events might have been collected.
- 25× on other/helper processes: No CUDA events collected. Does the process use CUDA?
- 8× on other/helper processes: CUDA profiling might have not been started correctly.
- 8× on captured trainer ranks: Not all CUDA events might have been collected.

Capture completeness is qualified by these warnings. Complete step anchors and runtime links do not prove zero event loss.

## Expert-weight prefetch arrivals

Arrival relative to preceding-block compute completion; not a zero-communication speedup estimate.

| Rank | Step | Attributed gathers | Ready by preceding block end | Status |
|---|---:|---:|---|---|
| 0 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 4 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 3 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 6 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 2 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 7 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 5 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 1 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |

## Forward expert-wrapper CUDA API costs

Original + recomputed forwards, averaged per FB. API residence can overlap GPU work; it is not additive or automatically recoverable time.

| Rank | API | Calls/FB | Host API ms/FB |
|---|---|---:|---:|
| 0 | cuLaunchKernelEx | 66176 | 304.91 |
| 0 | cudaLaunchKernelExC_v11060 | 6262 | 25.29 |
| 0 | cuKernelGetAttribute | 132352 | 9.52 |
| 4 | cuLaunchKernelEx | 62980 | 216.63 |
| 4 | cudaLaunchKernelExC_v11060 | 5460 | 17.18 |
| 4 | cuKernelGetAttribute | 125960 | 8.53 |
| 3 | cuLaunchKernelEx | 62544 | 282.02 |
| 3 | cudaLaunchKernelExC_v11060 | 5178 | 20.80 |
| 3 | cuKernelGetAttribute | 125088 | 8.46 |
| 6 | cuLaunchKernelEx | 62688 | 209.80 |
| 6 | cudaLaunchKernelExC_v11060 | 5346 | 16.09 |
| 6 | cuKernelGetAttribute | 125376 | 8.54 |
| 2 | cuLaunchKernelEx | 63156 | 232.62 |
| 2 | cudaLaunchKernelExC_v11060 | 5452 | 17.85 |
| 2 | cuKernelGetAttribute | 126312 | 8.50 |
| 7 | cuLaunchKernelEx | 62336 | 265.47 |
| 7 | cudaLaunchKernelExC_v11060 | 5294 | 17.35 |
| 7 | cuKernelGetAttribute | 124672 | 8.63 |
| 5 | cuLaunchKernelEx | 62992 | 278.73 |
| 5 | cudaLaunchKernelExC_v11060 | 5446 | 16.81 |
| 5 | cuKernelGetAttribute | 125984 | 8.54 |
| 1 | cuLaunchKernelEx | 62976 | 342.31 |
| 1 | cudaLaunchKernelExC_v11060 | 5250 | 20.46 |
| 1 | cuKernelGetAttribute | 125952 | 8.55 |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3017.25 | 4436.30 |
| expert_gemm_path | 1087.79 | 2072.34 |
| fsdp_gather:expert | 1029.41 | 4289.31 |
| gpu_idle | 912.35 | 912.35 |
| other_compute | 529.08 | 863.75 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3152.08 | 4600.67 |
| expert_gemm_path | 1141.81 | 1983.25 |
| fsdp_gather:expert | 987.45 | 4450.73 |
| gpu_idle | 592.37 | 592.37 |
| other_compute | 529.63 | 918.43 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3145.99 | 4600.07 |
| expert_gemm_path | 1091.44 | 2022.22 |
| fsdp_gather:expert | 960.27 | 4353.48 |
| gpu_idle | 837.02 | 837.02 |
| other_compute | 531.71 | 927.07 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3121.99 | 4571.52 |
| expert_gemm_path | 1071.18 | 2006.73 |
| fsdp_gather:expert | 1015.91 | 4470.65 |
| gpu_idle | 707.57 | 707.57 |
| other_compute | 528.03 | 911.05 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3153.05 | 4610.66 |
| expert_gemm_path | 1049.06 | 2050.13 |
| fsdp_gather:expert | 951.76 | 4430.68 |
| gpu_idle | 895.31 | 895.31 |
| other_compute | 531.02 | 981.63 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3193.93 | 4663.40 |
| expert_gemm_path | 1184.54 | 2025.53 |
| fsdp_gather:expert | 854.13 | 4351.94 |
| gpu_idle | 575.67 | 575.67 |
| other_compute | 540.78 | 1000.94 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3145.07 | 4594.76 |
| expert_gemm_path | 1141.17 | 1982.22 |
| fsdp_gather:expert | 923.91 | 4396.86 |
| gpu_idle | 645.47 | 645.47 |
| other_compute | 527.63 | 921.69 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3172.56 | 4631.69 |
| expert_gemm_path | 1168.30 | 2016.20 |
| fsdp_gather:expert | 909.77 | 4290.67 |
| gpu_idle | 730.16 | 730.16 |
| other_compute | 531.49 | 901.01 |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
