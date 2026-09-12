# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/nsys_workflow_20260911/devbox-ep1-te-metadata/timing.sqlite`
- Ranks: 0, 1, 6, 5, 2, 7, 4, 3; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: -0.839739054093902%.
- Hardware metrics present: False; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 11194.66 | 10172.35 | 1022.31 | 2705.74 | 1455.24 | 4714.85 | 1664.77 | 459.76 |
| 1 | 1 | 11188.89 | 10380.26 | 808.63 | 2528.74 | 1227.68 | 4702.61 | 1701.29 | 599.56 |
| 6 | 1 | 11163.26 | 10356.57 | 806.68 | 2506.98 | 1263.99 | 4764.62 | 1682.29 | 527.77 |
| 5 | 1 | 11178.63 | 10490.17 | 688.47 | 2529.26 | 1149.13 | 4828.74 | 1823.09 | 640.06 |
| 2 | 1 | 11179.55 | 10255.89 | 923.66 | 2376.97 | 1241.51 | 4668.81 | 1434.56 | 416.97 |
| 7 | 1 | 11185.85 | 10521.70 | 664.15 | 2279.01 | 1056.07 | 4687.30 | 1596.34 | 527.90 |
| 4 | 1 | 11177.95 | 10514.21 | 663.74 | 2531.85 | 1107.84 | 4853.72 | 1850.28 | 682.39 |
| 3 | 1 | 11182.34 | 10259.13 | 923.21 | 2488.70 | 1274.99 | 4655.76 | 1546.68 | 437.09 |

### Capture diagnostics

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
| 1 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 6 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 5 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 2 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 7 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 4 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 3 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |

## Forward expert-wrapper CUDA API costs

Original + recomputed forwards, averaged per FB. API residence can overlap GPU work; it is not additive or automatically recoverable time.

| Rank | API | Calls/FB | Host API ms/FB |
|---|---|---:|---:|
| 0 | cuLaunchKernelEx | 74748 | 339.48 |
| 0 | cudaLaunchKernelExC_v11060 | 4368 | 18.14 |
| 0 | cuKernelGetAttribute | 149496 | 10.53 |
| 1 | cuLaunchKernelEx | 74592 | 332.73 |
| 1 | cudaLaunchKernelExC_v11060 | 4336 | 17.00 |
| 1 | cuKernelGetAttribute | 149184 | 10.10 |
| 6 | cuLaunchKernelEx | 74512 | 249.60 |
| 6 | cudaLaunchKernelExC_v11060 | 4602 | 14.05 |
| 6 | cuKernelGetAttribute | 149024 | 10.32 |
| 5 | cuLaunchKernelEx | 74548 | 254.71 |
| 5 | cudaLaunchKernelExC_v11060 | 4500 | 14.52 |
| 5 | cuKernelGetAttribute | 149096 | 10.15 |
| 2 | cuLaunchKernelEx | 74548 | 272.07 |
| 2 | cudaLaunchKernelExC_v11060 | 4418 | 14.71 |
| 2 | cuKernelGetAttribute | 149096 | 10.12 |
| 7 | cuLaunchKernelEx | 74452 | 274.17 |
| 7 | cudaLaunchKernelExC_v11060 | 4464 | 15.12 |
| 7 | cuKernelGetAttribute | 148904 | 10.13 |
| 4 | cuLaunchKernelEx | 74504 | 249.18 |
| 4 | cudaLaunchKernelExC_v11060 | 4536 | 13.69 |
| 4 | cuKernelGetAttribute | 149008 | 10.01 |
| 3 | cuLaunchKernelEx | 74596 | 332.46 |
| 3 | cudaLaunchKernelExC_v11060 | 4604 | 18.01 |
| 3 | cuKernelGetAttribute | 149192 | 9.98 |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3033.13 | 4427.84 |
| expert_gemm_path | 1146.73 | 2180.27 |
| gpu_idle | 1022.31 | 1022.31 |
| fsdp_gather:expert | 881.40 | 4190.02 |
| other_compute | 526.57 | 864.41 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3182.81 | 4612.12 |
| expert_gemm_path | 1229.68 | 2141.35 |
| gpu_idle | 808.63 | 808.63 |
| fsdp_gather:expert | 771.52 | 4208.46 |
| other_compute | 530.51 | 901.84 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3136.62 | 4562.58 |
| expert_gemm_path | 1146.39 | 2140.56 |
| fsdp_gather:expert | 842.37 | 4346.77 |
| gpu_idle | 806.68 | 806.68 |
| other_compute | 531.09 | 910.62 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3166.95 | 4587.20 |
| expert_gemm_path | 1205.41 | 2113.47 |
| fsdp_gather:expert | 829.69 | 4319.31 |
| gpu_idle | 688.47 | 688.47 |
| other_compute | 528.98 | 922.19 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3174.61 | 4604.10 |
| expert_gemm_path | 1118.73 | 2180.67 |
| gpu_idle | 923.66 | 923.66 |
| fsdp_gather:expert | 783.02 | 4303.07 |
| other_compute | 530.10 | 985.97 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3203.19 | 4643.32 |
| expert_gemm_path | 1247.28 | 2144.56 |
| fsdp_gather:expert | 725.15 | 4259.91 |
| gpu_idle | 664.15 | 664.15 |
| other_compute | 538.44 | 1002.31 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3167.68 | 4594.21 |
| expert_gemm_path | 1207.29 | 2109.14 |
| fsdp_gather:expert | 818.35 | 4340.58 |
| gpu_idle | 663.74 | 663.74 |
| other_compute | 528.55 | 919.57 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3163.50 | 4588.01 |
| expert_gemm_path | 1161.48 | 2160.68 |
| gpu_idle | 923.21 | 923.21 |
| fsdp_gather:expert | 789.84 | 4231.35 |
| other_compute | 528.54 | 928.51 |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
