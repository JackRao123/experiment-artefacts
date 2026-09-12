# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `devbox-ep1-te-repeat/timing.sqlite`
- Ranks: 0, 3, 1, 4, 7, 6, 5, 2; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: 3.057039277063156%.
- Hardware metrics present: False; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 12170.86 | 10717.11 | 1453.75 | 3753.03 | 1922.09 | 5242.57 | 2280.74 | 603.32 |
| 3 | 1 | 12171.36 | 10708.06 | 1463.30 | 3512.78 | 1831.62 | 5204.60 | 2030.77 | 431.55 |
| 1 | 1 | 12195.24 | 10500.32 | 1694.92 | 3512.42 | 1880.85 | 4931.03 | 1798.91 | 421.00 |
| 4 | 1 | 12203.87 | 10655.07 | 1548.80 | 3513.14 | 1938.45 | 5110.79 | 1946.61 | 410.73 |
| 7 | 1 | 12146.36 | 10929.58 | 1216.78 | 3380.67 | 1714.50 | 5273.00 | 2145.68 | 470.12 |
| 6 | 1 | 12165.35 | 10415.43 | 1749.93 | 3473.21 | 2243.12 | 4843.30 | 1705.60 | 396.86 |
| 5 | 1 | 12160.44 | 10881.84 | 1278.60 | 3520.99 | 1774.01 | 5320.18 | 2224.62 | 523.18 |
| 2 | 1 | 12166.17 | 10814.70 | 1351.46 | 3549.80 | 1826.00 | 5241.82 | 2179.68 | 471.41 |

### Capture diagnostics

- Profile median is outside the observed control range: instrumentation, remaining warmup, or workload drift may affect attribution.
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
| 3 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 1 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 4 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 7 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 6 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 5 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 2 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |

## Forward expert-wrapper CUDA API costs

Original + recomputed forwards, averaged per FB. API residence can overlap GPU work; it is not additive or automatically recoverable time.

| Rank | API | Calls/FB | Host API ms/FB |
|---|---|---:|---:|
| 0 | cuLaunchKernelEx | 74792 | 331.48 |
| 0 | cudaLaunchKernelExC_v11060 | 4412 | 17.61 |
| 0 | cuKernelGetAttribute | 149584 | 10.36 |
| 3 | cuLaunchKernelEx | 74480 | 275.07 |
| 3 | cudaLaunchKernelExC_v11060 | 4526 | 15.15 |
| 3 | cuKernelGetAttribute | 148960 | 10.16 |
| 1 | cuLaunchKernelEx | 74632 | 310.22 |
| 1 | cudaLaunchKernelExC_v11060 | 4324 | 16.17 |
| 1 | cuKernelGetAttribute | 149264 | 10.11 |
| 4 | cuLaunchKernelEx | 74536 | 312.70 |
| 4 | cudaLaunchKernelExC_v11060 | 4550 | 17.15 |
| 4 | cuKernelGetAttribute | 149072 | 10.34 |
| 7 | cuLaunchKernelEx | 74380 | 249.34 |
| 7 | cudaLaunchKernelExC_v11060 | 4462 | 13.83 |
| 7 | cuKernelGetAttribute | 148760 | 10.16 |
| 6 | cuLaunchKernelEx | 74528 | 387.53 |
| 6 | cudaLaunchKernelExC_v11060 | 4546 | 22.14 |
| 6 | cuKernelGetAttribute | 149056 | 12.29 |
| 5 | cuLaunchKernelEx | 74504 | 252.76 |
| 5 | cudaLaunchKernelExC_v11060 | 4480 | 13.83 |
| 5 | cuKernelGetAttribute | 149008 | 10.34 |
| 2 | cuLaunchKernelEx | 74540 | 252.05 |
| 2 | cudaLaunchKernelExC_v11060 | 4458 | 13.62 |
| 2 | cuKernelGetAttribute | 149080 | 10.01 |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3006.27 | 4413.63 |
| gpu_idle | 1453.75 | 1453.75 |
| expert_gemm_path | 1189.22 | 2119.51 |
| fsdp_gather:expert | 1151.18 | 4466.02 |
| other_compute | 526.18 | 868.04 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3132.79 | 4558.56 |
| gpu_idle | 1463.30 | 1463.30 |
| fsdp_gather:expert | 1128.70 | 4554.83 |
| expert_gemm_path | 1095.52 | 2149.41 |
| other_compute | 526.63 | 937.05 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3145.78 | 4587.00 |
| gpu_idle | 1694.92 | 1694.92 |
| expert_gemm_path | 1160.29 | 2167.70 |
| fsdp_gather:expert | 1048.49 | 4475.53 |
| other_compute | 529.19 | 925.26 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3127.95 | 4553.08 |
| gpu_idle | 1548.80 | 1548.80 |
| expert_gemm_path | 1115.91 | 2181.74 |
| fsdp_gather:expert | 1020.23 | 4454.37 |
| other_compute | 524.27 | 913.96 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3155.84 | 4599.05 |
| gpu_idle | 1216.78 | 1216.78 |
| expert_gemm_path | 1170.73 | 2154.75 |
| fsdp_gather:expert | 1102.26 | 4622.46 |
| other_compute | 534.37 | 945.67 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3102.17 | 4534.86 |
| gpu_idle | 1749.93 | 1749.93 |
| expert_gemm_path | 1165.77 | 2200.65 |
| fsdp_gather:expert | 794.62 | 4220.92 |
| other_compute | 526.50 | 914.49 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3127.00 | 4546.23 |
| gpu_idle | 1278.60 | 1278.60 |
| expert_gemm_path | 1130.45 | 2131.95 |
| fsdp_gather:expert | 1118.13 | 4606.64 |
| other_compute | 525.60 | 917.22 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3138.99 | 4574.29 |
| gpu_idle | 1351.46 | 1351.46 |
| fsdp_gather:expert | 1168.86 | 4567.65 |
| expert_gemm_path | 1144.75 | 2140.78 |
| other_compute | 526.73 | 871.40 |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
