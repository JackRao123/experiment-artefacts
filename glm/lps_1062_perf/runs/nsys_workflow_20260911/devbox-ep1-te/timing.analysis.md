# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/nsys_workflow_20260911/devbox-ep1-te/timing.sqlite`
- Ranks: 0, 1, 6, 4, 2, 7, 3, 5; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: 6.66681312059243%.
- Hardware metrics present: False; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 12491.14 | 10766.06 | 1725.08 | 3987.39 | 2563.42 | 5337.45 | 2243.46 | 392.28 |
| 1 | 1 | 12475.43 | 10518.51 | 1956.92 | 3752.46 | 2177.23 | 4877.99 | 1776.92 | 623.53 |
| 6 | 1 | 12451.35 | 11210.43 | 1240.92 | 3826.81 | 2232.51 | 5657.14 | 2568.12 | 584.51 |
| 4 | 1 | 12467.17 | 11163.03 | 1304.14 | 3836.37 | 2251.15 | 5657.95 | 2514.55 | 532.48 |
| 2 | 1 | 12469.26 | 11022.90 | 1446.36 | 3830.13 | 2381.52 | 5441.83 | 2365.33 | 461.86 |
| 7 | 1 | 12461.65 | 11295.18 | 1166.47 | 3748.40 | 2142.93 | 5591.22 | 2563.83 | 645.96 |
| 3 | 1 | 12462.52 | 10955.36 | 1507.16 | 3838.97 | 2431.23 | 5430.91 | 2313.16 | 425.91 |
| 5 | 1 | 12463.43 | 11290.36 | 1173.07 | 3830.37 | 2186.15 | 5677.51 | 2639.80 | 699.89 |

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
| 1 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 6 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 4 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 2 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 7 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 3 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 5 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3007.40 | 4419.98 |
| gpu_idle | 1725.08 | 1725.08 |
| expert_gemm_path | 1142.73 | 2172.56 |
| fsdp_gather:expert | 895.75 | 4192.93 |
| cp_communication | 861.55 | 925.94 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3148.92 | 4579.65 |
| gpu_idle | 1956.92 | 1956.92 |
| expert_gemm_path | 1228.68 | 2141.09 |
| fsdp_gather:expert | 754.03 | 4318.72 |
| other_compute | 528.60 | 998.31 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3113.73 | 4536.35 |
| gpu_idle | 1240.92 | 1240.92 |
| expert_gemm_path | 1138.12 | 2134.09 |
| fsdp_gather:expert | 910.30 | 4471.85 |
| cp_communication | 757.08 | 1070.88 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3122.32 | 4542.50 |
| gpu_idle | 1304.14 | 1304.14 |
| expert_gemm_path | 1102.45 | 2159.84 |
| fsdp_gather:expert | 937.57 | 4488.82 |
| cp_communication | 766.82 | 1078.01 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3132.59 | 4563.05 |
| gpu_idle | 1446.36 | 1446.36 |
| expert_gemm_path | 1162.72 | 2169.27 |
| fsdp_gather:expert | 898.44 | 4323.04 |
| cp_communication | 755.58 | 957.86 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3161.21 | 4597.35 |
| expert_gemm_path | 1238.58 | 2140.30 |
| gpu_idle | 1166.47 | 1166.47 |
| fsdp_gather:expert | 850.46 | 4400.82 |
| cp_communication | 711.63 | 981.51 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3124.92 | 4548.15 |
| gpu_idle | 1507.16 | 1507.16 |
| expert_gemm_path | 1124.58 | 2187.83 |
| fsdp_gather:expert | 901.39 | 4330.35 |
| cp_communication | 757.13 | 979.40 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3125.08 | 4540.34 |
| expert_gemm_path | 1204.29 | 2108.24 |
| gpu_idle | 1173.07 | 1173.07 |
| fsdp_gather:expert | 841.85 | 4423.56 |
| cp_communication | 758.25 | 1043.67 |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
