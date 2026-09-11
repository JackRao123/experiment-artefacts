# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/nsys_workflow_20260911/devbox-ep1-grouped/timing.sqlite`
- Ranks: 0, 7, 5, 6, 4, 3, 2, 1; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: 0.3834373318774009%.
- Hardware metrics present: False; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 11682.66 | 10563.40 | 1119.26 | 2995.28 | 1421.89 | 5025.38 | 1857.11 | 466.12 |
| 7 | 1 | 11634.16 | 10806.42 | 827.74 | 2469.49 | 1093.94 | 5053.66 | 1623.05 | 368.68 |
| 5 | 1 | 11642.05 | 10776.46 | 865.59 | 2697.78 | 1197.45 | 5130.87 | 1814.04 | 430.15 |
| 6 | 1 | 11623.60 | 10802.62 | 820.98 | 2712.22 | 1318.31 | 5077.36 | 1873.39 | 388.39 |
| 4 | 1 | 11636.34 | 10815.01 | 821.33 | 2612.22 | 1049.61 | 5216.01 | 1772.72 | 490.08 |
| 3 | 1 | 11645.08 | 10739.74 | 905.34 | 2788.92 | 1242.54 | 5122.86 | 1864.52 | 413.75 |
| 2 | 1 | 11644.92 | 10692.86 | 952.05 | 2762.72 | 1398.63 | 4965.49 | 1791.90 | 324.66 |
| 1 | 1 | 11648.68 | 10574.13 | 1074.55 | 2618.20 | 1380.26 | 4815.81 | 1524.83 | 298.27 |

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
| 7 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 5 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 6 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 4 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 3 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 2 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 1 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3078.15 | 4508.87 |
| expert_gemm_path | 1187.73 | 2295.99 |
| gpu_idle | 1119.26 | 1119.26 |
| fsdp_gather:expert | 829.63 | 4347.56 |
| other_compute | 533.12 | 876.65 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3255.64 | 4724.70 |
| expert_gemm_path | 1168.34 | 2415.78 |
| gpu_idle | 827.74 | 827.74 |
| fsdp_gather:expert | 762.57 | 4470.46 |
| other_compute | 543.95 | 966.38 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3193.66 | 4638.78 |
| expert_gemm_path | 1150.88 | 2337.68 |
| gpu_idle | 865.59 | 865.59 |
| fsdp_gather:expert | 812.69 | 4455.31 |
| other_compute | 534.84 | 934.69 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3185.46 | 4629.21 |
| expert_gemm_path | 1173.70 | 2260.65 |
| gpu_idle | 820.98 | 820.98 |
| fsdp_gather:expert | 775.31 | 4365.12 |
| other_compute | 533.54 | 923.24 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3194.50 | 4637.20 |
| expert_gemm_path | 1108.26 | 2378.27 |
| gpu_idle | 821.33 | 821.33 |
| fsdp_gather:expert | 815.03 | 4554.60 |
| other_compute | 533.25 | 978.59 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3208.00 | 4651.63 |
| expert_gemm_path | 1137.01 | 2315.12 |
| gpu_idle | 905.34 | 905.34 |
| fsdp_gather:expert | 878.22 | 4451.69 |
| other_compute | 536.42 | 888.59 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3229.23 | 4685.12 |
| expert_gemm_path | 1189.67 | 2267.36 |
| gpu_idle | 952.05 | 952.05 |
| fsdp_gather:expert | 780.91 | 4302.29 |
| other_compute | 537.76 | 892.00 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3251.09 | 4708.45 |
| expert_gemm_path | 1202.04 | 2288.37 |
| gpu_idle | 1074.55 | 1074.55 |
| fsdp_gather:expert | 684.58 | 4322.21 |
| other_compute | 543.28 | 1002.54 |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
