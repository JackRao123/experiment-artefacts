# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/nsys_workflow_20260911/ep8-te/timing.sqlite`
- Ranks: 0, 1, 5, 6, 7, 3, 2, 4; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: 0.9520141124372605%.
- Hardware metrics present: False; this timing report does not infer saturation.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3 | 10423.58 | 10164.43 | 259.14 | 3468.21 | 357.36 | 3269.72 | 3194.42 | 2933.54 |
| 1 | 3 | 10418.84 | 10160.46 | 258.38 | 3266.26 | 315.82 | 3069.18 | 2993.38 | 2777.76 |
| 5 | 3 | 10415.69 | 10166.01 | 249.67 | 3348.88 | 301.76 | 3158.79 | 3085.00 | 2876.61 |
| 6 | 3 | 10419.46 | 10154.81 | 264.65 | 3092.70 | 333.63 | 2891.50 | 2813.95 | 2583.33 |
| 7 | 3 | 10414.58 | 10154.32 | 260.26 | 3425.82 | 330.22 | 3228.15 | 3151.55 | 2920.67 |
| 3 | 3 | 10418.77 | 10163.46 | 255.32 | 3314.93 | 328.64 | 3121.58 | 3044.93 | 2811.40 |
| 2 | 3 | 10416.76 | 10166.85 | 249.92 | 3379.22 | 336.37 | 3193.91 | 3114.66 | 2868.48 |
| 4 | 3 | 10414.17 | 10154.94 | 259.23 | 3379.36 | 303.14 | 3177.86 | 3105.95 | 2899.41 |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3983.88 | 4046.25 |
| expert_dispatch_combine | 2637.55 | 2787.27 |
| expert_gemm_path | 1278.85 | 1278.85 |
| other_compute | 707.26 | 717.26 |
| other_gemm | 566.87 | 571.84 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4124.94 | 4188.03 |
| expert_dispatch_combine | 2579.59 | 2729.68 |
| expert_gemm_path | 1296.09 | 1296.09 |
| other_compute | 748.89 | 758.68 |
| other_gemm | 568.02 | 572.96 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4132.81 | 4192.12 |
| expert_dispatch_combine | 2657.28 | 2811.94 |
| expert_gemm_path | 1229.55 | 1229.55 |
| other_compute | 736.08 | 747.57 |
| other_gemm | 560.50 | 565.66 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4117.78 | 4181.43 |
| expert_dispatch_combine | 2385.63 | 2539.03 |
| expert_gemm_path | 1457.24 | 1457.24 |
| other_compute | 764.16 | 775.12 |
| other_gemm | 561.06 | 565.99 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4099.72 | 4162.08 |
| expert_dispatch_combine | 2694.56 | 2845.05 |
| expert_gemm_path | 1195.61 | 1195.61 |
| other_compute | 731.82 | 742.83 |
| other_gemm | 552.72 | 557.93 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4099.88 | 4164.03 |
| expert_dispatch_combine | 2591.93 | 2745.71 |
| expert_gemm_path | 1309.41 | 1309.41 |
| other_compute | 747.40 | 756.77 |
| other_gemm | 557.15 | 562.45 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4084.45 | 4148.74 |
| expert_dispatch_combine | 2636.28 | 2787.73 |
| expert_gemm_path | 1247.03 | 1247.03 |
| other_compute | 738.91 | 750.71 |
| other_gemm | 551.82 | 557.15 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4138.08 | 4194.94 |
| expert_dispatch_combine | 2683.52 | 2837.62 |
| expert_gemm_path | 1233.29 | 1233.29 |
| other_compute | 738.87 | 750.51 |
| other_gemm | 551.87 | 557.25 |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
