# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/nsys_workflow_20260911/ep1-te/timing.sqlite`
- Ranks: 5, 3, 1, 2, 6, 4, 0, 7; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: 3.779067327664909%.
- Hardware metrics present: False; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 3 | 11925.65 | 10580.69 | 1344.96 | 3095.14 | 1624.01 | 4940.31 | 1733.00 | 482.96 |
| 3 | 3 | 11922.23 | 10792.98 | 1129.26 | 2798.36 | 1357.79 | 4907.88 | 1650.93 | 458.05 |
| 1 | 3 | 11932.33 | 10502.14 | 1430.20 | 2818.05 | 1466.14 | 4711.64 | 1369.79 | 421.66 |
| 2 | 3 | 11913.79 | 10695.80 | 1217.99 | 2741.48 | 1378.26 | 4831.46 | 1505.32 | 436.68 |
| 6 | 3 | 11920.08 | 10826.80 | 1093.29 | 2818.43 | 1329.08 | 4947.24 | 1708.10 | 497.55 |
| 4 | 3 | 11922.57 | 10375.13 | 1547.44 | 3014.22 | 1681.21 | 4744.22 | 1449.88 | 422.11 |
| 0 | 3 | 11934.97 | 10402.92 | 1532.05 | 2892.31 | 1582.13 | 4587.24 | 1342.49 | 363.22 |
| 7 | 3 | 11924.49 | 10564.60 | 1359.88 | 2967.76 | 1548.82 | 4849.66 | 1590.87 | 445.08 |

### Capture diagnostics

- 25× on other/helper processes: Not all NVTX events might have been collected.
- 25× on other/helper processes: No NVTX events collected. Does the process use NVTX?
- 8× on captured trainer ranks: Not all NVTX events might have been collected.
- 17× on other/helper processes: Not all CUDA events might have been collected.
- 25× on other/helper processes: No CUDA events collected. Does the process use CUDA?
- 8× on captured trainer ranks: Not all CUDA events might have been collected.
- 8× on other/helper processes: CUDA profiling might have not been started correctly.

Capture completeness is qualified by these warnings. Complete step anchors and runtime links do not prove zero event loss.

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3142.42 | 4565.16 |
| collective_unclassified | 1292.48 | 4770.44 |
| expert_gemm_path | 1119.47 | 2172.90 |
| other_compute | 530.34 | 911.01 |
| other_gemm | 441.88 | 609.90 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3156.88 | 4585.70 |
| collective_unclassified | 1235.37 | 4739.46 |
| expert_gemm_path | 1222.37 | 2125.42 |
| other_compute | 529.34 | 914.11 |
| other_gemm | 440.01 | 609.53 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3157.76 | 4592.41 |
| expert_gemm_path | 1173.19 | 2168.00 |
| collective_unclassified | 1081.67 | 4639.37 |
| other_compute | 530.63 | 999.68 |
| other_gemm | 446.29 | 617.15 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3143.32 | 4572.07 |
| expert_gemm_path | 1216.55 | 2114.26 |
| collective_unclassified | 1114.44 | 4662.14 |
| other_compute | 529.84 | 993.31 |
| other_gemm | 435.04 | 601.85 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3125.44 | 4554.58 |
| collective_unclassified | 1253.93 | 4777.51 |
| expert_gemm_path | 1220.68 | 2118.73 |
| other_compute | 529.92 | 904.45 |
| other_gemm | 442.21 | 611.14 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3133.06 | 4557.01 |
| expert_gemm_path | 1134.09 | 2195.67 |
| collective_unclassified | 1117.01 | 4613.45 |
| other_compute | 527.36 | 956.66 |
| other_gemm | 435.38 | 603.09 |

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3053.47 | 4473.00 |
| expert_gemm_path | 1232.48 | 2161.21 |
| collective_unclassified | 1162.99 | 4490.26 |
| other_compute | 532.32 | 880.67 |
| other_gemm | 446.91 | 618.04 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3130.70 | 4556.50 |
| collective_unclassified | 1183.37 | 4682.78 |
| expert_gemm_path | 1156.62 | 2155.20 |
| other_compute | 528.12 | 923.60 |
| other_gemm | 437.09 | 605.80 |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
