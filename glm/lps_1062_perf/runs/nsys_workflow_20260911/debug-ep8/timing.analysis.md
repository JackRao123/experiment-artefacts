# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/nsys_workflow_20260911/debug-ep8/timing.sqlite`
- Ranks: 0, 3, 6, 2, 5, 4, 1, 7; explicit NVTX rank/forward-backward anchors.
- Same-run capture slowdown: -4.078306698808309%.
- Hardware metrics present: False; this timing report does not infer saturation.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 508.30 | 484.24 | 24.06 | 90.22 | 20.53 | 68.55 | 64.84 | 52.17 |
| 3 | 1 | 503.71 | 484.25 | 19.47 | 133.35 | 23.41 | 119.75 | 112.62 | 93.15 |
| 6 | 1 | 501.02 | 480.56 | 20.47 | 115.79 | 24.11 | 100.40 | 94.11 | 75.73 |
| 2 | 1 | 498.93 | 479.69 | 19.25 | 118.47 | 23.68 | 105.03 | 97.97 | 79.07 |
| 5 | 1 | 498.40 | 479.08 | 19.33 | 91.79 | 20.68 | 75.87 | 71.24 | 55.69 |
| 4 | 1 | 498.21 | 478.17 | 20.05 | 120.30 | 22.82 | 103.49 | 99.04 | 82.31 |
| 1 | 1 | 497.85 | 478.02 | 19.83 | 114.29 | 23.25 | 98.60 | 93.19 | 76.12 |
| 7 | 1 | 496.72 | 476.43 | 20.30 | 114.51 | 24.34 | 97.53 | 93.01 | 74.72 |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 185.09 | 188.47 |
| other_gemm | 69.23 | 69.32 |
| expert_gemm_path | 55.04 | 55.04 |
| other_compute | 54.62 | 55.04 |
| expert_dispatch_combine | 39.32 | 42.22 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 179.82 | 183.40 |
| expert_dispatch_combine | 74.72 | 77.56 |
| other_gemm | 66.53 | 66.61 |
| other_compute | 47.30 | 50.92 |
| lm_head | 35.25 | 35.25 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 181.62 | 184.98 |
| other_gemm | 67.27 | 67.35 |
| expert_dispatch_combine | 62.79 | 65.69 |
| other_compute | 49.35 | 52.36 |
| lm_head | 35.49 | 35.49 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 181.67 | 185.31 |
| expert_dispatch_combine | 67.84 | 70.70 |
| other_gemm | 66.29 | 66.37 |
| other_compute | 48.45 | 51.93 |
| lm_head | 35.18 | 35.18 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 190.73 | 192.53 |
| other_gemm | 67.13 | 67.22 |
| other_compute | 51.15 | 54.07 |
| expert_dispatch_combine | 50.87 | 53.73 |
| expert_gemm_path | 44.80 | 44.80 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 185.29 | 186.91 |
| expert_dispatch_combine | 71.84 | 74.71 |
| other_gemm | 66.55 | 66.63 |
| other_compute | 48.66 | 51.55 |
| lm_head | 35.24 | 35.24 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 181.87 | 185.05 |
| other_gemm | 68.81 | 68.89 |
| expert_dispatch_combine | 67.36 | 70.24 |
| other_compute | 49.47 | 51.76 |
| lm_head | 35.60 | 35.60 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 183.20 | 185.30 |
| other_gemm | 66.90 | 66.99 |
| expert_dispatch_combine | 66.90 | 69.82 |
| other_compute | 49.67 | 52.17 |
| lm_head | 35.28 | 35.28 |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
