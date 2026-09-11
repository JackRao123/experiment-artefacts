# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `/root/glm53-fsdp-nsys-131k-20260911/devbox-debug/timing.sqlite`
- Ranks: 0, 3, 6, 1, 5, 7, 2, 4; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: -67.79319144885288%.
- Hardware metrics present: False; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 511.59 | 487.31 | 24.28 | 90.91 | 29.30 | 70.67 | 65.32 | 44.74 |
| 3 | 1 | 504.74 | 485.33 | 19.41 | 132.09 | 29.49 | 116.85 | 111.42 | 89.06 |
| 6 | 1 | 502.40 | 483.58 | 18.81 | 115.60 | 30.05 | 100.95 | 95.54 | 72.61 |
| 1 | 1 | 502.00 | 482.16 | 19.84 | 112.73 | 29.06 | 95.80 | 91.69 | 71.86 |
| 5 | 1 | 502.02 | 483.19 | 18.83 | 89.41 | 29.03 | 73.76 | 69.35 | 47.91 |
| 7 | 1 | 501.32 | 482.04 | 19.28 | 108.20 | 29.35 | 90.98 | 87.67 | 67.84 |
| 2 | 1 | 500.41 | 478.28 | 22.14 | 113.84 | 31.81 | 94.85 | 90.45 | 69.01 |
| 4 | 1 | 500.29 | 478.96 | 21.33 | 114.89 | 29.78 | 96.07 | 92.33 | 71.56 |

### Capture diagnostics

- Fewer than three controls; steady-state variability is not established.
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
| 0 | 0 | 0 | unknown | no attributed expert-weight gathers; check group annotation coverage |
| 3 | 0 | 0 | unknown | no attributed expert-weight gathers; check group annotation coverage |
| 6 | 0 | 0 | unknown | no attributed expert-weight gathers; check group annotation coverage |
| 1 | 0 | 0 | unknown | no attributed expert-weight gathers; check group annotation coverage |
| 5 | 0 | 0 | unknown | no attributed expert-weight gathers; check group annotation coverage |
| 7 | 0 | 0 | unknown | no attributed expert-weight gathers; check group annotation coverage |
| 2 | 0 | 0 | unknown | no attributed expert-weight gathers; check group annotation coverage |
| 4 | 0 | 0 | unknown | no attributed expert-weight gathers; check group annotation coverage |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 184.80 | 188.72 |
| other_gemm | 67.69 | 67.79 |
| expert_gemm_path | 61.46 | 61.46 |
| other_compute | 53.30 | 54.81 |
| expert_dispatch_combine | 40.87 | 43.80 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 179.93 | 183.77 |
| expert_dispatch_combine | 81.94 | 84.80 |
| other_gemm | 66.64 | 66.72 |
| other_compute | 48.70 | 50.36 |
| lm_head | 35.38 | 35.38 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 179.80 | 183.54 |
| expert_dispatch_combine | 67.63 | 70.55 |
| other_gemm | 66.98 | 67.07 |
| other_compute | 50.43 | 52.19 |
| expert_gemm_path | 37.30 | 37.30 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 182.36 | 185.85 |
| expert_dispatch_combine | 70.08 | 72.99 |
| other_gemm | 69.25 | 69.35 |
| other_compute | 51.68 | 52.37 |
| lm_head | 36.05 | 36.05 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 189.49 | 191.72 |
| other_gemm | 67.60 | 67.70 |
| other_compute | 51.80 | 54.07 |
| expert_gemm_path | 51.21 | 51.21 |
| expert_dispatch_combine | 49.51 | 52.41 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 184.44 | 187.28 |
| other_gemm | 69.33 | 69.43 |
| expert_dispatch_combine | 67.31 | 70.26 |
| other_compute | 51.47 | 52.03 |
| expert_gemm_path | 36.66 | 36.66 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 182.78 | 185.85 |
| expert_dispatch_combine | 71.28 | 74.17 |
| other_gemm | 67.96 | 68.06 |
| other_compute | 50.04 | 51.47 |
| lm_head | 35.71 | 35.71 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 188.07 | 189.81 |
| expert_dispatch_combine | 72.66 | 75.53 |
| other_gemm | 66.74 | 66.84 |
| other_compute | 49.77 | 51.84 |
| lm_head | 35.33 | 35.33 |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
