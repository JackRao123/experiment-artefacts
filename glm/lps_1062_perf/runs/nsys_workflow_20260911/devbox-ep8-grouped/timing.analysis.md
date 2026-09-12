# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `devbox-ep8-grouped/timing.sqlite`
- Ranks: 0, 1, 7, 2, 6, 3, 4, 5; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: 0.6578082534485441%.
- Hardware metrics present: False; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 11084.59 | 10794.63 | 289.97 | 4011.13 | 841.74 | 3785.97 | 3706.03 | 2975.69 |
| 1 | 1 | 11077.29 | 10785.37 | 291.92 | 3706.83 | 796.57 | 3476.04 | 3399.91 | 2726.32 |
| 7 | 1 | 11077.36 | 10790.82 | 286.54 | 3763.95 | 780.03 | 3541.69 | 3463.12 | 2796.34 |
| 2 | 1 | 11075.41 | 10787.35 | 288.06 | 3839.94 | 819.95 | 3616.66 | 3536.96 | 2826.23 |
| 6 | 1 | 11072.13 | 10781.01 | 291.12 | 3508.56 | 828.32 | 3284.18 | 3202.78 | 2492.59 |
| 3 | 1 | 11073.41 | 10790.05 | 283.36 | 3815.89 | 823.58 | 3599.07 | 3517.93 | 2801.45 |
| 4 | 1 | 11074.77 | 10785.72 | 289.05 | 3767.17 | 770.04 | 3542.21 | 3463.49 | 2805.32 |
| 5 | 1 | 11070.36 | 10781.74 | 288.62 | 3854.98 | 790.03 | 3630.71 | 3551.73 | 2871.02 |

### Capture diagnostics

- 25× on other/helper processes: Not all NVTX events might have been collected.
- 25× on other/helper processes: No NVTX events collected. Does the process use NVTX?
- 8× on captured trainer ranks: Not all NVTX events might have been collected.
- 8× on other/helper processes: CUDA profiling might have not been started correctly.
- 25× on other/helper processes: No CUDA events collected. Does the process use CUDA?
- 17× on other/helper processes: Not all CUDA events might have been collected.
- 8× on captured trainer ranks: Not all CUDA events might have been collected.

Capture completeness is qualified by these warnings. Complete step anchors and runtime links do not prove zero event loss.

## Expert-weight prefetch arrivals

Arrival relative to preceding-block compute completion; not a zero-communication speedup estimate.

| Rank | Step | Attributed gathers | Ready by preceding block end | Status |
|---|---:|---:|---|---|
| 0 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 1 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 7 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 2 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 6 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 3 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 4 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 5 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3991.11 | 4056.55 |
| expert_dispatch_combine | 3079.70 | 3232.44 |
| expert_gemm_path | 1418.83 | 1418.83 |
| other_compute | 708.96 | 721.29 |
| other_gemm | 560.80 | 566.49 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4187.00 | 4252.06 |
| expert_dispatch_combine | 2947.34 | 3099.26 |
| expert_gemm_path | 1464.92 | 1464.92 |
| other_compute | 761.79 | 771.31 |
| other_gemm | 573.00 | 578.30 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4173.35 | 4238.26 |
| expert_dispatch_combine | 2979.09 | 3131.84 |
| expert_gemm_path | 1403.56 | 1403.56 |
| other_compute | 743.00 | 754.99 |
| other_gemm | 576.89 | 582.26 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4115.84 | 4181.23 |
| expert_dispatch_combine | 3015.36 | 3165.86 |
| expert_gemm_path | 1432.78 | 1432.78 |
| other_compute | 707.22 | 719.81 |
| other_gemm | 560.29 | 565.95 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4149.91 | 4216.73 |
| expert_dispatch_combine | 2709.84 | 2861.94 |
| expert_gemm_path | 1648.36 | 1648.36 |
| other_compute | 773.08 | 785.89 |
| other_gemm | 557.08 | 562.86 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4121.86 | 4186.90 |
| expert_dispatch_combine | 2996.05 | 3146.35 |
| expert_gemm_path | 1472.47 | 1472.47 |
| other_compute | 712.34 | 726.01 |
| other_gemm | 553.33 | 559.51 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4185.20 | 4248.99 |
| expert_dispatch_combine | 2960.10 | 3111.86 |
| expert_gemm_path | 1403.52 | 1403.52 |
| other_compute | 744.60 | 757.63 |
| other_gemm | 556.10 | 561.92 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4146.41 | 4209.55 |
| expert_dispatch_combine | 3026.55 | 3179.14 |
| expert_gemm_path | 1361.69 | 1361.69 |
| other_compute | 735.44 | 749.42 |
| other_gemm | 556.19 | 561.96 |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
