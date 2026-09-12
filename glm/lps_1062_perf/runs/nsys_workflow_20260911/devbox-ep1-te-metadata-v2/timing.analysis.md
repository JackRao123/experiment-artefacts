# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `devbox-ep1-te-metadata-v2/timing.sqlite`
- Ranks: 0, 5, 7, 4, 1, 6, 3, 2; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: unavailable%.
- Hardware metrics present: False; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 12304.28 | 10997.71 | 1306.57 | 3739.83 | 1840.96 | 5595.34 | 2414.68 | 840.14 |
| 5 | 1 | 12290.82 | 11658.72 | 632.10 | 3617.92 | 1608.57 | 5996.48 | 2968.27 | 1049.40 |
| 7 | 1 | 12288.48 | 11597.79 | 690.69 | 3444.81 | 1501.68 | 5845.76 | 2736.16 | 985.68 |
| 4 | 1 | 12175.96 | 11229.79 | 946.17 | 3488.46 | 1840.28 | 5686.60 | 2524.69 | 728.87 |
| 1 | 1 | 12180.56 | 11290.37 | 890.19 | 3448.14 | 1635.48 | 5590.80 | 2539.22 | 880.65 |
| 6 | 1 | 12168.26 | 11142.51 | 1025.75 | 3484.74 | 1940.88 | 5615.77 | 2441.35 | 773.02 |
| 3 | 1 | 12171.69 | 10642.14 | 1529.55 | 3514.06 | 1919.48 | 5066.92 | 1965.79 | 718.65 |
| 2 | 1 | 12153.72 | 10667.50 | 1486.21 | 3442.03 | 2007.49 | 4960.11 | 1937.21 | 586.37 |

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
| 5 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 7 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 4 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 1 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 6 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 3 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 2 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |

## Forward expert-wrapper CUDA API costs

Original + recomputed forwards, averaged per FB. API residence can overlap GPU work; it is not additive or automatically recoverable time.

| Rank | API | Calls/FB | Host API ms/FB |
|---|---|---:|---:|
| 0 | cuLaunchKernelEx | 74928 | 329.34 |
| 0 | cudaLaunchKernelExC_v11060 | 4112 | 16.12 |
| 0 | cuKernelGetAttribute | 149856 | 10.30 |
| 5 | cuLaunchKernelEx | 74764 | 258.77 |
| 5 | cudaLaunchKernelExC_v11060 | 4110 | 13.13 |
| 5 | cuKernelGetAttribute | 149528 | 9.98 |
| 7 | cuLaunchKernelEx | 74676 | 271.26 |
| 7 | cudaLaunchKernelExC_v11060 | 4052 | 13.46 |
| 7 | cuKernelGetAttribute | 149352 | 10.34 |
| 4 | cuLaunchKernelEx | 74776 | 247.56 |
| 4 | cudaLaunchKernelExC_v11060 | 4172 | 12.76 |
| 4 | cuKernelGetAttribute | 149552 | 10.15 |
| 1 | cuLaunchKernelEx | 74736 | 274.86 |
| 1 | cudaLaunchKernelExC_v11060 | 4098 | 13.74 |
| 1 | cuKernelGetAttribute | 149472 | 9.86 |
| 6 | cuLaunchKernelEx | 74636 | 248.95 |
| 6 | cudaLaunchKernelExC_v11060 | 4120 | 13.22 |
| 6 | cuKernelGetAttribute | 149272 | 10.00 |
| 3 | cuLaunchKernelEx | 74804 | 335.25 |
| 3 | cudaLaunchKernelExC_v11060 | 4254 | 17.73 |
| 3 | cuKernelGetAttribute | 149608 | 10.22 |
| 2 | cuLaunchKernelEx | 74776 | 335.73 |
| 2 | cudaLaunchKernelExC_v11060 | 4102 | 16.35 |
| 2 | cuKernelGetAttribute | 149552 | 10.36 |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3017.20 | 4431.28 |
| gpu_idle | 1306.57 | 1306.57 |
| expert_gemm_path | 1125.34 | 2193.60 |
| fsdp_gather:expert | 1089.40 | 4467.80 |
| cp_communication | 749.57 | 843.24 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3149.95 | 4595.44 |
| expert_gemm_path | 1217.36 | 2111.18 |
| fsdp_gather:expert | 1041.91 | 4597.30 |
| cp_communication | 934.62 | 1170.40 |
| gpu_idle | 632.10 | 632.10 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3186.30 | 4653.32 |
| expert_gemm_path | 1250.28 | 2144.25 |
| fsdp_gather:expert | 972.44 | 4581.82 |
| cp_communication | 806.72 | 1016.96 |
| gpu_idle | 690.69 | 690.69 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3144.37 | 4595.10 |
| expert_gemm_path | 1109.22 | 2165.93 |
| fsdp_gather:expert | 1020.84 | 4515.73 |
| cp_communication | 954.99 | 1188.20 |
| gpu_idle | 946.17 | 946.17 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3184.29 | 4647.26 |
| expert_gemm_path | 1229.84 | 2130.32 |
| fsdp_gather:expert | 962.59 | 4486.23 |
| gpu_idle | 890.19 | 890.19 |
| cp_communication | 758.58 | 944.73 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3120.10 | 4571.67 |
| expert_gemm_path | 1100.51 | 2159.89 |
| gpu_idle | 1025.75 | 1025.75 |
| cp_communication | 966.90 | 1216.60 |
| fsdp_gather:expert | 843.33 | 4394.31 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3153.82 | 4606.14 |
| gpu_idle | 1529.55 | 1529.55 |
| expert_gemm_path | 1160.90 | 2159.74 |
| fsdp_gather:expert | 940.04 | 4389.27 |
| other_compute | 530.19 | 896.68 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3162.09 | 4620.97 |
| gpu_idle | 1486.21 | 1486.21 |
| expert_gemm_path | 1241.78 | 2128.91 |
| fsdp_gather:expert | 928.12 | 4350.66 |
| other_compute | 535.46 | 929.67 |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
