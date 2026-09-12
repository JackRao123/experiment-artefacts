# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/nsys_workflow_20260911/devbox-ep1-grouped-async/timing.sqlite`
- Ranks: 5, 0, 1, 3, 6, 4, 2, 7; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: 1.958510305603367%.
- Hardware metrics present: False; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1 | 11937.45 | 11527.90 | 409.56 | 2756.51 | 538.93 | 5892.82 | 2329.36 | 654.44 |
| 0 | 1 | 11955.50 | 11211.57 | 743.93 | 2934.14 | 554.30 | 5645.41 | 2171.56 | 676.56 |
| 1 | 1 | 11937.62 | 11105.21 | 832.41 | 2636.30 | 534.60 | 5328.97 | 1785.44 | 517.19 |
| 3 | 1 | 11955.09 | 11526.35 | 428.75 | 2957.35 | 552.43 | 5777.80 | 2509.94 | 514.50 |
| 6 | 1 | 11935.31 | 11520.35 | 414.97 | 2848.93 | 552.47 | 5770.14 | 2416.13 | 570.43 |
| 4 | 1 | 11943.25 | 11507.50 | 435.75 | 2720.92 | 557.56 | 5829.89 | 2267.41 | 616.31 |
| 2 | 1 | 11949.99 | 11518.08 | 431.91 | 2846.15 | 541.99 | 5688.86 | 2395.88 | 484.00 |
| 7 | 1 | 11942.86 | 11514.67 | 428.19 | 2420.40 | 548.91 | 5365.11 | 1973.96 | 292.52 |

### Capture diagnostics

- Profile median is outside the observed control range: instrumentation, remaining warmup, or workload drift may affect attribution.
- 8× on captured trainer ranks: Not all NVTX events might have been collected.
- 25× on other/helper processes: Not all NVTX events might have been collected.
- 25× on other/helper processes: No NVTX events collected. Does the process use NVTX?
- 8× on captured trainer ranks: Not all CUDA events might have been collected.
- 17× on other/helper processes: Not all CUDA events might have been collected.
- 25× on other/helper processes: No CUDA events collected. Does the process use CUDA?
- 8× on other/helper processes: CUDA profiling might have not been started correctly.

Capture completeness is qualified by these warnings. Complete step anchors and runtime links do not prove zero event loss.

## Expert-weight prefetch arrivals

Arrival relative to preceding-block compute completion; not a zero-communication speedup estimate.

| Rank | Step | Attributed gathers | Ready by preceding block end | Status |
|---|---:|---:|---|---|
| 5 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 0 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 1 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 3 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 6 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 4 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 2 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 7 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |

## Forward expert-wrapper CUDA API costs

Original + recomputed forwards, averaged per FB. API residence can overlap GPU work; it is not additive or automatically recoverable time.

| Rank | API | Calls/FB | Host API ms/FB |
|---|---|---:|---:|
| 5 | cudaLaunchKernel_v7000 | 900 | 6.13 |
| 5 | cudaMemcpyAsync_v3020 | 300 | 3.96 |
| 5 | cudaLaunchKernelExC_v11060 | 300 | 2.25 |
| 0 | cudaLaunchKernel_v7000 | 900 | 6.60 |
| 0 | cudaMemcpyAsync_v3020 | 300 | 4.38 |
| 0 | cudaLaunchKernelExC_v11060 | 300 | 2.59 |
| 1 | cudaLaunchKernel_v7000 | 900 | 6.83 |
| 1 | cudaMemcpyAsync_v3020 | 300 | 4.18 |
| 1 | cudaLaunchKernelExC_v11060 | 300 | 2.51 |
| 3 | cudaLaunchKernel_v7000 | 900 | 6.27 |
| 3 | cudaMemcpyAsync_v3020 | 300 | 4.48 |
| 3 | cudaLaunchKernelExC_v11060 | 300 | 2.26 |
| 6 | cudaLaunchKernel_v7000 | 900 | 5.97 |
| 6 | cudaMemcpyAsync_v3020 | 300 | 3.66 |
| 6 | cudaLaunchKernelExC_v11060 | 300 | 2.17 |
| 4 | cudaLaunchKernel_v7000 | 900 | 6.02 |
| 4 | cudaMemcpyAsync_v3020 | 300 | 4.49 |
| 4 | cudaLaunchKernelExC_v11060 | 300 | 2.24 |
| 2 | cudaLaunchKernel_v7000 | 900 | 7.28 |
| 2 | cudaMemcpyAsync_v3020 | 300 | 4.18 |
| 2 | cudaLaunchKernelExC_v11060 | 300 | 2.51 |
| 7 | cudaLaunchKernel_v7000 | 900 | 7.19 |
| 7 | cudaMemcpyAsync_v3020 | 300 | 4.23 |
| 7 | cudaLaunchKernelExC_v11060 | 300 | 2.57 |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3274.19 | 4806.63 |
| fsdp_gather:expert | 1095.50 | 5074.78 |
| expert_gemm_path | 1029.34 | 2396.67 |
| cp_communication | 619.59 | 983.19 |
| other_compute | 558.91 | 934.68 |

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3181.03 | 4697.16 |
| fsdp_gather:expert | 1155.66 | 5011.23 |
| expert_gemm_path | 1049.26 | 2414.53 |
| gpu_idle | 743.93 | 743.93 |
| other_compute | 569.02 | 870.20 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3352.53 | 4889.64 |
| expert_gemm_path | 1093.67 | 2439.41 |
| fsdp_gather:expert | 1056.50 | 4980.30 |
| gpu_idle | 832.41 | 832.41 |
| other_compute | 578.49 | 946.81 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3299.32 | 4791.84 |
| expert_gemm_path | 1131.59 | 2326.47 |
| fsdp_gather:expert | 1086.62 | 4841.52 |
| cp_communication | 601.15 | 849.63 |
| other_compute | 575.81 | 873.11 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3236.97 | 4722.52 |
| expert_gemm_path | 1126.73 | 2343.16 |
| fsdp_gather:expert | 934.41 | 4793.56 |
| cp_communication | 637.26 | 905.56 |
| other_compute | 619.18 | 989.53 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3259.86 | 4776.23 |
| expert_gemm_path | 1048.53 | 2391.10 |
| fsdp_gather:expert | 1009.54 | 4976.36 |
| cp_communication | 632.26 | 952.87 |
| other_compute | 579.92 | 997.25 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3311.19 | 4799.14 |
| expert_gemm_path | 1145.52 | 2351.64 |
| fsdp_gather:expert | 1012.99 | 4783.54 |
| other_compute | 595.05 | 909.58 |
| cp_communication | 585.09 | 823.78 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3333.18 | 4874.87 |
| expert_gemm_path | 1205.39 | 2387.96 |
| fsdp_gather:expert | 765.55 | 4488.67 |
| other_compute | 632.84 | 1003.64 |
| cp_communication | 568.97 | 635.66 |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
