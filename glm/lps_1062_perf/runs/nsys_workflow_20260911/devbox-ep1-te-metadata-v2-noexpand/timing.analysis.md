# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/nsys_workflow_20260911/devbox-ep1-te-metadata-v2-noexpand/timing.sqlite`
- Ranks: 0, 5, 7, 3, 4, 1, 2, 6; explicit NVTX rank/forward-backward anchors.
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
| 0 | 1 | 11367.32 | 10510.20 | 857.12 | 2945.57 | 1346.10 | 5020.75 | 2069.66 | 793.11 |
| 5 | 1 | 11295.27 | 10529.21 | 766.05 | 2572.12 | 1154.48 | 4988.71 | 1788.33 | 697.61 |
| 7 | 1 | 11285.19 | 10813.22 | 471.96 | 2251.90 | 841.98 | 4865.75 | 1761.78 | 718.65 |
| 3 | 1 | 11249.47 | 10416.79 | 832.68 | 2606.99 | 1274.08 | 4819.65 | 1755.86 | 609.31 |
| 4 | 1 | 11251.09 | 10548.91 | 702.18 | 2584.04 | 1122.49 | 4963.23 | 1864.32 | 705.65 |
| 1 | 1 | 11253.57 | 10581.54 | 672.03 | 2333.41 | 1063.55 | 4778.11 | 1642.69 | 564.22 |
| 2 | 1 | 11227.79 | 10410.67 | 817.12 | 2512.04 | 1230.23 | 4798.65 | 1676.35 | 549.06 |
| 6 | 1 | 11253.15 | 10488.08 | 765.07 | 2482.18 | 1110.47 | 4910.22 | 1699.28 | 650.51 |

### Capture diagnostics

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
| 0 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 5 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 7 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 3 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 4 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 1 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 2 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 6 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |

## Forward expert-wrapper CUDA API costs

Original + recomputed forwards, averaged per FB. API residence can overlap GPU work; it is not additive or automatically recoverable time.

| Rank | API | Calls/FB | Host API ms/FB |
|---|---|---:|---:|
| 0 | cuLaunchKernelEx | 74940 | 278.57 |
| 0 | cudaLaunchKernelExC_v11060 | 4186 | 14.12 |
| 0 | cuKernelGetAttribute | 149880 | 10.34 |
| 5 | cuLaunchKernelEx | 74756 | 251.48 |
| 5 | cudaLaunchKernelExC_v11060 | 4244 | 13.01 |
| 5 | cuKernelGetAttribute | 149512 | 10.27 |
| 7 | cuLaunchKernelEx | 74616 | 275.23 |
| 7 | cudaLaunchKernelExC_v11060 | 4072 | 13.76 |
| 7 | cuKernelGetAttribute | 149232 | 10.01 |
| 3 | cuLaunchKernelEx | 74808 | 333.89 |
| 3 | cudaLaunchKernelExC_v11060 | 4340 | 17.21 |
| 3 | cuKernelGetAttribute | 149616 | 10.00 |
| 4 | cuLaunchKernelEx | 74684 | 250.49 |
| 4 | cudaLaunchKernelExC_v11060 | 4142 | 12.83 |
| 4 | cuKernelGetAttribute | 149368 | 10.01 |
| 1 | cuLaunchKernelEx | 74760 | 281.12 |
| 1 | cudaLaunchKernelExC_v11060 | 4114 | 13.97 |
| 1 | cuKernelGetAttribute | 149520 | 10.39 |
| 2 | cuLaunchKernelEx | 74712 | 333.13 |
| 2 | cudaLaunchKernelExC_v11060 | 4052 | 16.43 |
| 2 | cuKernelGetAttribute | 149424 | 9.89 |
| 6 | cuLaunchKernelEx | 74668 | 250.41 |
| 6 | cudaLaunchKernelExC_v11060 | 4108 | 12.65 |
| 6 | cuKernelGetAttribute | 149336 | 10.53 |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3028.36 | 4437.36 |
| expert_gemm_path | 1191.98 | 2118.94 |
| fsdp_gather:expert | 907.94 | 4234.06 |
| gpu_idle | 857.12 | 857.12 |
| other_compute | 534.16 | 870.90 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3147.84 | 4586.88 |
| expert_gemm_path | 1100.26 | 2162.83 |
| fsdp_gather:expert | 804.65 | 4385.41 |
| gpu_idle | 766.05 | 766.05 |
| other_compute | 530.51 | 948.23 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3197.75 | 4660.82 |
| expert_gemm_path | 1254.24 | 2143.73 |
| fsdp_gather:expert | 737.35 | 4270.40 |
| other_compute | 537.30 | 1000.10 |
| gpu_idle | 471.96 | 471.96 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3157.18 | 4600.55 |
| expert_gemm_path | 1159.31 | 2155.07 |
| fsdp_gather:expert | 841.44 | 4240.83 |
| gpu_idle | 832.68 | 832.68 |
| other_compute | 530.10 | 873.41 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3147.32 | 4596.14 |
| expert_gemm_path | 1148.06 | 2141.59 |
| fsdp_gather:expert | 851.10 | 4361.76 |
| gpu_idle | 702.18 | 702.18 |
| other_compute | 530.11 | 905.28 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3182.69 | 4640.05 |
| expert_gemm_path | 1172.28 | 2167.55 |
| fsdp_gather:expert | 777.72 | 4262.39 |
| gpu_idle | 672.03 | 672.03 |
| other_compute | 534.00 | 931.39 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3166.70 | 4616.06 |
| expert_gemm_path | 1132.39 | 2190.04 |
| fsdp_gather:expert | 860.65 | 4238.33 |
| gpu_idle | 817.12 | 817.12 |
| other_compute | 528.55 | 858.36 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3127.61 | 4571.00 |
| expert_gemm_path | 1092.99 | 2160.90 |
| fsdp_gather:expert | 820.02 | 4372.29 |
| gpu_idle | 765.07 | 765.07 |
| other_compute | 530.38 | 948.04 |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
