# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/nsys_workflow_20260911/devbox-ep8-te-metadata-v2/timing.sqlite`
- Ranks: 0, 5, 2, 7, 4, 1, 3, 6; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: 0.9298998839807116%.
- Hardware metrics present: False; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 10342.44 | 10138.55 | 203.89 | 3498.25 | 368.54 | 3359.76 | 3279.80 | 2997.86 |
| 5 | 1 | 10331.68 | 10142.06 | 189.61 | 3274.16 | 292.43 | 3147.43 | 3070.52 | 2854.76 |
| 2 | 1 | 10333.88 | 10135.87 | 198.01 | 3281.13 | 325.90 | 3146.92 | 3068.75 | 2827.97 |
| 7 | 1 | 10334.59 | 10145.06 | 189.53 | 3211.64 | 284.13 | 3084.41 | 3007.94 | 2802.74 |
| 4 | 1 | 10347.95 | 10121.05 | 226.90 | 3264.29 | 309.50 | 3096.62 | 3023.31 | 2811.56 |
| 1 | 1 | 10330.61 | 10136.26 | 194.35 | 3213.04 | 308.14 | 3082.65 | 3004.24 | 2777.70 |
| 3 | 1 | 10332.17 | 10131.26 | 200.90 | 3300.86 | 332.96 | 3164.00 | 3085.44 | 2838.62 |
| 6 | 1 | 10328.45 | 10132.16 | 196.29 | 3026.57 | 332.44 | 2895.32 | 2816.48 | 2567.40 |

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
| 0 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 5 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 2 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 7 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 4 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 1 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 3 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 6 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |

## Forward expert-wrapper CUDA API costs

Original + recomputed forwards, averaged per FB. API residence can overlap GPU work; it is not additive or automatically recoverable time.

| Rank | API | Calls/FB | Host API ms/FB |
|---|---|---:|---:|
| 0 | cuLaunchKernelEx | 9504 | 45.14 |
| 0 | cudaEventRecord_v3020 | 1500 | 5.81 |
| 0 | cudaStreamWaitEvent_v3020 | 2400 | 2.20 |
| 5 | cuLaunchKernelEx | 9512 | 40.69 |
| 5 | cudaEventRecord_v3020 | 1500 | 4.88 |
| 5 | cudaStreamWaitEvent_v3020 | 2400 | 1.96 |
| 2 | cuLaunchKernelEx | 9496 | 51.59 |
| 2 | cudaEventRecord_v3020 | 1500 | 6.73 |
| 2 | cudaStreamWaitEvent_v3020 | 2400 | 2.05 |
| 7 | cuLaunchKernelEx | 9512 | 39.95 |
| 7 | cudaEventRecord_v3020 | 1500 | 4.97 |
| 7 | cudaStreamWaitEvent_v3020 | 2400 | 2.12 |
| 4 | cuLaunchKernelEx | 9516 | 49.54 |
| 4 | cudaEventRecord_v3020 | 1500 | 5.97 |
| 4 | cuKernelGetAttribute | 19032 | 2.37 |
| 1 | cuLaunchKernelEx | 9520 | 51.26 |
| 1 | cudaEventRecord_v3020 | 1500 | 6.50 |
| 1 | cuKernelGetAttribute | 19040 | 2.06 |
| 3 | cuLaunchKernelEx | 9492 | 51.60 |
| 3 | cudaEventRecord_v3020 | 1500 | 6.63 |
| 3 | cuKernelGetAttribute | 18984 | 2.10 |
| 6 | cuLaunchKernelEx | 9496 | 40.64 |
| 6 | cudaEventRecord_v3020 | 1500 | 5.00 |
| 6 | cuKernelGetAttribute | 18992 | 2.12 |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3948.56 | 4014.08 |
| expert_dispatch_combine | 2775.76 | 2928.61 |
| expert_gemm_path | 1244.95 | 1244.95 |
| other_compute | 702.38 | 714.65 |
| other_gemm | 557.16 | 562.99 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4130.74 | 4191.37 |
| expert_dispatch_combine | 2677.78 | 2830.09 |
| expert_gemm_path | 1225.90 | 1225.90 |
| other_compute | 734.44 | 747.85 |
| other_gemm | 554.83 | 561.58 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4096.95 | 4161.59 |
| expert_dispatch_combine | 2684.72 | 2834.99 |
| expert_gemm_path | 1274.82 | 1274.82 |
| other_compute | 704.97 | 716.95 |
| other_gemm | 560.55 | 565.98 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4147.55 | 4208.68 |
| expert_dispatch_combine | 2642.30 | 2794.76 |
| expert_gemm_path | 1255.32 | 1255.32 |
| other_compute | 734.86 | 748.50 |
| other_gemm | 573.45 | 579.00 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4137.35 | 4199.04 |
| expert_dispatch_combine | 2649.41 | 2801.06 |
| expert_gemm_path | 1242.19 | 1242.19 |
| other_compute | 740.73 | 750.50 |
| other_gemm | 553.59 | 559.29 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4125.79 | 4187.90 |
| expert_dispatch_combine | 2638.45 | 2790.30 |
| expert_gemm_path | 1299.16 | 1299.16 |
| other_compute | 742.92 | 756.31 |
| other_gemm | 567.03 | 573.61 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4092.76 | 4155.91 |
| expert_dispatch_combine | 2692.52 | 2842.70 |
| expert_gemm_path | 1288.71 | 1288.71 |
| other_compute | 706.21 | 719.46 |
| other_gemm | 551.94 | 557.64 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4113.91 | 4176.97 |
| expert_dispatch_combine | 2449.23 | 2600.94 |
| expert_gemm_path | 1441.49 | 1441.49 |
| other_compute | 763.30 | 776.86 |
| other_gemm | 554.39 | 560.48 |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
