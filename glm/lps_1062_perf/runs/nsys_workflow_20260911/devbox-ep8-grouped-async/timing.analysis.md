# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/nsys_workflow_20260911/devbox-ep8-grouped-async/timing.sqlite`
- Ranks: 0, 5, 2, 4, 3, 6, 1, 7; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: -0.052654252279327185%.
- Hardware metrics present: False; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 11011.17 | 10750.38 | 260.79 | 3935.42 | 356.99 | 3735.92 | 3659.63 | 2942.15 |
| 5 | 1 | 10988.30 | 10759.04 | 229.26 | 3744.41 | 285.44 | 3575.69 | 3500.60 | 2834.76 |
| 2 | 1 | 10996.19 | 10754.35 | 241.84 | 3761.79 | 318.22 | 3584.46 | 3505.11 | 2809.52 |
| 4 | 1 | 10988.93 | 10758.46 | 230.47 | 3694.65 | 280.00 | 3525.37 | 3449.70 | 2787.01 |
| 3 | 1 | 10996.26 | 10756.81 | 239.45 | 3739.47 | 322.97 | 3565.73 | 3485.03 | 2777.01 |
| 6 | 1 | 10989.48 | 10755.56 | 233.92 | 3424.05 | 305.04 | 3255.82 | 3175.61 | 2476.64 |
| 1 | 1 | 10996.19 | 10745.53 | 250.66 | 3631.44 | 291.76 | 3443.49 | 3365.87 | 2710.12 |
| 7 | 1 | 10994.44 | 10749.13 | 245.31 | 3676.91 | 281.11 | 3489.04 | 3416.97 | 2767.75 |

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
| 0 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 5 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 2 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 4 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 3 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 6 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 1 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 7 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |

## Forward expert-wrapper CUDA API costs

Original + recomputed forwards, averaged per FB. API residence can overlap GPU work; it is not additive or automatically recoverable time.

| Rank | API | Calls/FB | Host API ms/FB |
|---|---|---:|---:|
| 0 | cudaLaunchKernel_v7000 | 900 | 5.35 |
| 0 | cudaMemcpyAsync_v3020 | 300 | 3.90 |
| 0 | cudaLaunchKernelExC_v11060 | 300 | 2.48 |
| 5 | cudaLaunchKernel_v7000 | 900 | 4.82 |
| 5 | cudaMemcpyAsync_v3020 | 300 | 3.54 |
| 5 | cudaLaunchKernelExC_v11060 | 300 | 2.19 |
| 2 | cudaLaunchKernel_v7000 | 900 | 5.15 |
| 2 | cudaMemcpyAsync_v3020 | 300 | 3.71 |
| 2 | cudaLaunchKernelExC_v11060 | 300 | 2.48 |
| 4 | cudaLaunchKernel_v7000 | 900 | 4.45 |
| 4 | cudaMemcpyAsync_v3020 | 300 | 3.44 |
| 4 | cudaLaunchKernelExC_v11060 | 300 | 2.07 |
| 3 | cudaLaunchKernel_v7000 | 900 | 4.78 |
| 3 | cudaMemcpyAsync_v3020 | 300 | 3.64 |
| 3 | cudaLaunchKernelExC_v11060 | 300 | 2.18 |
| 6 | cudaLaunchKernel_v7000 | 900 | 4.41 |
| 6 | cudaMemcpyAsync_v3020 | 300 | 3.50 |
| 6 | cudaLaunchKernelExC_v11060 | 300 | 2.06 |
| 1 | cudaLaunchKernel_v7000 | 900 | 4.51 |
| 1 | cudaMemcpyAsync_v3020 | 300 | 3.56 |
| 1 | cudaLaunchKernelExC_v11060 | 300 | 2.06 |
| 7 | cudaLaunchKernel_v7000 | 900 | 4.50 |
| 7 | cudaMemcpyAsync_v3020 | 300 | 3.42 |
| 7 | cudaLaunchKernelExC_v11060 | 300 | 2.10 |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3990.14 | 4054.83 |
| expert_dispatch_combine | 3057.30 | 3210.05 |
| expert_gemm_path | 1422.16 | 1422.16 |
| other_compute | 712.94 | 722.71 |
| other_gemm | 560.15 | 565.76 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4168.45 | 4228.07 |
| expert_dispatch_combine | 3002.00 | 3154.79 |
| expert_gemm_path | 1364.58 | 1364.58 |
| other_compute | 740.76 | 753.81 |
| other_gemm | 557.23 | 563.60 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4113.56 | 4177.94 |
| expert_dispatch_combine | 2991.10 | 3141.99 |
| expert_gemm_path | 1433.46 | 1433.46 |
| other_compute | 706.51 | 720.03 |
| other_gemm | 561.74 | 567.16 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4173.53 | 4232.31 |
| expert_dispatch_combine | 2948.14 | 3100.51 |
| expert_gemm_path | 1409.40 | 1409.40 |
| other_compute | 742.22 | 755.75 |
| other_gemm | 554.24 | 561.57 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4121.08 | 4185.52 |
| expert_dispatch_combine | 2977.97 | 3128.58 |
| expert_gemm_path | 1474.03 | 1474.03 |
| other_compute | 712.14 | 725.73 |
| other_gemm | 553.26 | 559.66 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4154.70 | 4219.34 |
| expert_dispatch_combine | 2698.17 | 2850.25 |
| expert_gemm_path | 1647.90 | 1647.90 |
| other_compute | 770.01 | 783.90 |
| other_gemm | 557.77 | 563.44 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4183.13 | 4246.13 |
| expert_dispatch_combine | 2922.03 | 3074.06 |
| expert_gemm_path | 1464.03 | 1464.03 |
| other_compute | 757.24 | 770.32 |
| other_gemm | 572.90 | 578.30 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4181.24 | 4240.98 |
| expert_dispatch_combine | 2949.92 | 3102.65 |
| expert_gemm_path | 1405.24 | 1405.24 |
| other_compute | 743.87 | 755.08 |
| other_gemm | 576.52 | 581.51 |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
