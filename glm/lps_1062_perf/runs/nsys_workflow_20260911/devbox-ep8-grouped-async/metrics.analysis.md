# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/nsys_workflow_20260911/devbox-ep8-grouped-async/metrics.sqlite`
- Ranks: 0, 3, 5, 2, 4, 6, 1, 7; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: 0.6443299616012643%.
- Hardware metrics present: True; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 11088.51 | 10831.92 | 256.59 | 3991.96 | 324.45 | 3798.19 | 3720.32 | 2993.93 |
| 3 | 1 | 11069.96 | 10837.05 | 232.91 | 3783.37 | 305.45 | 3616.88 | 3535.45 | 2817.30 |
| 5 | 1 | 11068.67 | 10826.17 | 242.50 | 3803.26 | 267.32 | 3620.21 | 3546.18 | 2875.34 |
| 2 | 1 | 11071.87 | 10831.54 | 240.33 | 3810.15 | 298.54 | 3633.18 | 3554.96 | 2843.51 |
| 4 | 1 | 11067.84 | 10824.73 | 243.11 | 3774.79 | 265.98 | 3592.43 | 3517.20 | 2843.62 |
| 6 | 1 | 11065.70 | 10836.12 | 229.59 | 3441.90 | 279.12 | 3278.17 | 3197.75 | 2492.01 |
| 1 | 1 | 11072.47 | 10829.02 | 243.45 | 3678.15 | 251.48 | 3496.93 | 3419.79 | 2758.32 |
| 7 | 1 | 11062.30 | 10825.17 | 237.13 | 3731.57 | 268.85 | 3556.10 | 3479.78 | 2815.74 |

### Capture diagnostics

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
| 0 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 3 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 5 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 2 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 4 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 6 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 1 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 7 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |

## Forward expert-wrapper CUDA API costs

Original + recomputed forwards, averaged per FB. API residence can overlap GPU work; it is not additive or automatically recoverable time.

| Rank | API | Calls/FB | Host API ms/FB |
|---|---|---:|---:|
| 0 | cudaLaunchKernel_v7000 | 900 | 5.34 |
| 0 | cudaMemcpyAsync_v3020 | 300 | 3.82 |
| 0 | cudaLaunchKernelExC_v11060 | 300 | 2.41 |
| 3 | cudaLaunchKernel_v7000 | 900 | 5.02 |
| 3 | cudaMemcpyAsync_v3020 | 300 | 3.66 |
| 3 | cudaLaunchKernelExC_v11060 | 300 | 2.20 |
| 5 | cudaLaunchKernel_v7000 | 900 | 5.01 |
| 5 | cudaMemcpyAsync_v3020 | 300 | 3.61 |
| 5 | cudaLaunchKernelExC_v11060 | 300 | 2.20 |
| 2 | cudaLaunchKernel_v7000 | 900 | 5.30 |
| 2 | cudaMemcpyAsync_v3020 | 300 | 3.68 |
| 2 | cudaLaunchKernelExC_v11060 | 300 | 2.51 |
| 4 | cudaLaunchKernel_v7000 | 900 | 4.62 |
| 4 | cudaMemcpyAsync_v3020 | 300 | 3.57 |
| 4 | cudaLaunchKernelExC_v11060 | 300 | 2.13 |
| 6 | cudaLaunchKernel_v7000 | 900 | 4.38 |
| 6 | cudaMemcpyAsync_v3020 | 300 | 3.49 |
| 6 | cudaLaunchKernelExC_v11060 | 300 | 2.05 |
| 1 | cudaLaunchKernel_v7000 | 900 | 4.60 |
| 1 | cudaMemcpyAsync_v3020 | 300 | 3.39 |
| 1 | cudaLaunchKernelExC_v11060 | 300 | 2.04 |
| 7 | cudaLaunchKernel_v7000 | 900 | 4.57 |
| 7 | cudaMemcpyAsync_v3020 | 300 | 3.54 |
| 7 | cudaLaunchKernelExC_v11060 | 300 | 2.10 |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4000.19 | 4066.76 |
| expert_dispatch_combine | 3084.67 | 3238.05 |
| expert_gemm_path | 1428.94 | 1428.94 |
| other_compute | 710.78 | 720.51 |
| other_gemm | 563.60 | 568.95 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4133.31 | 4200.18 |
| expert_dispatch_combine | 3007.41 | 3158.48 |
| expert_gemm_path | 1481.92 | 1481.92 |
| other_compute | 716.56 | 729.10 |
| other_gemm | 556.36 | 562.24 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4175.12 | 4235.87 |
| expert_dispatch_combine | 3032.30 | 3185.43 |
| expert_gemm_path | 1371.59 | 1371.59 |
| other_compute | 742.35 | 753.86 |
| other_gemm | 560.68 | 566.41 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4130.09 | 4194.52 |
| expert_dispatch_combine | 3031.67 | 3182.54 |
| expert_gemm_path | 1437.38 | 1437.38 |
| other_compute | 711.66 | 723.91 |
| other_gemm | 564.01 | 569.55 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4165.29 | 4226.01 |
| expert_dispatch_combine | 2985.51 | 3138.06 |
| expert_gemm_path | 1409.87 | 1409.87 |
| other_compute | 745.90 | 758.67 |
| other_gemm | 558.48 | 564.19 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4195.25 | 4259.87 |
| expert_dispatch_combine | 2718.20 | 2870.66 |
| expert_gemm_path | 1656.35 | 1656.35 |
| other_compute | 775.09 | 788.95 |
| other_gemm | 560.34 | 566.34 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4189.35 | 4254.34 |
| expert_dispatch_combine | 2949.38 | 3101.71 |
| expert_gemm_path | 1478.98 | 1478.98 |
| other_compute | 757.82 | 768.39 |
| other_gemm | 574.47 | 579.91 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4183.73 | 4247.16 |
| expert_dispatch_combine | 2985.72 | 3139.11 |
| expert_gemm_path | 1410.91 | 1410.91 |
| other_compute | 742.24 | 753.99 |
| other_gemm | 578.88 | 584.05 |

## Expert-GEMM hardware samples

Category-exclusive samples only. Excludes launch preparation and gaps. Device-wide counters are not per-kernel rooflines.

| Rank | Metric | Exclusive mean | Sample count | Exclusive fraction |
|---|---|---:|---:|---:|
| 0 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.42e+09 (not trustworthy as MHz) | 14292 | 100.0% |
| 0 | SMs Active [Throughput %] | 97.16 | 14292 | 100.0% |
| 0 | SM Issue [Throughput %] | 5.96 | 14292 | 100.0% |
| 0 | Tensor Active [Throughput %] | 91.75 | 14292 | 100.0% |
| 0 | DRAM Read Bandwidth [Throughput %] | 53.60 | 14292 | 100.0% |
| 0 | DRAM Write Bandwidth [Throughput %] | 5.14 | 14292 | 100.0% |
| 0 | NVLink RX Requests Protocol Data [Throughput %] | 0.46 | 14292 | 100.0% |
| 0 | NVLink RX Requests User Data [Throughput %] | 0.01 | 14292 | 100.0% |
| 0 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 14292 | 100.0% |
| 0 | NVLink RX Responses User Data [Throughput %] | 0.00 | 14292 | 100.0% |
| 0 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 14292 | 100.0% |
| 0 | NVLink TX Requests User Data [Throughput %] | 0.00 | 14292 | 100.0% |
| 0 | NVLink TX Responses Protocol Data [Throughput %] | 0.46 | 14292 | 100.0% |
| 0 | NVLink TX Responses User Data [Throughput %] | 0.46 | 14292 | 100.0% |
| 3 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.44e+09 (not trustworthy as MHz) | 14822 | 100.0% |
| 3 | SMs Active [Throughput %] | 97.73 | 14822 | 100.0% |
| 3 | SM Issue [Throughput %] | 5.92 | 14822 | 100.0% |
| 3 | Tensor Active [Throughput %] | 90.85 | 14822 | 100.0% |
| 3 | DRAM Read Bandwidth [Throughput %] | 55.10 | 14822 | 100.0% |
| 3 | DRAM Write Bandwidth [Throughput %] | 5.19 | 14822 | 100.0% |
| 3 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 14822 | 100.0% |
| 3 | NVLink RX Requests User Data [Throughput %] | 0.00 | 14822 | 100.0% |
| 3 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 14822 | 100.0% |
| 3 | NVLink RX Responses User Data [Throughput %] | 0.00 | 14822 | 100.0% |
| 3 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 14822 | 100.0% |
| 3 | NVLink TX Requests User Data [Throughput %] | 0.00 | 14822 | 100.0% |
| 3 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 14822 | 100.0% |
| 3 | NVLink TX Responses User Data [Throughput %] | 0.00 | 14822 | 100.0% |
| 5 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.42e+09 (not trustworthy as MHz) | 13713 | 100.0% |
| 5 | SMs Active [Throughput %] | 98.75 | 13713 | 100.0% |
| 5 | SM Issue [Throughput %] | 5.94 | 13713 | 100.0% |
| 5 | Tensor Active [Throughput %] | 92.53 | 13713 | 100.0% |
| 5 | DRAM Read Bandwidth [Throughput %] | 51.09 | 13713 | 100.0% |
| 5 | DRAM Write Bandwidth [Throughput %] | 5.23 | 13713 | 100.0% |
| 5 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 13713 | 100.0% |
| 5 | NVLink RX Requests User Data [Throughput %] | 0.00 | 13713 | 100.0% |
| 5 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 13713 | 100.0% |
| 5 | NVLink RX Responses User Data [Throughput %] | 0.00 | 13713 | 100.0% |
| 5 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 13713 | 100.0% |
| 5 | NVLink TX Requests User Data [Throughput %] | 0.00 | 13713 | 100.0% |
| 5 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 13713 | 100.0% |
| 5 | NVLink TX Responses User Data [Throughput %] | 0.00 | 13713 | 100.0% |
| 2 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.39e+09 (not trustworthy as MHz) | 14362 | 100.0% |
| 2 | SMs Active [Throughput %] | 98.39 | 14362 | 100.0% |
| 2 | SM Issue [Throughput %] | 5.92 | 14362 | 100.0% |
| 2 | Tensor Active [Throughput %] | 92.98 | 14362 | 100.0% |
| 2 | DRAM Read Bandwidth [Throughput %] | 52.57 | 14362 | 100.0% |
| 2 | DRAM Write Bandwidth [Throughput %] | 5.12 | 14362 | 100.0% |
| 2 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 14362 | 100.0% |
| 2 | NVLink RX Requests User Data [Throughput %] | 0.00 | 14362 | 100.0% |
| 2 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 14362 | 100.0% |
| 2 | NVLink RX Responses User Data [Throughput %] | 0.00 | 14362 | 100.0% |
| 2 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 14362 | 100.0% |
| 2 | NVLink TX Requests User Data [Throughput %] | 0.00 | 14362 | 100.0% |
| 2 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 14362 | 100.0% |
| 2 | NVLink TX Responses User Data [Throughput %] | 0.00 | 14362 | 100.0% |
| 4 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.43e+09 (not trustworthy as MHz) | 14097 | 100.0% |
| 4 | SMs Active [Throughput %] | 98.04 | 14097 | 100.0% |
| 4 | SM Issue [Throughput %] | 5.90 | 14097 | 100.0% |
| 4 | Tensor Active [Throughput %] | 91.70 | 14097 | 100.0% |
| 4 | DRAM Read Bandwidth [Throughput %] | 53.26 | 14097 | 100.0% |
| 4 | DRAM Write Bandwidth [Throughput %] | 5.17 | 14097 | 100.0% |
| 4 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 14097 | 100.0% |
| 4 | NVLink RX Requests User Data [Throughput %] | 0.00 | 14097 | 100.0% |
| 4 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 14097 | 100.0% |
| 4 | NVLink RX Responses User Data [Throughput %] | 0.00 | 14097 | 100.0% |
| 4 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 14097 | 100.0% |
| 4 | NVLink TX Requests User Data [Throughput %] | 0.00 | 14097 | 100.0% |
| 4 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 14097 | 100.0% |
| 4 | NVLink TX Responses User Data [Throughput %] | 0.00 | 14097 | 100.0% |
| 6 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.42e+09 (not trustworthy as MHz) | 16572 | 100.0% |
| 6 | SMs Active [Throughput %] | 98.65 | 16572 | 100.0% |
| 6 | SM Issue [Throughput %] | 5.79 | 16572 | 100.0% |
| 6 | Tensor Active [Throughput %] | 91.38 | 16572 | 100.0% |
| 6 | DRAM Read Bandwidth [Throughput %] | 56.22 | 16572 | 100.0% |
| 6 | DRAM Write Bandwidth [Throughput %] | 5.15 | 16572 | 100.0% |
| 6 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 16572 | 100.0% |
| 6 | NVLink RX Requests User Data [Throughput %] | 0.00 | 16572 | 100.0% |
| 6 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 16572 | 100.0% |
| 6 | NVLink RX Responses User Data [Throughput %] | 0.00 | 16572 | 100.0% |
| 6 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 16572 | 100.0% |
| 6 | NVLink TX Requests User Data [Throughput %] | 0.00 | 16572 | 100.0% |
| 6 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 16572 | 100.0% |
| 6 | NVLink TX Responses User Data [Throughput %] | 0.00 | 16572 | 100.0% |
| 1 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.37e+09 (not trustworthy as MHz) | 14805 | 100.0% |
| 1 | SMs Active [Throughput %] | 97.93 | 14805 | 100.0% |
| 1 | SM Issue [Throughput %] | 5.99 | 14805 | 100.0% |
| 1 | Tensor Active [Throughput %] | 93.14 | 14805 | 100.0% |
| 1 | DRAM Read Bandwidth [Throughput %] | 53.30 | 14805 | 100.0% |
| 1 | DRAM Write Bandwidth [Throughput %] | 5.07 | 14805 | 100.0% |
| 1 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 14805 | 100.0% |
| 1 | NVLink RX Requests User Data [Throughput %] | 0.00 | 14805 | 100.0% |
| 1 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 14805 | 100.0% |
| 1 | NVLink RX Responses User Data [Throughput %] | 0.00 | 14805 | 100.0% |
| 1 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 14805 | 100.0% |
| 1 | NVLink TX Requests User Data [Throughput %] | 0.00 | 14805 | 100.0% |
| 1 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 14805 | 100.0% |
| 1 | NVLink TX Responses User Data [Throughput %] | 0.00 | 14805 | 100.0% |
| 7 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.33e+09 (not trustworthy as MHz) | 14116 | 100.0% |
| 7 | SMs Active [Throughput %] | 98.42 | 14116 | 100.0% |
| 7 | SM Issue [Throughput %] | 6.06 | 14116 | 100.0% |
| 7 | Tensor Active [Throughput %] | 94.53 | 14116 | 100.0% |
| 7 | DRAM Read Bandwidth [Throughput %] | 50.48 | 14116 | 100.0% |
| 7 | DRAM Write Bandwidth [Throughput %] | 4.97 | 14116 | 100.0% |
| 7 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 14116 | 100.0% |
| 7 | NVLink RX Requests User Data [Throughput %] | 0.00 | 14116 | 100.0% |
| 7 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 14116 | 100.0% |
| 7 | NVLink RX Responses User Data [Throughput %] | 0.00 | 14116 | 100.0% |
| 7 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 14116 | 100.0% |
| 7 | NVLink TX Requests User Data [Throughput %] | 0.00 | 14116 | 100.0% |
| 7 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 14116 | 100.0% |
| 7 | NVLink TX Responses User Data [Throughput %] | 0.00 | 14116 | 100.0% |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
