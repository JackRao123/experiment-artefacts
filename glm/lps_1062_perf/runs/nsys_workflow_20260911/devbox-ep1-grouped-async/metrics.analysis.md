# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/nsys_workflow_20260911/devbox-ep1-grouped-async/metrics.sqlite`
- Ranks: 0, 4, 3, 7, 6, 5, 1, 2; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: -0.73533112288372%.
- Hardware metrics present: True; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 11637.21 | 11191.81 | 445.40 | 2565.48 | 108.95 | 5598.45 | 2101.27 | 687.87 |
| 4 | 1 | 11594.31 | 11190.33 | 403.98 | 2321.23 | 97.96 | 5479.13 | 1899.59 | 634.44 |
| 3 | 1 | 11606.96 | 11187.07 | 419.90 | 2567.47 | 110.21 | 5419.50 | 2129.25 | 530.44 |
| 7 | 1 | 11602.39 | 11160.93 | 441.46 | 2017.09 | 114.92 | 4976.09 | 1557.55 | 285.05 |
| 6 | 1 | 11600.60 | 11177.82 | 422.78 | 2426.02 | 105.22 | 5354.54 | 1985.54 | 527.99 |
| 5 | 1 | 11654.60 | 11177.77 | 476.83 | 2431.66 | 105.06 | 5530.99 | 1937.10 | 644.88 |
| 1 | 1 | 11604.31 | 11174.68 | 429.63 | 2267.12 | 101.78 | 5377.33 | 1819.02 | 504.30 |
| 2 | 1 | 11607.71 | 11174.00 | 433.71 | 2451.42 | 102.28 | 5318.84 | 1999.59 | 484.61 |

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
| 4 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 3 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 7 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 6 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 5 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 1 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 2 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |

## Forward expert-wrapper CUDA API costs

Original + recomputed forwards, averaged per FB. API residence can overlap GPU work; it is not additive or automatically recoverable time.

| Rank | API | Calls/FB | Host API ms/FB |
|---|---|---:|---:|
| 0 | cudaLaunchKernel_v7000 | 900 | 7.07 |
| 0 | cudaMemcpyAsync_v3020 | 300 | 4.43 |
| 0 | cudaLaunchKernelExC_v11060 | 300 | 2.51 |
| 4 | cudaLaunchKernel_v7000 | 900 | 5.66 |
| 4 | cudaMemcpyAsync_v3020 | 300 | 3.85 |
| 4 | cudaLaunchKernelExC_v11060 | 300 | 2.19 |
| 3 | cudaLaunchKernel_v7000 | 900 | 6.03 |
| 3 | cudaMemcpyAsync_v3020 | 300 | 3.84 |
| 3 | cudaLaunchKernelExC_v11060 | 300 | 2.20 |
| 7 | cudaLaunchKernel_v7000 | 900 | 7.09 |
| 7 | cudaMemcpyAsync_v3020 | 300 | 4.13 |
| 7 | cudaLaunchKernelExC_v11060 | 300 | 2.54 |
| 6 | cudaLaunchKernel_v7000 | 900 | 5.74 |
| 6 | cudaMemcpyAsync_v3020 | 300 | 3.59 |
| 6 | cudaLaunchKernelExC_v11060 | 300 | 2.12 |
| 5 | cudaLaunchKernel_v7000 | 900 | 5.96 |
| 5 | cudaMemcpyAsync_v3020 | 300 | 3.88 |
| 5 | cudaLaunchKernelExC_v11060 | 300 | 2.21 |
| 1 | cudaLaunchKernel_v7000 | 900 | 6.73 |
| 1 | cudaMemcpyAsync_v3020 | 300 | 4.11 |
| 1 | cudaLaunchKernelExC_v11060 | 300 | 2.55 |
| 2 | cudaLaunchKernel_v7000 | 900 | 6.99 |
| 2 | cudaMemcpyAsync_v3020 | 300 | 4.15 |
| 2 | cudaLaunchKernelExC_v11060 | 300 | 2.57 |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3197.60 | 4730.85 |
| fsdp_gather:expert | 1180.57 | 5047.96 |
| expert_gemm_path | 1050.16 | 2423.18 |
| other_compute | 573.64 | 871.29 |
| other_gemm | 445.75 | 617.20 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3279.25 | 4811.78 |
| expert_gemm_path | 1043.73 | 2401.37 |
| fsdp_gather:expert | 1034.09 | 5018.25 |
| other_compute | 596.31 | 997.26 |
| other_gemm | 442.94 | 613.47 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3313.05 | 4807.10 |
| expert_gemm_path | 1128.64 | 2347.39 |
| fsdp_gather:expert | 1107.68 | 4887.10 |
| other_compute | 580.02 | 875.56 |
| other_gemm | 442.74 | 610.79 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3355.46 | 4899.73 |
| expert_gemm_path | 1205.34 | 2415.09 |
| fsdp_gather:expert | 779.06 | 4510.97 |
| other_compute | 638.89 | 1007.85 |
| other_gemm | 461.52 | 636.16 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3282.13 | 4785.60 |
| expert_gemm_path | 1136.92 | 2354.59 |
| fsdp_gather:expert | 960.20 | 4808.64 |
| other_compute | 630.39 | 994.16 |
| other_gemm | 444.19 | 614.02 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3279.24 | 4827.01 |
| fsdp_gather:expert | 1118.33 | 5107.21 |
| expert_gemm_path | 1032.98 | 2412.59 |
| other_compute | 558.81 | 935.49 |
| gpu_idle | 476.83 | 476.83 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3358.38 | 4910.18 |
| expert_gemm_path | 1102.86 | 2447.70 |
| fsdp_gather:expert | 1088.10 | 4988.95 |
| other_compute | 581.23 | 946.30 |
| other_gemm | 456.47 | 630.74 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3326.36 | 4815.42 |
| expert_gemm_path | 1145.10 | 2378.04 |
| fsdp_gather:expert | 1026.79 | 4818.31 |
| other_compute | 601.28 | 914.62 |
| other_gemm | 451.94 | 622.63 |

## Expert-GEMM hardware samples

Category-exclusive samples only. Excludes launch preparation and gaps. Device-wide counters are not per-kernel rooflines.

| Rank | Metric | Exclusive mean | Sample count | Exclusive fraction |
|---|---|---:|---:|---:|
| 0 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.52e+09 (not trustworthy as MHz) | 10488 | 43.3% |
| 0 | SMs Active [Throughput %] | 87.77 | 10488 | 43.3% |
| 0 | SM Issue [Throughput %] | 5.65 | 10488 | 43.3% |
| 0 | Tensor Active [Throughput %] | 80.22 | 10488 | 43.3% |
| 0 | DRAM Read Bandwidth [Throughput %] | 48.69 | 10488 | 43.3% |
| 0 | DRAM Write Bandwidth [Throughput %] | 4.51 | 10488 | 43.3% |
| 0 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 10488 | 43.3% |
| 0 | NVLink RX Requests User Data [Throughput %] | 0.00 | 10488 | 43.3% |
| 0 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 10488 | 43.3% |
| 0 | NVLink RX Responses User Data [Throughput %] | 0.00 | 10488 | 43.3% |
| 0 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 10488 | 43.3% |
| 0 | NVLink TX Requests User Data [Throughput %] | 0.00 | 10488 | 43.3% |
| 0 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 10488 | 43.3% |
| 0 | NVLink TX Responses User Data [Throughput %] | 0.00 | 10488 | 43.3% |
| 4 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.54e+09 (not trustworthy as MHz) | 10444 | 43.5% |
| 4 | SMs Active [Throughput %] | 87.47 | 10444 | 43.5% |
| 4 | SM Issue [Throughput %] | 5.58 | 10444 | 43.5% |
| 4 | Tensor Active [Throughput %] | 79.57 | 10444 | 43.5% |
| 4 | DRAM Read Bandwidth [Throughput %] | 47.32 | 10444 | 43.5% |
| 4 | DRAM Write Bandwidth [Throughput %] | 4.52 | 10444 | 43.5% |
| 4 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 10444 | 43.5% |
| 4 | NVLink RX Requests User Data [Throughput %] | 0.00 | 10444 | 43.5% |
| 4 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 10444 | 43.5% |
| 4 | NVLink RX Responses User Data [Throughput %] | 0.00 | 10444 | 43.5% |
| 4 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 10444 | 43.5% |
| 4 | NVLink TX Requests User Data [Throughput %] | 0.00 | 10444 | 43.5% |
| 4 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 10444 | 43.5% |
| 4 | NVLink TX Responses User Data [Throughput %] | 0.00 | 10444 | 43.5% |
| 3 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.57e+09 (not trustworthy as MHz) | 11287 | 48.1% |
| 3 | SMs Active [Throughput %] | 85.46 | 11287 | 48.1% |
| 3 | SM Issue [Throughput %] | 5.37 | 11287 | 48.1% |
| 3 | Tensor Active [Throughput %] | 77.27 | 11287 | 48.1% |
| 3 | DRAM Read Bandwidth [Throughput %] | 50.15 | 11287 | 48.1% |
| 3 | DRAM Write Bandwidth [Throughput %] | 4.38 | 11287 | 48.1% |
| 3 | NVLink RX Requests Protocol Data [Throughput %] | 0.01 | 11287 | 48.1% |
| 3 | NVLink RX Requests User Data [Throughput %] | 0.02 | 11287 | 48.1% |
| 3 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11287 | 48.1% |
| 3 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11287 | 48.1% |
| 3 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11287 | 48.1% |
| 3 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11287 | 48.1% |
| 3 | NVLink TX Responses Protocol Data [Throughput %] | 0.01 | 11287 | 48.1% |
| 3 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11287 | 48.1% |
| 7 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.43e+09 (not trustworthy as MHz) | 12049 | 49.9% |
| 7 | SMs Active [Throughput %] | 89.16 | 12049 | 49.9% |
| 7 | SM Issue [Throughput %] | 5.57 | 12049 | 49.9% |
| 7 | Tensor Active [Throughput %] | 83.25 | 12049 | 49.9% |
| 7 | DRAM Read Bandwidth [Throughput %] | 48.69 | 12049 | 49.9% |
| 7 | DRAM Write Bandwidth [Throughput %] | 4.24 | 12049 | 49.9% |
| 7 | NVLink RX Requests Protocol Data [Throughput %] | 0.16 | 12049 | 49.9% |
| 7 | NVLink RX Requests User Data [Throughput %] | 0.79 | 12049 | 49.9% |
| 7 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12049 | 49.9% |
| 7 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12049 | 49.9% |
| 7 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12049 | 49.9% |
| 7 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12049 | 49.9% |
| 7 | NVLink TX Responses Protocol Data [Throughput %] | 0.03 | 12049 | 49.9% |
| 7 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12049 | 49.9% |
| 6 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.56e+09 (not trustworthy as MHz) | 11360 | 48.3% |
| 6 | SMs Active [Throughput %] | 86.08 | 11360 | 48.3% |
| 6 | SM Issue [Throughput %] | 5.33 | 11360 | 48.3% |
| 6 | Tensor Active [Throughput %] | 78.14 | 11360 | 48.3% |
| 6 | DRAM Read Bandwidth [Throughput %] | 49.21 | 11360 | 48.3% |
| 6 | DRAM Write Bandwidth [Throughput %] | 4.57 | 11360 | 48.3% |
| 6 | NVLink RX Requests Protocol Data [Throughput %] | 0.37 | 11360 | 48.3% |
| 6 | NVLink RX Requests User Data [Throughput %] | 1.90 | 11360 | 48.3% |
| 6 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11360 | 48.3% |
| 6 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11360 | 48.3% |
| 6 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11360 | 48.3% |
| 6 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11360 | 48.3% |
| 6 | NVLink TX Responses Protocol Data [Throughput %] | 0.06 | 11360 | 48.3% |
| 6 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11360 | 48.3% |
| 5 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.53e+09 (not trustworthy as MHz) | 10330 | 42.8% |
| 5 | SMs Active [Throughput %] | 88.69 | 10330 | 42.8% |
| 5 | SM Issue [Throughput %] | 5.64 | 10330 | 42.8% |
| 5 | Tensor Active [Throughput %] | 80.74 | 10330 | 42.8% |
| 5 | DRAM Read Bandwidth [Throughput %] | 48.55 | 10330 | 42.8% |
| 5 | DRAM Write Bandwidth [Throughput %] | 4.57 | 10330 | 42.8% |
| 5 | NVLink RX Requests Protocol Data [Throughput %] | 0.01 | 10330 | 42.8% |
| 5 | NVLink RX Requests User Data [Throughput %] | 0.06 | 10330 | 42.8% |
| 5 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 10330 | 42.8% |
| 5 | NVLink RX Responses User Data [Throughput %] | 0.00 | 10330 | 42.8% |
| 5 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 10330 | 42.8% |
| 5 | NVLink TX Requests User Data [Throughput %] | 0.00 | 10330 | 42.8% |
| 5 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 10330 | 42.8% |
| 5 | NVLink TX Responses User Data [Throughput %] | 0.00 | 10330 | 42.8% |
| 1 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.49e+09 (not trustworthy as MHz) | 11033 | 45.1% |
| 1 | SMs Active [Throughput %] | 85.55 | 11033 | 45.1% |
| 1 | SM Issue [Throughput %] | 5.63 | 11033 | 45.1% |
| 1 | Tensor Active [Throughput %] | 78.88 | 11033 | 45.1% |
| 1 | DRAM Read Bandwidth [Throughput %] | 46.76 | 11033 | 45.1% |
| 1 | DRAM Write Bandwidth [Throughput %] | 4.44 | 11033 | 45.1% |
| 1 | NVLink RX Requests Protocol Data [Throughput %] | 0.08 | 11033 | 45.1% |
| 1 | NVLink RX Requests User Data [Throughput %] | 0.39 | 11033 | 45.1% |
| 1 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11033 | 45.1% |
| 1 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11033 | 45.1% |
| 1 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11033 | 45.1% |
| 1 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11033 | 45.1% |
| 1 | NVLink TX Responses Protocol Data [Throughput %] | 0.01 | 11033 | 45.1% |
| 1 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11033 | 45.1% |
| 2 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.53e+09 (not trustworthy as MHz) | 11451 | 48.2% |
| 2 | SMs Active [Throughput %] | 85.97 | 11451 | 48.2% |
| 2 | SM Issue [Throughput %] | 5.39 | 11451 | 48.2% |
| 2 | Tensor Active [Throughput %] | 78.60 | 11451 | 48.2% |
| 2 | DRAM Read Bandwidth [Throughput %] | 49.23 | 11451 | 48.2% |
| 2 | DRAM Write Bandwidth [Throughput %] | 4.48 | 11451 | 48.2% |
| 2 | NVLink RX Requests Protocol Data [Throughput %] | 0.30 | 11451 | 48.2% |
| 2 | NVLink RX Requests User Data [Throughput %] | 1.59 | 11451 | 48.2% |
| 2 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11451 | 48.2% |
| 2 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11451 | 48.2% |
| 2 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11451 | 48.2% |
| 2 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11451 | 48.2% |
| 2 | NVLink TX Responses Protocol Data [Throughput %] | 0.04 | 11451 | 48.2% |
| 2 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11451 | 48.2% |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
