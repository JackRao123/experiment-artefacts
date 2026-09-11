# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `devbox-ep1-grouped/metrics.sqlite`
- Ranks: 5, 7, 6, 4, 0, 3, 2, 1; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: 14.47087351192582%.
- Hardware metrics present: True; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1 | 13318.56 | 12427.30 | 891.26 | 4330.64 | 2664.57 | 6763.49 | 3421.18 | 447.76 |
| 7 | 1 | 13312.35 | 10745.34 | 2567.01 | 4080.40 | 2608.41 | 4952.76 | 1494.73 | 351.75 |
| 6 | 1 | 13316.08 | 12447.81 | 868.27 | 4376.41 | 2804.78 | 6711.02 | 3489.93 | 395.58 |
| 4 | 1 | 13301.58 | 12464.31 | 837.27 | 4255.79 | 2516.02 | 6862.25 | 3400.30 | 523.09 |
| 0 | 1 | 13345.99 | 12329.34 | 1016.65 | 4615.51 | 2911.72 | 6759.37 | 3580.08 | 467.87 |
| 3 | 1 | 13312.51 | 12434.60 | 877.91 | 4403.97 | 2696.62 | 6788.70 | 3507.17 | 440.07 |
| 2 | 1 | 13311.56 | 12357.45 | 954.11 | 4390.50 | 2854.18 | 6609.76 | 3417.56 | 359.75 |
| 1 | 1 | 13318.11 | 12361.97 | 956.14 | 4246.31 | 2827.30 | 6581.58 | 3271.48 | 304.88 |

### Capture diagnostics

- Profile median is outside the observed control range: instrumentation, remaining warmup, or workload drift may affect attribution.
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
| 5 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 7 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 6 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 4 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 0 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 3 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 2 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 1 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3209.68 | 4664.00 |
| cp_communication | 1864.56 | 2042.49 |
| expert_gemm_path | 1152.89 | 2349.72 |
| gpu_idle | 891.26 | 891.26 |
| fsdp_gather:expert | 856.67 | 4522.05 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3283.13 | 4761.32 |
| gpu_idle | 2567.01 | 2567.01 |
| expert_gemm_path | 1174.81 | 2430.22 |
| fsdp_gather:expert | 802.18 | 4524.70 |
| other_compute | 547.25 | 975.20 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3190.12 | 4648.81 |
| cp_communication | 1877.22 | 1997.21 |
| expert_gemm_path | 1180.15 | 2265.38 |
| gpu_idle | 868.27 | 868.27 |
| fsdp_gather:expert | 831.58 | 4418.70 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3199.11 | 4641.23 |
| cp_communication | 1881.05 | 2123.95 |
| expert_gemm_path | 1107.09 | 2391.37 |
| fsdp_gather:expert | 854.21 | 4617.92 |
| gpu_idle | 837.27 | 837.27 |

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3099.02 | 4527.80 |
| cp_communication | 1964.08 | 2073.84 |
| expert_gemm_path | 1189.48 | 2312.25 |
| gpu_idle | 1016.65 | 1016.65 |
| fsdp_gather:expert | 857.56 | 4395.32 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3234.50 | 4682.36 |
| cp_communication | 1850.63 | 2044.66 |
| expert_gemm_path | 1137.74 | 2324.95 |
| fsdp_gather:expert | 935.13 | 4553.74 |
| gpu_idle | 877.91 | 877.91 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3242.74 | 4701.08 |
| cp_communication | 1840.47 | 1964.67 |
| expert_gemm_path | 1193.87 | 2282.92 |
| gpu_idle | 954.11 | 954.11 |
| fsdp_gather:expert | 814.44 | 4377.82 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3267.63 | 4728.96 |
| cp_communication | 1818.38 | 1910.36 |
| expert_gemm_path | 1206.14 | 2300.37 |
| gpu_idle | 956.14 | 956.14 |
| fsdp_gather:expert | 734.64 | 4383.08 |

## Expert-GEMM hardware samples

Category-exclusive samples only. Excludes launch preparation and gaps. Device-wide counters are not per-kernel rooflines.

| Rank | Metric | Exclusive mean | Sample count | Exclusive fraction |
|---|---|---:|---:|---:|
| 5 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.54e+09 (not trustworthy as MHz) | 11522 | 49.0% |
| 5 | SMs Active [Throughput %] | 88.41 | 11522 | 49.0% |
| 5 | SM Issue [Throughput %] | 5.39 | 11522 | 49.0% |
| 5 | Tensor Active [Throughput %] | 79.99 | 11522 | 49.0% |
| 5 | DRAM Read Bandwidth [Throughput %] | 51.90 | 11522 | 49.0% |
| 5 | DRAM Write Bandwidth [Throughput %] | 4.35 | 11522 | 49.0% |
| 5 | NVLink RX Requests Protocol Data [Throughput %] | 0.03 | 11522 | 49.0% |
| 5 | NVLink RX Requests User Data [Throughput %] | 0.16 | 11522 | 49.0% |
| 5 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11522 | 49.0% |
| 5 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11522 | 49.0% |
| 5 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11522 | 49.0% |
| 5 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11522 | 49.0% |
| 5 | NVLink TX Responses Protocol Data [Throughput %] | 0.01 | 11522 | 49.0% |
| 5 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11522 | 49.0% |
| 7 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.46e+09 (not trustworthy as MHz) | 11754 | 48.4% |
| 7 | SMs Active [Throughput %] | 86.97 | 11754 | 48.4% |
| 7 | SM Issue [Throughput %] | 5.48 | 11754 | 48.4% |
| 7 | Tensor Active [Throughput %] | 81.20 | 11754 | 48.4% |
| 7 | DRAM Read Bandwidth [Throughput %] | 47.69 | 11754 | 48.4% |
| 7 | DRAM Write Bandwidth [Throughput %] | 4.21 | 11754 | 48.4% |
| 7 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11754 | 48.4% |
| 7 | NVLink RX Requests User Data [Throughput %] | 0.00 | 11754 | 48.4% |
| 7 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11754 | 48.4% |
| 7 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11754 | 48.4% |
| 7 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11754 | 48.4% |
| 7 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11754 | 48.4% |
| 7 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11754 | 48.4% |
| 7 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11754 | 48.4% |
| 6 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.52e+09 (not trustworthy as MHz) | 11803 | 52.1% |
| 6 | SMs Active [Throughput %] | 93.74 | 11803 | 52.1% |
| 6 | SM Issue [Throughput %] | 5.57 | 11803 | 52.1% |
| 6 | Tensor Active [Throughput %] | 84.77 | 11803 | 52.1% |
| 6 | DRAM Read Bandwidth [Throughput %] | 53.70 | 11803 | 52.1% |
| 6 | DRAM Write Bandwidth [Throughput %] | 4.40 | 11803 | 52.1% |
| 6 | NVLink RX Requests Protocol Data [Throughput %] | 0.03 | 11803 | 52.1% |
| 6 | NVLink RX Requests User Data [Throughput %] | 0.14 | 11803 | 52.1% |
| 6 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11803 | 52.1% |
| 6 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11803 | 52.1% |
| 6 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11803 | 52.1% |
| 6 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11803 | 52.1% |
| 6 | NVLink TX Responses Protocol Data [Throughput %] | 0.01 | 11803 | 52.1% |
| 6 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11803 | 52.1% |
| 4 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.57e+09 (not trustworthy as MHz) | 11077 | 46.3% |
| 4 | SMs Active [Throughput %] | 84.91 | 11077 | 46.3% |
| 4 | SM Issue [Throughput %] | 5.32 | 11077 | 46.3% |
| 4 | Tensor Active [Throughput %] | 77.18 | 11077 | 46.3% |
| 4 | DRAM Read Bandwidth [Throughput %] | 47.79 | 11077 | 46.3% |
| 4 | DRAM Write Bandwidth [Throughput %] | 4.37 | 11077 | 46.3% |
| 4 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11077 | 46.3% |
| 4 | NVLink RX Requests User Data [Throughput %] | 0.01 | 11077 | 46.3% |
| 4 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11077 | 46.3% |
| 4 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11077 | 46.3% |
| 4 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11077 | 46.3% |
| 4 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11077 | 46.3% |
| 4 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11077 | 46.3% |
| 4 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11077 | 46.3% |
| 0 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.53e+09 (not trustworthy as MHz) | 11892 | 51.4% |
| 0 | SMs Active [Throughput %] | 91.97 | 11892 | 51.4% |
| 0 | SM Issue [Throughput %] | 5.50 | 11892 | 51.4% |
| 0 | Tensor Active [Throughput %] | 82.70 | 11892 | 51.4% |
| 0 | DRAM Read Bandwidth [Throughput %] | 55.35 | 11892 | 51.4% |
| 0 | DRAM Write Bandwidth [Throughput %] | 4.38 | 11892 | 51.4% |
| 0 | NVLink RX Requests Protocol Data [Throughput %] | 0.02 | 11892 | 51.4% |
| 0 | NVLink RX Requests User Data [Throughput %] | 0.12 | 11892 | 51.4% |
| 0 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11892 | 51.4% |
| 0 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11892 | 51.4% |
| 0 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11892 | 51.4% |
| 0 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11892 | 51.4% |
| 0 | NVLink TX Responses Protocol Data [Throughput %] | 0.01 | 11892 | 51.4% |
| 0 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11892 | 51.4% |
| 3 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.56e+09 (not trustworthy as MHz) | 11374 | 48.9% |
| 3 | SMs Active [Throughput %] | 88.25 | 11374 | 48.9% |
| 3 | SM Issue [Throughput %] | 5.49 | 11374 | 48.9% |
| 3 | Tensor Active [Throughput %] | 79.78 | 11374 | 48.9% |
| 3 | DRAM Read Bandwidth [Throughput %] | 50.93 | 11374 | 48.9% |
| 3 | DRAM Write Bandwidth [Throughput %] | 4.43 | 11374 | 48.9% |
| 3 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11374 | 48.9% |
| 3 | NVLink RX Requests User Data [Throughput %] | 0.00 | 11374 | 48.9% |
| 3 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11374 | 48.9% |
| 3 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11374 | 48.9% |
| 3 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11374 | 48.9% |
| 3 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11374 | 48.9% |
| 3 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11374 | 48.9% |
| 3 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11374 | 48.9% |
| 2 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.49e+09 (not trustworthy as MHz) | 11934 | 52.3% |
| 2 | SMs Active [Throughput %] | 94.47 | 11934 | 52.3% |
| 2 | SM Issue [Throughput %] | 5.57 | 11934 | 52.3% |
| 2 | Tensor Active [Throughput %] | 86.23 | 11934 | 52.3% |
| 2 | DRAM Read Bandwidth [Throughput %] | 53.87 | 11934 | 52.3% |
| 2 | DRAM Write Bandwidth [Throughput %] | 4.37 | 11934 | 52.3% |
| 2 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11934 | 52.3% |
| 2 | NVLink RX Requests User Data [Throughput %] | 0.01 | 11934 | 52.3% |
| 2 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11934 | 52.3% |
| 2 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11934 | 52.3% |
| 2 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11934 | 52.3% |
| 2 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11934 | 52.3% |
| 2 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11934 | 52.3% |
| 2 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11934 | 52.3% |
| 1 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.47e+09 (not trustworthy as MHz) | 12064 | 52.4% |
| 1 | SMs Active [Throughput %] | 94.45 | 12064 | 52.4% |
| 1 | SM Issue [Throughput %] | 5.74 | 12064 | 52.4% |
| 1 | Tensor Active [Throughput %] | 86.62 | 12064 | 52.4% |
| 1 | DRAM Read Bandwidth [Throughput %] | 53.31 | 12064 | 52.4% |
| 1 | DRAM Write Bandwidth [Throughput %] | 4.39 | 12064 | 52.4% |
| 1 | NVLink RX Requests Protocol Data [Throughput %] | 0.01 | 12064 | 52.4% |
| 1 | NVLink RX Requests User Data [Throughput %] | 0.07 | 12064 | 52.4% |
| 1 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12064 | 52.4% |
| 1 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12064 | 52.4% |
| 1 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12064 | 52.4% |
| 1 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12064 | 52.4% |
| 1 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12064 | 52.4% |
| 1 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12064 | 52.4% |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
