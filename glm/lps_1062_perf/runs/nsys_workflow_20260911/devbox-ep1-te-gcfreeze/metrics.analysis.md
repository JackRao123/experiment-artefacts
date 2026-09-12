# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/nsys_workflow_20260911/devbox-ep1-te-gcfreeze/metrics.sqlite`
- Ranks: 0, 1, 3, 5, 7, 2, 4, 6; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: 17.70402744669257%.
- Hardware metrics present: True; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 13566.57 | 11956.97 | 1609.59 | 5096.72 | 3236.34 | 6548.43 | 3468.64 | 468.34 |
| 1 | 1 | 13549.75 | 12148.76 | 1400.99 | 4821.65 | 3037.52 | 6553.62 | 3402.23 | 453.39 |
| 3 | 1 | 13558.79 | 12268.92 | 1289.86 | 4916.78 | 3048.98 | 6661.58 | 3608.48 | 630.64 |
| 5 | 1 | 13545.96 | 12193.64 | 1352.32 | 4835.91 | 3104.66 | 6625.28 | 3466.01 | 473.82 |
| 7 | 1 | 13535.66 | 10602.60 | 2933.06 | 4782.45 | 3094.32 | 4869.75 | 1831.44 | 572.62 |
| 2 | 1 | 13546.19 | 12291.35 | 1254.84 | 4877.27 | 3004.63 | 6624.26 | 3603.92 | 613.92 |
| 4 | 1 | 13549.63 | 12007.11 | 1542.51 | 4804.15 | 3109.31 | 6423.35 | 3243.92 | 404.00 |
| 6 | 1 | 13539.71 | 12231.73 | 1307.98 | 4841.92 | 3089.08 | 6673.62 | 3516.20 | 530.67 |

### Capture diagnostics

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
| 0 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 1 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 3 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 5 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 7 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 2 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 4 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 6 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |

## Forward expert-wrapper CUDA API costs

Original + recomputed forwards, averaged per FB. API residence can overlap GPU work; it is not additive or automatically recoverable time.

| Rank | API | Calls/FB | Host API ms/FB |
|---|---|---:|---:|
| 0 | cuLaunchKernelEx | 74632 | 331.70 |
| 0 | cudaLaunchKernelExC_v11060 | 4412 | 17.71 |
| 0 | cuKernelGetAttribute | 149264 | 10.19 |
| 1 | cuLaunchKernelEx | 74448 | 253.90 |
| 1 | cudaLaunchKernelExC_v11060 | 4324 | 13.55 |
| 1 | cuKernelGetAttribute | 148896 | 10.23 |
| 3 | cuLaunchKernelEx | 74460 | 277.37 |
| 3 | cudaLaunchKernelExC_v11060 | 4652 | 15.86 |
| 3 | cuKernelGetAttribute | 148920 | 10.45 |
| 5 | cuLaunchKernelEx | 74528 | 254.51 |
| 5 | cudaLaunchKernelExC_v11060 | 4652 | 14.52 |
| 5 | cuKernelGetAttribute | 149056 | 10.57 |
| 7 | cuLaunchKernelEx | 74316 | 252.16 |
| 7 | cudaLaunchKernelExC_v11060 | 4498 | 14.02 |
| 7 | cuKernelGetAttribute | 148632 | 10.51 |
| 2 | cuLaunchKernelEx | 74516 | 274.51 |
| 2 | cudaLaunchKernelExC_v11060 | 4532 | 15.02 |
| 2 | cuKernelGetAttribute | 149032 | 10.26 |
| 4 | cuLaunchKernelEx | 74484 | 363.40 |
| 4 | cudaLaunchKernelExC_v11060 | 4620 | 19.68 |
| 4 | cuKernelGetAttribute | 148968 | 10.19 |
| 6 | cuLaunchKernelEx | 74448 | 252.70 |
| 6 | cudaLaunchKernelExC_v11060 | 4554 | 14.11 |
| 6 | cuKernelGetAttribute | 148896 | 10.45 |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3008.94 | 4423.72 |
| cp_communication | 1968.99 | 2070.22 |
| gpu_idle | 1609.59 | 1609.59 |
| expert_gemm_path | 1138.94 | 2185.44 |
| fsdp_gather:expert | 954.58 | 4276.59 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3164.32 | 4612.93 |
| cp_communication | 1847.16 | 2022.80 |
| gpu_idle | 1400.99 | 1400.99 |
| expert_gemm_path | 1161.37 | 2175.79 |
| fsdp_gather:expert | 897.84 | 4367.52 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3143.58 | 4582.10 |
| cp_communication | 1859.32 | 2051.26 |
| gpu_idle | 1289.86 | 1289.86 |
| expert_gemm_path | 1202.82 | 2117.91 |
| fsdp_gather:expert | 870.99 | 4361.97 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3137.64 | 4572.63 |
| cp_communication | 1858.99 | 2044.13 |
| gpu_idle | 1352.32 | 1352.32 |
| expert_gemm_path | 1143.46 | 2157.46 |
| fsdp_gather:expert | 893.96 | 4383.04 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3163.05 | 4617.15 |
| gpu_idle | 2933.06 | 2933.06 |
| expert_gemm_path | 1250.95 | 2156.60 |
| fsdp_gather:expert | 874.59 | 4346.90 |
| other_compute | 538.52 | 928.76 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3151.93 | 4599.64 |
| cp_communication | 1855.11 | 2008.64 |
| gpu_idle | 1254.84 | 1254.84 |
| expert_gemm_path | 1219.51 | 2135.04 |
| fsdp_gather:expert | 915.70 | 4336.34 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3139.91 | 4580.77 |
| cp_communication | 1852.58 | 1976.31 |
| gpu_idle | 1542.51 | 1542.51 |
| expert_gemm_path | 1158.10 | 2180.43 |
| fsdp_gather:expert | 818.85 | 4263.70 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3113.69 | 4550.39 |
| cp_communication | 1869.02 | 2101.78 |
| gpu_idle | 1307.98 | 1307.98 |
| expert_gemm_path | 1140.69 | 2152.33 |
| fsdp_gather:expert | 873.29 | 4412.99 |

## Expert-GEMM hardware samples

Category-exclusive samples only. Excludes launch preparation and gaps. Device-wide counters are not per-kernel rooflines.

| Rank | Metric | Exclusive mean | Sample count | Exclusive fraction |
|---|---|---:|---:|---:|
| 0 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.77e+09 (not trustworthy as MHz) | 11338 | 52.0% |
| 0 | SMs Active [Throughput %] | 83.81 | 11338 | 52.0% |
| 0 | SM Issue [Throughput %] | 4.81 | 11338 | 52.0% |
| 0 | Tensor Active [Throughput %] | 54.19 | 11338 | 52.0% |
| 0 | DRAM Read Bandwidth [Throughput %] | 33.42 | 11338 | 52.0% |
| 0 | DRAM Write Bandwidth [Throughput %] | 4.02 | 11338 | 52.0% |
| 0 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11338 | 52.0% |
| 0 | NVLink RX Requests User Data [Throughput %] | 0.01 | 11338 | 52.0% |
| 0 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11338 | 52.0% |
| 0 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11338 | 52.0% |
| 0 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11338 | 52.0% |
| 0 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11338 | 52.0% |
| 0 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11338 | 52.0% |
| 0 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11338 | 52.0% |
| 1 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.74e+09 (not trustworthy as MHz) | 11614 | 53.3% |
| 1 | SMs Active [Throughput %] | 87.72 | 11614 | 53.3% |
| 1 | SM Issue [Throughput %] | 5.12 | 11614 | 53.3% |
| 1 | Tensor Active [Throughput %] | 57.26 | 11614 | 53.3% |
| 1 | DRAM Read Bandwidth [Throughput %] | 34.44 | 11614 | 53.3% |
| 1 | DRAM Write Bandwidth [Throughput %] | 4.20 | 11614 | 53.3% |
| 1 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11614 | 53.3% |
| 1 | NVLink RX Requests User Data [Throughput %] | 0.00 | 11614 | 53.3% |
| 1 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11614 | 53.3% |
| 1 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11614 | 53.3% |
| 1 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11614 | 53.3% |
| 1 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11614 | 53.3% |
| 1 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11614 | 53.3% |
| 1 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11614 | 53.3% |
| 3 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.77e+09 (not trustworthy as MHz) | 12005 | 56.7% |
| 3 | SMs Active [Throughput %] | 89.37 | 12005 | 56.7% |
| 3 | SM Issue [Throughput %] | 5.22 | 12005 | 56.7% |
| 3 | Tensor Active [Throughput %] | 57.89 | 12005 | 56.7% |
| 3 | DRAM Read Bandwidth [Throughput %] | 35.96 | 12005 | 56.7% |
| 3 | DRAM Write Bandwidth [Throughput %] | 4.22 | 12005 | 56.7% |
| 3 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 12005 | 56.7% |
| 3 | NVLink RX Requests User Data [Throughput %] | 0.00 | 12005 | 56.7% |
| 3 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12005 | 56.7% |
| 3 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12005 | 56.7% |
| 3 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12005 | 56.7% |
| 3 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12005 | 56.7% |
| 3 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12005 | 56.7% |
| 3 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12005 | 56.7% |
| 5 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.76e+09 (not trustworthy as MHz) | 11460 | 53.0% |
| 5 | SMs Active [Throughput %] | 87.96 | 11460 | 53.0% |
| 5 | SM Issue [Throughput %] | 5.03 | 11460 | 53.0% |
| 5 | Tensor Active [Throughput %] | 57.01 | 11460 | 53.0% |
| 5 | DRAM Read Bandwidth [Throughput %] | 34.85 | 11460 | 53.0% |
| 5 | DRAM Write Bandwidth [Throughput %] | 4.20 | 11460 | 53.0% |
| 5 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11460 | 53.0% |
| 5 | NVLink RX Requests User Data [Throughput %] | 0.00 | 11460 | 53.0% |
| 5 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11460 | 53.0% |
| 5 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11460 | 53.0% |
| 5 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11460 | 53.0% |
| 5 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11460 | 53.0% |
| 5 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11460 | 53.0% |
| 5 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11460 | 53.0% |
| 7 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.71e+09 (not trustworthy as MHz) | 12519 | 58.0% |
| 7 | SMs Active [Throughput %] | 90.24 | 12519 | 58.0% |
| 7 | SM Issue [Throughput %] | 5.29 | 12519 | 58.0% |
| 7 | Tensor Active [Throughput %] | 59.43 | 12519 | 58.0% |
| 7 | DRAM Read Bandwidth [Throughput %] | 35.17 | 12519 | 58.0% |
| 7 | DRAM Write Bandwidth [Throughput %] | 4.12 | 12519 | 58.0% |
| 7 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 12519 | 58.0% |
| 7 | NVLink RX Requests User Data [Throughput %] | 0.02 | 12519 | 58.0% |
| 7 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12519 | 58.0% |
| 7 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12519 | 58.0% |
| 7 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12519 | 58.0% |
| 7 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12519 | 58.0% |
| 7 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12519 | 58.0% |
| 7 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12519 | 58.0% |
| 2 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.75e+09 (not trustworthy as MHz) | 12174 | 57.0% |
| 2 | SMs Active [Throughput %] | 89.65 | 12174 | 57.0% |
| 2 | SM Issue [Throughput %] | 5.23 | 12174 | 57.0% |
| 2 | Tensor Active [Throughput %] | 58.33 | 12174 | 57.0% |
| 2 | DRAM Read Bandwidth [Throughput %] | 35.61 | 12174 | 57.0% |
| 2 | DRAM Write Bandwidth [Throughput %] | 4.15 | 12174 | 57.0% |
| 2 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 12174 | 57.0% |
| 2 | NVLink RX Requests User Data [Throughput %] | 0.01 | 12174 | 57.0% |
| 2 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12174 | 57.0% |
| 2 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12174 | 57.0% |
| 2 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12174 | 57.0% |
| 2 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12174 | 57.0% |
| 2 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12174 | 57.0% |
| 2 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12174 | 57.0% |
| 4 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.77e+09 (not trustworthy as MHz) | 11630 | 53.3% |
| 4 | SMs Active [Throughput %] | 83.86 | 11630 | 53.3% |
| 4 | SM Issue [Throughput %] | 4.79 | 11630 | 53.3% |
| 4 | Tensor Active [Throughput %] | 54.37 | 11630 | 53.3% |
| 4 | DRAM Read Bandwidth [Throughput %] | 33.33 | 11630 | 53.3% |
| 4 | DRAM Write Bandwidth [Throughput %] | 3.99 | 11630 | 53.3% |
| 4 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11630 | 53.3% |
| 4 | NVLink RX Requests User Data [Throughput %] | 0.01 | 11630 | 53.3% |
| 4 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11630 | 53.3% |
| 4 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11630 | 53.3% |
| 4 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11630 | 53.3% |
| 4 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11630 | 53.3% |
| 4 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11630 | 53.3% |
| 4 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11630 | 53.3% |
| 6 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.77e+09 (not trustworthy as MHz) | 11416 | 53.0% |
| 6 | SMs Active [Throughput %] | 87.71 | 11416 | 53.0% |
| 6 | SM Issue [Throughput %] | 5.01 | 11416 | 53.0% |
| 6 | Tensor Active [Throughput %] | 56.87 | 11416 | 53.0% |
| 6 | DRAM Read Bandwidth [Throughput %] | 34.81 | 11416 | 53.0% |
| 6 | DRAM Write Bandwidth [Throughput %] | 4.20 | 11416 | 53.0% |
| 6 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11416 | 53.0% |
| 6 | NVLink RX Requests User Data [Throughput %] | 0.01 | 11416 | 53.0% |
| 6 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11416 | 53.0% |
| 6 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11416 | 53.0% |
| 6 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11416 | 53.0% |
| 6 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11416 | 53.0% |
| 6 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11416 | 53.0% |
| 6 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11416 | 53.0% |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
