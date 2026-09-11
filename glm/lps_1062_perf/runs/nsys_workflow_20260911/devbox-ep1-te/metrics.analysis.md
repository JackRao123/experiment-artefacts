# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `devbox-ep1-te/metrics.sqlite`
- Ranks: 0, 1, 4, 7, 6, 5, 2, 3; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: 9.430733993006513%.
- Hardware metrics present: True; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 12815.68 | 11164.59 | 1651.09 | 4290.43 | 2458.97 | 5739.51 | 2620.66 | 474.01 |
| 1 | 1 | 12786.69 | 11412.99 | 1373.70 | 3994.61 | 2211.30 | 5739.79 | 2602.49 | 595.80 |
| 4 | 1 | 12772.17 | 11435.23 | 1336.93 | 4093.34 | 2272.43 | 5917.99 | 2738.97 | 485.02 |
| 7 | 1 | 12782.29 | 10693.94 | 2088.35 | 4020.90 | 2245.66 | 4983.50 | 1914.22 | 612.17 |
| 6 | 1 | 12774.76 | 11537.12 | 1237.64 | 4125.19 | 2281.52 | 5995.10 | 2869.86 | 569.17 |
| 5 | 1 | 12772.79 | 11618.87 | 1153.91 | 4100.08 | 2221.48 | 6006.45 | 2928.71 | 709.32 |
| 2 | 1 | 12783.32 | 11266.94 | 1516.38 | 4087.34 | 2379.73 | 5665.21 | 2552.62 | 416.87 |
| 3 | 1 | 12776.36 | 11268.52 | 1507.83 | 4103.79 | 2384.81 | 5735.21 | 2577.36 | 384.06 |

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
| 4 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 7 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 6 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 5 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 2 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 3 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3014.08 | 4423.59 |
| gpu_idle | 1651.09 | 1651.09 |
| cp_communication | 1162.46 | 1269.14 |
| expert_gemm_path | 1129.20 | 2183.96 |
| fsdp_gather:expert | 905.32 | 4266.00 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3177.99 | 4626.05 |
| gpu_idle | 1373.70 | 1373.70 |
| expert_gemm_path | 1231.05 | 2153.76 |
| cp_communication | 1013.15 | 1177.94 |
| fsdp_gather:expert | 749.51 | 4295.48 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3143.82 | 4574.72 |
| gpu_idle | 1336.93 | 1336.93 |
| expert_gemm_path | 1097.66 | 2168.49 |
| cp_communication | 1056.09 | 1295.14 |
| fsdp_gather:expert | 976.64 | 4479.44 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3168.53 | 4622.22 |
| gpu_idle | 2088.35 | 2088.35 |
| expert_gemm_path | 1237.65 | 2152.68 |
| fsdp_gather:expert | 902.52 | 4414.52 |
| other_compute | 529.52 | 932.48 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3118.94 | 4547.41 |
| gpu_idle | 1237.64 | 1237.64 |
| expert_gemm_path | 1126.53 | 2142.73 |
| cp_communication | 1060.92 | 1323.89 |
| fsdp_gather:expert | 967.36 | 4499.63 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3138.29 | 4564.39 |
| expert_gemm_path | 1194.74 | 2117.95 |
| gpu_idle | 1153.91 | 1153.91 |
| cp_communication | 1058.40 | 1296.04 |
| fsdp_gather:expert | 889.44 | 4449.42 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3156.89 | 4600.02 |
| gpu_idle | 1516.38 | 1516.38 |
| expert_gemm_path | 1155.74 | 2182.09 |
| cp_communication | 1038.83 | 1178.25 |
| fsdp_gather:expert | 899.21 | 4284.64 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3148.52 | 4589.12 |
| gpu_idle | 1507.83 | 1507.83 |
| expert_gemm_path | 1110.94 | 2191.17 |
| cp_communication | 1053.50 | 1205.76 |
| fsdp_gather:expert | 965.65 | 4350.48 |

## Expert-GEMM hardware samples

Category-exclusive samples only. Excludes launch preparation and gaps. Device-wide counters are not per-kernel rooflines.

| Rank | Metric | Exclusive mean | Sample count | Exclusive fraction |
|---|---|---:|---:|---:|
| 0 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.78e+09 (not trustworthy as MHz) | 11232 | 51.6% |
| 0 | SMs Active [Throughput %] | 83.98 | 11232 | 51.6% |
| 0 | SM Issue [Throughput %] | 4.83 | 11232 | 51.6% |
| 0 | Tensor Active [Throughput %] | 54.23 | 11232 | 51.6% |
| 0 | DRAM Read Bandwidth [Throughput %] | 33.58 | 11232 | 51.6% |
| 0 | DRAM Write Bandwidth [Throughput %] | 4.04 | 11232 | 51.6% |
| 0 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11232 | 51.6% |
| 0 | NVLink RX Requests User Data [Throughput %] | 0.02 | 11232 | 51.6% |
| 0 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11232 | 51.6% |
| 0 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11232 | 51.6% |
| 0 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11232 | 51.6% |
| 0 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11232 | 51.6% |
| 0 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11232 | 51.6% |
| 0 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11232 | 51.6% |
| 1 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.74e+09 (not trustworthy as MHz) | 12324 | 57.1% |
| 1 | SMs Active [Throughput %] | 88.22 | 12324 | 57.1% |
| 1 | SM Issue [Throughput %] | 5.15 | 12324 | 57.1% |
| 1 | Tensor Active [Throughput %] | 57.57 | 12324 | 57.1% |
| 1 | DRAM Read Bandwidth [Throughput %] | 34.96 | 12324 | 57.1% |
| 1 | DRAM Write Bandwidth [Throughput %] | 4.11 | 12324 | 57.1% |
| 1 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 12324 | 57.1% |
| 1 | NVLink RX Requests User Data [Throughput %] | 0.00 | 12324 | 57.1% |
| 1 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12324 | 57.1% |
| 1 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12324 | 57.1% |
| 1 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12324 | 57.1% |
| 1 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12324 | 57.1% |
| 1 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12324 | 57.1% |
| 1 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12324 | 57.1% |
| 4 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.77e+09 (not trustworthy as MHz) | 10995 | 50.7% |
| 4 | SMs Active [Throughput %] | 87.54 | 10995 | 50.7% |
| 4 | SM Issue [Throughput %] | 5.06 | 10995 | 50.7% |
| 4 | Tensor Active [Throughput %] | 56.55 | 10995 | 50.7% |
| 4 | DRAM Read Bandwidth [Throughput %] | 34.70 | 10995 | 50.7% |
| 4 | DRAM Write Bandwidth [Throughput %] | 4.24 | 10995 | 50.7% |
| 4 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 10995 | 50.7% |
| 4 | NVLink RX Requests User Data [Throughput %] | 0.00 | 10995 | 50.7% |
| 4 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 10995 | 50.7% |
| 4 | NVLink RX Responses User Data [Throughput %] | 0.00 | 10995 | 50.7% |
| 4 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 10995 | 50.7% |
| 4 | NVLink TX Requests User Data [Throughput %] | 0.00 | 10995 | 50.7% |
| 4 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 10995 | 50.7% |
| 4 | NVLink TX Responses User Data [Throughput %] | 0.00 | 10995 | 50.7% |
| 7 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.7e+09 (not trustworthy as MHz) | 12328 | 57.4% |
| 7 | SMs Active [Throughput %] | 90.90 | 12328 | 57.4% |
| 7 | SM Issue [Throughput %] | 5.36 | 12328 | 57.4% |
| 7 | Tensor Active [Throughput %] | 59.73 | 12328 | 57.4% |
| 7 | DRAM Read Bandwidth [Throughput %] | 35.48 | 12328 | 57.4% |
| 7 | DRAM Write Bandwidth [Throughput %] | 4.16 | 12328 | 57.4% |
| 7 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 12328 | 57.4% |
| 7 | NVLink RX Requests User Data [Throughput %] | 0.01 | 12328 | 57.4% |
| 7 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12328 | 57.4% |
| 7 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12328 | 57.4% |
| 7 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12328 | 57.4% |
| 7 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12328 | 57.4% |
| 7 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12328 | 57.4% |
| 7 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12328 | 57.4% |
| 6 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.77e+09 (not trustworthy as MHz) | 11238 | 52.5% |
| 6 | SMs Active [Throughput %] | 89.44 | 11238 | 52.5% |
| 6 | SM Issue [Throughput %] | 5.19 | 11238 | 52.5% |
| 6 | Tensor Active [Throughput %] | 57.90 | 11238 | 52.5% |
| 6 | DRAM Read Bandwidth [Throughput %] | 35.51 | 11238 | 52.5% |
| 6 | DRAM Write Bandwidth [Throughput %] | 4.30 | 11238 | 52.5% |
| 6 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11238 | 52.5% |
| 6 | NVLink RX Requests User Data [Throughput %] | 0.00 | 11238 | 52.5% |
| 6 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11238 | 52.5% |
| 6 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11238 | 52.5% |
| 6 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11238 | 52.5% |
| 6 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11238 | 52.5% |
| 6 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11238 | 52.5% |
| 6 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11238 | 52.5% |
| 5 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.76e+09 (not trustworthy as MHz) | 11943 | 56.4% |
| 5 | SMs Active [Throughput %] | 90.53 | 11943 | 56.4% |
| 5 | SM Issue [Throughput %] | 5.30 | 11943 | 56.4% |
| 5 | Tensor Active [Throughput %] | 58.66 | 11943 | 56.4% |
| 5 | DRAM Read Bandwidth [Throughput %] | 36.19 | 11943 | 56.4% |
| 5 | DRAM Write Bandwidth [Throughput %] | 4.25 | 11943 | 56.4% |
| 5 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11943 | 56.4% |
| 5 | NVLink RX Requests User Data [Throughput %] | 0.00 | 11943 | 56.4% |
| 5 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11943 | 56.4% |
| 5 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11943 | 56.4% |
| 5 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11943 | 56.4% |
| 5 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11943 | 56.4% |
| 5 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11943 | 56.4% |
| 5 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11943 | 56.4% |
| 2 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.76e+09 (not trustworthy as MHz) | 11501 | 52.9% |
| 2 | SMs Active [Throughput %] | 85.84 | 11501 | 52.9% |
| 2 | SM Issue [Throughput %] | 4.89 | 11501 | 52.9% |
| 2 | Tensor Active [Throughput %] | 55.92 | 11501 | 52.9% |
| 2 | DRAM Read Bandwidth [Throughput %] | 33.94 | 11501 | 52.9% |
| 2 | DRAM Write Bandwidth [Throughput %] | 4.06 | 11501 | 52.9% |
| 2 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11501 | 52.9% |
| 2 | NVLink RX Requests User Data [Throughput %] | 0.00 | 11501 | 52.9% |
| 2 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11501 | 52.9% |
| 2 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11501 | 52.9% |
| 2 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11501 | 52.9% |
| 2 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11501 | 52.9% |
| 2 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11501 | 52.9% |
| 2 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11501 | 52.9% |
| 3 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.78e+09 (not trustworthy as MHz) | 11088 | 50.6% |
| 3 | SMs Active [Throughput %] | 83.32 | 11088 | 50.6% |
| 3 | SM Issue [Throughput %] | 4.81 | 11088 | 50.6% |
| 3 | Tensor Active [Throughput %] | 53.73 | 11088 | 50.6% |
| 3 | DRAM Read Bandwidth [Throughput %] | 33.39 | 11088 | 50.6% |
| 3 | DRAM Write Bandwidth [Throughput %] | 4.05 | 11088 | 50.6% |
| 3 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11088 | 50.6% |
| 3 | NVLink RX Requests User Data [Throughput %] | 0.00 | 11088 | 50.6% |
| 3 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11088 | 50.6% |
| 3 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11088 | 50.6% |
| 3 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11088 | 50.6% |
| 3 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11088 | 50.6% |
| 3 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11088 | 50.6% |
| 3 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11088 | 50.6% |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
