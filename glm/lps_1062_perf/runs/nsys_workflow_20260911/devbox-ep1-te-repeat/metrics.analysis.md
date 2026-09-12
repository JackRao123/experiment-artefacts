# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/nsys_workflow_20260911/devbox-ep1-te-repeat/metrics.sqlite`
- Ranks: 7, 5, 2, 3, 1, 0, 4, 6; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: 9.49282069094628%.
- Hardware metrics present: True; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 1 | 12919.08 | 10807.31 | 2111.77 | 4113.64 | 2204.59 | 5160.05 | 1983.59 | 521.37 |
| 5 | 1 | 12910.72 | 11599.80 | 1310.93 | 4235.83 | 2234.83 | 6041.49 | 2907.36 | 596.44 |
| 2 | 1 | 12911.64 | 11543.08 | 1368.56 | 4265.17 | 2230.63 | 5971.64 | 2878.22 | 540.88 |
| 3 | 1 | 12913.30 | 11451.36 | 1461.94 | 4236.77 | 2265.30 | 5959.37 | 2756.26 | 502.89 |
| 1 | 1 | 12939.81 | 11233.79 | 1706.02 | 4214.35 | 2394.55 | 5627.78 | 2489.65 | 362.64 |
| 0 | 1 | 12916.82 | 11484.02 | 1432.80 | 4486.50 | 2323.48 | 6008.47 | 3035.18 | 670.13 |
| 4 | 1 | 12919.82 | 11256.85 | 1662.97 | 4195.88 | 2405.35 | 5715.66 | 2515.34 | 405.28 |
| 6 | 1 | 12902.41 | 11197.81 | 1704.60 | 4186.00 | 2491.07 | 5627.26 | 2463.75 | 452.04 |

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
| 7 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 5 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 2 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 3 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 1 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 0 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 4 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 6 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |

## Forward expert-wrapper CUDA API costs

Original + recomputed forwards, averaged per FB. API residence can overlap GPU work; it is not additive or automatically recoverable time.

| Rank | API | Calls/FB | Host API ms/FB |
|---|---|---:|---:|
| 7 | cuLaunchKernelEx | 74352 | 248.68 |
| 7 | cudaLaunchKernelExC_v11060 | 4468 | 13.51 |
| 7 | cuKernelGetAttribute | 148704 | 10.55 |
| 5 | cuLaunchKernelEx | 74464 | 256.33 |
| 5 | cudaLaunchKernelExC_v11060 | 4534 | 14.29 |
| 5 | cuKernelGetAttribute | 148928 | 10.59 |
| 2 | cuLaunchKernelEx | 74528 | 258.20 |
| 2 | cudaLaunchKernelExC_v11060 | 4452 | 14.21 |
| 2 | cuKernelGetAttribute | 149056 | 9.91 |
| 3 | cuLaunchKernelEx | 74428 | 275.50 |
| 3 | cudaLaunchKernelExC_v11060 | 4648 | 15.53 |
| 3 | cuKernelGetAttribute | 148856 | 10.38 |
| 1 | cuLaunchKernelEx | 74512 | 322.33 |
| 1 | cudaLaunchKernelExC_v11060 | 4428 | 16.78 |
| 1 | cuKernelGetAttribute | 149024 | 10.85 |
| 0 | cuLaunchKernelEx | 74676 | 334.92 |
| 0 | cudaLaunchKernelExC_v11060 | 4410 | 17.47 |
| 0 | cuKernelGetAttribute | 149352 | 10.39 |
| 4 | cuLaunchKernelEx | 74464 | 316.92 |
| 4 | cudaLaunchKernelExC_v11060 | 4624 | 18.05 |
| 4 | cuKernelGetAttribute | 148928 | 11.03 |
| 6 | cuLaunchKernelEx | 74476 | 380.62 |
| 6 | cudaLaunchKernelExC_v11060 | 4664 | 22.10 |
| 6 | cuKernelGetAttribute | 148952 | 12.33 |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3157.07 | 4611.12 |
| gpu_idle | 2111.77 | 2111.77 |
| expert_gemm_path | 1172.14 | 2171.76 |
| fsdp_gather:expert | 1068.75 | 4624.75 |
| other_compute | 529.59 | 949.04 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3126.38 | 4556.74 |
| gpu_idle | 1310.93 | 1310.93 |
| expert_gemm_path | 1135.87 | 2149.54 |
| fsdp_gather:expert | 1063.20 | 4603.92 |
| cp_communication | 1002.54 | 1259.49 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3139.22 | 4584.01 |
| gpu_idle | 1368.56 | 1368.56 |
| expert_gemm_path | 1151.43 | 2155.38 |
| fsdp_gather:expert | 1124.85 | 4559.18 |
| cp_communication | 1003.53 | 1203.23 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3124.51 | 4556.88 |
| gpu_idle | 1461.94 | 1461.94 |
| expert_gemm_path | 1102.18 | 2164.29 |
| fsdp_gather:expert | 1091.99 | 4571.32 |
| cp_communication | 1017.31 | 1240.81 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3155.95 | 4605.52 |
| gpu_idle | 1706.02 | 1706.02 |
| expert_gemm_path | 1177.75 | 2185.24 |
| fsdp_gather:expert | 991.81 | 4353.76 |
| cp_communication | 970.32 | 1055.53 |

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3006.02 | 4415.83 |
| gpu_idle | 1432.80 | 1432.80 |
| expert_gemm_path | 1197.16 | 2124.23 |
| fsdp_gather:expert | 1162.05 | 4500.38 |
| cp_communication | 1092.12 | 1208.51 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3126.68 | 4563.53 |
| gpu_idle | 1662.97 | 1662.97 |
| expert_gemm_path | 1119.30 | 2197.19 |
| cp_communication | 1003.79 | 1140.67 |
| fsdp_gather:expert | 941.59 | 4361.63 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3116.92 | 4551.79 |
| gpu_idle | 1704.60 | 1704.60 |
| expert_gemm_path | 1159.01 | 2201.47 |
| cp_communication | 979.91 | 1142.54 |
| fsdp_gather:expert | 816.17 | 4284.45 |

## Expert-GEMM hardware samples

Category-exclusive samples only. Excludes launch preparation and gaps. Device-wide counters are not per-kernel rooflines.

| Rank | Metric | Exclusive mean | Sample count | Exclusive fraction |
|---|---|---:|---:|---:|
| 7 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.71e+09 (not trustworthy as MHz) | 11750 | 54.0% |
| 7 | SMs Active [Throughput %] | 89.17 | 11750 | 54.0% |
| 7 | SM Issue [Throughput %] | 5.17 | 11750 | 54.0% |
| 7 | Tensor Active [Throughput %] | 58.59 | 11750 | 54.0% |
| 7 | DRAM Read Bandwidth [Throughput %] | 34.54 | 11750 | 54.0% |
| 7 | DRAM Write Bandwidth [Throughput %] | 4.16 | 11750 | 54.0% |
| 7 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11750 | 54.0% |
| 7 | NVLink RX Requests User Data [Throughput %] | 0.00 | 11750 | 54.0% |
| 7 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11750 | 54.0% |
| 7 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11750 | 54.0% |
| 7 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11750 | 54.0% |
| 7 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11750 | 54.0% |
| 7 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11750 | 54.0% |
| 7 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11750 | 54.0% |
| 5 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.76e+09 (not trustworthy as MHz) | 11358 | 52.8% |
| 5 | SMs Active [Throughput %] | 88.23 | 11358 | 52.8% |
| 5 | SM Issue [Throughput %] | 5.06 | 11358 | 52.8% |
| 5 | Tensor Active [Throughput %] | 57.17 | 11358 | 52.8% |
| 5 | DRAM Read Bandwidth [Throughput %] | 34.95 | 11358 | 52.8% |
| 5 | DRAM Write Bandwidth [Throughput %] | 4.23 | 11358 | 52.8% |
| 5 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11358 | 52.8% |
| 5 | NVLink RX Requests User Data [Throughput %] | 0.00 | 11358 | 52.8% |
| 5 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11358 | 52.8% |
| 5 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11358 | 52.8% |
| 5 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11358 | 52.8% |
| 5 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11358 | 52.8% |
| 5 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11358 | 52.8% |
| 5 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11358 | 52.8% |
| 2 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.75e+09 (not trustworthy as MHz) | 11489 | 53.3% |
| 2 | SMs Active [Throughput %] | 87.51 | 11489 | 53.3% |
| 2 | SM Issue [Throughput %] | 5.02 | 11489 | 53.3% |
| 2 | Tensor Active [Throughput %] | 56.90 | 11489 | 53.3% |
| 2 | DRAM Read Bandwidth [Throughput %] | 34.57 | 11489 | 53.3% |
| 2 | DRAM Write Bandwidth [Throughput %] | 4.15 | 11489 | 53.3% |
| 2 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11489 | 53.3% |
| 2 | NVLink RX Requests User Data [Throughput %] | 0.00 | 11489 | 53.3% |
| 2 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11489 | 53.3% |
| 2 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11489 | 53.3% |
| 2 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11489 | 53.3% |
| 2 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11489 | 53.3% |
| 2 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11489 | 53.3% |
| 2 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11489 | 53.3% |
| 3 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.78e+09 (not trustworthy as MHz) | 10977 | 50.8% |
| 3 | SMs Active [Throughput %] | 85.36 | 10977 | 50.8% |
| 3 | SM Issue [Throughput %] | 4.97 | 10977 | 50.8% |
| 3 | Tensor Active [Throughput %] | 55.02 | 10977 | 50.8% |
| 3 | DRAM Read Bandwidth [Throughput %] | 34.23 | 10977 | 50.8% |
| 3 | DRAM Write Bandwidth [Throughput %] | 4.16 | 10977 | 50.8% |
| 3 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 10977 | 50.8% |
| 3 | NVLink RX Requests User Data [Throughput %] | 0.00 | 10977 | 50.8% |
| 3 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 10977 | 50.8% |
| 3 | NVLink RX Responses User Data [Throughput %] | 0.00 | 10977 | 50.8% |
| 3 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 10977 | 50.8% |
| 3 | NVLink TX Requests User Data [Throughput %] | 0.00 | 10977 | 50.8% |
| 3 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 10977 | 50.8% |
| 3 | NVLink TX Responses User Data [Throughput %] | 0.00 | 10977 | 50.8% |
| 1 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.74e+09 (not trustworthy as MHz) | 11786 | 53.9% |
| 1 | SMs Active [Throughput %] | 84.59 | 11786 | 53.9% |
| 1 | SM Issue [Throughput %] | 4.94 | 11786 | 53.9% |
| 1 | Tensor Active [Throughput %] | 55.14 | 11786 | 53.9% |
| 1 | DRAM Read Bandwidth [Throughput %] | 33.43 | 11786 | 53.9% |
| 1 | DRAM Write Bandwidth [Throughput %] | 4.03 | 11786 | 53.9% |
| 1 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11786 | 53.9% |
| 1 | NVLink RX Requests User Data [Throughput %] | 0.01 | 11786 | 53.9% |
| 1 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11786 | 53.9% |
| 1 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11786 | 53.9% |
| 1 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11786 | 53.9% |
| 1 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11786 | 53.9% |
| 1 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11786 | 53.9% |
| 1 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11786 | 53.9% |
| 0 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.77e+09 (not trustworthy as MHz) | 11934 | 56.3% |
| 0 | SMs Active [Throughput %] | 87.83 | 11934 | 56.3% |
| 0 | SM Issue [Throughput %] | 5.04 | 11934 | 56.3% |
| 0 | Tensor Active [Throughput %] | 56.93 | 11934 | 56.3% |
| 0 | DRAM Read Bandwidth [Throughput %] | 35.17 | 11934 | 56.3% |
| 0 | DRAM Write Bandwidth [Throughput %] | 4.13 | 11934 | 56.3% |
| 0 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11934 | 56.3% |
| 0 | NVLink RX Requests User Data [Throughput %] | 0.01 | 11934 | 56.3% |
| 0 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11934 | 56.3% |
| 0 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11934 | 56.3% |
| 0 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11934 | 56.3% |
| 0 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11934 | 56.3% |
| 0 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11934 | 56.3% |
| 0 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11934 | 56.3% |
| 4 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.78e+09 (not trustworthy as MHz) | 11172 | 51.0% |
| 4 | SMs Active [Throughput %] | 83.60 | 11172 | 51.0% |
| 4 | SM Issue [Throughput %] | 4.76 | 11172 | 51.0% |
| 4 | Tensor Active [Throughput %] | 53.94 | 11172 | 51.0% |
| 4 | DRAM Read Bandwidth [Throughput %] | 33.24 | 11172 | 51.0% |
| 4 | DRAM Write Bandwidth [Throughput %] | 4.03 | 11172 | 51.0% |
| 4 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11172 | 51.0% |
| 4 | NVLink RX Requests User Data [Throughput %] | 0.01 | 11172 | 51.0% |
| 4 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11172 | 51.0% |
| 4 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11172 | 51.0% |
| 4 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11172 | 51.0% |
| 4 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11172 | 51.0% |
| 4 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11172 | 51.0% |
| 4 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11172 | 51.0% |
| 6 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.78e+09 (not trustworthy as MHz) | 11599 | 52.6% |
| 6 | SMs Active [Throughput %] | 82.23 | 11599 | 52.6% |
| 6 | SM Issue [Throughput %] | 4.64 | 11599 | 52.6% |
| 6 | Tensor Active [Throughput %] | 53.19 | 11599 | 52.6% |
| 6 | DRAM Read Bandwidth [Throughput %] | 32.81 | 11599 | 52.6% |
| 6 | DRAM Write Bandwidth [Throughput %] | 3.92 | 11599 | 52.6% |
| 6 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11599 | 52.6% |
| 6 | NVLink RX Requests User Data [Throughput %] | 0.01 | 11599 | 52.6% |
| 6 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11599 | 52.6% |
| 6 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11599 | 52.6% |
| 6 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11599 | 52.6% |
| 6 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11599 | 52.6% |
| 6 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11599 | 52.6% |
| 6 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11599 | 52.6% |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
