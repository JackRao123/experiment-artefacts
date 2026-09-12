# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/nsys_workflow_20260911/devbox-ep1-te-metadata/metrics.sqlite`
- Ranks: 0, 4, 6, 7, 5, 2, 1, 3; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: 1.8197991509377287%.
- Hardware metrics present: True; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 11495.52 | 10437.26 | 1058.26 | 2968.86 | 1351.05 | 4974.45 | 1891.98 | 462.21 |
| 4 | 1 | 11497.58 | 10806.19 | 691.39 | 2822.17 | 1107.18 | 5142.67 | 2113.12 | 734.10 |
| 6 | 1 | 11508.14 | 10637.41 | 870.73 | 2810.78 | 1234.39 | 5050.12 | 1922.13 | 554.69 |
| 7 | 1 | 11495.09 | 10600.23 | 894.86 | 2525.51 | 1065.57 | 4745.93 | 1612.41 | 553.53 |
| 5 | 1 | 11497.73 | 10740.91 | 756.83 | 2820.79 | 1153.54 | 5076.88 | 2046.38 | 673.56 |
| 2 | 1 | 11495.42 | 10501.62 | 993.80 | 2648.58 | 1177.36 | 4903.57 | 1636.11 | 427.42 |
| 1 | 1 | 11500.07 | 10655.52 | 844.55 | 2784.99 | 1185.88 | 4948.41 | 1921.84 | 593.29 |
| 3 | 1 | 11503.70 | 10514.14 | 989.56 | 2775.29 | 1259.47 | 4905.75 | 1767.19 | 438.47 |

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
| 4 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 6 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 7 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 5 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 2 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 1 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |
| 3 | 0 | 150 | 149/149 | source-scoped sequential full-recompute pattern verified |

## Forward expert-wrapper CUDA API costs

Original + recomputed forwards, averaged per FB. API residence can overlap GPU work; it is not additive or automatically recoverable time.

| Rank | API | Calls/FB | Host API ms/FB |
|---|---|---:|---:|
| 0 | cuLaunchKernelEx | 74684 | 350.16 |
| 0 | cudaLaunchKernelExC_v11060 | 4384 | 18.79 |
| 0 | cuKernelGetAttribute | 149368 | 11.06 |
| 4 | cuLaunchKernelEx | 74488 | 256.20 |
| 4 | cudaLaunchKernelExC_v11060 | 4614 | 14.39 |
| 4 | cuKernelGetAttribute | 148976 | 10.15 |
| 6 | cuLaunchKernelEx | 74488 | 253.48 |
| 6 | cudaLaunchKernelExC_v11060 | 4670 | 14.17 |
| 6 | cuKernelGetAttribute | 148976 | 10.53 |
| 7 | cuLaunchKernelEx | 74316 | 286.56 |
| 7 | cudaLaunchKernelExC_v11060 | 4524 | 15.71 |
| 7 | cuKernelGetAttribute | 148632 | 10.57 |
| 5 | cuLaunchKernelEx | 74500 | 262.26 |
| 5 | cudaLaunchKernelExC_v11060 | 4574 | 14.57 |
| 5 | cuKernelGetAttribute | 149000 | 10.37 |
| 2 | cuLaunchKernelEx | 74520 | 283.10 |
| 2 | cudaLaunchKernelExC_v11060 | 4498 | 15.46 |
| 2 | cuKernelGetAttribute | 149040 | 10.36 |
| 1 | cuLaunchKernelEx | 74500 | 344.90 |
| 1 | cudaLaunchKernelExC_v11060 | 4464 | 17.90 |
| 1 | cuKernelGetAttribute | 149000 | 10.25 |
| 3 | cuLaunchKernelEx | 74468 | 348.34 |
| 3 | cudaLaunchKernelExC_v11060 | 4594 | 18.55 |
| 3 | cuKernelGetAttribute | 148936 | 10.55 |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3026.27 | 4446.98 |
| expert_gemm_path | 1153.73 | 2191.83 |
| gpu_idle | 1058.26 | 1058.26 |
| fsdp_gather:expert | 923.09 | 4241.94 |
| other_compute | 526.57 | 865.86 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3156.56 | 4611.22 |
| expert_gemm_path | 1218.08 | 2116.16 |
| fsdp_gather:expert | 863.03 | 4437.88 |
| gpu_idle | 691.39 | 691.39 |
| other_compute | 528.64 | 921.40 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3131.45 | 4587.16 |
| expert_gemm_path | 1150.63 | 2151.09 |
| gpu_idle | 870.73 | 870.73 |
| fsdp_gather:expert | 853.85 | 4418.43 |
| other_compute | 526.05 | 913.36 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3206.75 | 4680.83 |
| expert_gemm_path | 1261.86 | 2158.86 |
| gpu_idle | 894.86 | 894.86 |
| fsdp_gather:expert | 734.82 | 4319.98 |
| other_compute | 535.80 | 1006.38 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3156.46 | 4604.26 |
| expert_gemm_path | 1218.39 | 2120.29 |
| fsdp_gather:expert | 849.47 | 4382.21 |
| gpu_idle | 756.83 | 756.83 |
| other_compute | 527.05 | 924.09 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3170.05 | 4633.92 |
| expert_gemm_path | 1133.22 | 2190.20 |
| gpu_idle | 993.80 | 993.80 |
| fsdp_gather:expert | 778.89 | 4342.40 |
| other_compute | 528.70 | 986.93 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3190.55 | 4649.53 |
| expert_gemm_path | 1245.99 | 2152.63 |
| gpu_idle | 844.55 | 844.55 |
| fsdp_gather:expert | 831.50 | 4274.13 |
| other_compute | 532.14 | 903.72 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3162.05 | 4611.33 |
| expert_gemm_path | 1164.22 | 2168.66 |
| gpu_idle | 989.56 | 989.56 |
| fsdp_gather:expert | 812.64 | 4280.80 |
| other_compute | 530.07 | 929.87 |

## Expert-GEMM hardware samples

Category-exclusive samples only. Excludes launch preparation and gaps. Device-wide counters are not per-kernel rooflines.

| Rank | Metric | Exclusive mean | Sample count | Exclusive fraction |
|---|---|---:|---:|---:|
| 0 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.77e+09 (not trustworthy as MHz) | 11487 | 52.5% |
| 0 | SMs Active [Throughput %] | 82.84 | 11487 | 52.5% |
| 0 | SM Issue [Throughput %] | 4.79 | 11487 | 52.5% |
| 0 | Tensor Active [Throughput %] | 53.56 | 11487 | 52.5% |
| 0 | DRAM Read Bandwidth [Throughput %] | 33.04 | 11487 | 52.5% |
| 0 | DRAM Write Bandwidth [Throughput %] | 3.95 | 11487 | 52.5% |
| 0 | NVLink RX Requests Protocol Data [Throughput %] | 0.01 | 11487 | 52.5% |
| 0 | NVLink RX Requests User Data [Throughput %] | 0.04 | 11487 | 52.5% |
| 0 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11487 | 52.5% |
| 0 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11487 | 52.5% |
| 0 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11487 | 52.5% |
| 0 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11487 | 52.5% |
| 0 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11487 | 52.5% |
| 0 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11487 | 52.5% |
| 4 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.76e+09 (not trustworthy as MHz) | 12150 | 57.5% |
| 4 | SMs Active [Throughput %] | 90.26 | 12150 | 57.5% |
| 4 | SM Issue [Throughput %] | 5.22 | 12150 | 57.5% |
| 4 | Tensor Active [Throughput %] | 58.60 | 12150 | 57.5% |
| 4 | DRAM Read Bandwidth [Throughput %] | 35.94 | 12150 | 57.5% |
| 4 | DRAM Write Bandwidth [Throughput %] | 4.20 | 12150 | 57.5% |
| 4 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 12150 | 57.5% |
| 4 | NVLink RX Requests User Data [Throughput %] | 0.00 | 12150 | 57.5% |
| 4 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12150 | 57.5% |
| 4 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12150 | 57.5% |
| 4 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12150 | 57.5% |
| 4 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12150 | 57.5% |
| 4 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12150 | 57.5% |
| 4 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12150 | 57.5% |
| 6 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.77e+09 (not trustworthy as MHz) | 11512 | 53.5% |
| 6 | SMs Active [Throughput %] | 88.03 | 11512 | 53.5% |
| 6 | SM Issue [Throughput %] | 5.07 | 11512 | 53.5% |
| 6 | Tensor Active [Throughput %] | 56.99 | 11512 | 53.5% |
| 6 | DRAM Read Bandwidth [Throughput %] | 34.97 | 11512 | 53.5% |
| 6 | DRAM Write Bandwidth [Throughput %] | 4.19 | 11512 | 53.5% |
| 6 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11512 | 53.5% |
| 6 | NVLink RX Requests User Data [Throughput %] | 0.01 | 11512 | 53.5% |
| 6 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11512 | 53.5% |
| 6 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11512 | 53.5% |
| 6 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11512 | 53.5% |
| 6 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11512 | 53.5% |
| 6 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11512 | 53.5% |
| 6 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11512 | 53.5% |
| 7 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.7e+09 (not trustworthy as MHz) | 12641 | 58.4% |
| 7 | SMs Active [Throughput %] | 89.89 | 12641 | 58.4% |
| 7 | SM Issue [Throughput %] | 5.31 | 12641 | 58.4% |
| 7 | Tensor Active [Throughput %] | 59.27 | 12641 | 58.4% |
| 7 | DRAM Read Bandwidth [Throughput %] | 34.75 | 12641 | 58.4% |
| 7 | DRAM Write Bandwidth [Throughput %] | 4.06 | 12641 | 58.4% |
| 7 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 12641 | 58.4% |
| 7 | NVLink RX Requests User Data [Throughput %] | 0.01 | 12641 | 58.4% |
| 7 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12641 | 58.4% |
| 7 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12641 | 58.4% |
| 7 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12641 | 58.4% |
| 7 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12641 | 58.4% |
| 7 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12641 | 58.4% |
| 7 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12641 | 58.4% |
| 5 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.76e+09 (not trustworthy as MHz) | 12175 | 57.4% |
| 5 | SMs Active [Throughput %] | 89.70 | 12175 | 57.4% |
| 5 | SM Issue [Throughput %] | 5.22 | 12175 | 57.4% |
| 5 | Tensor Active [Throughput %] | 58.15 | 12175 | 57.4% |
| 5 | DRAM Read Bandwidth [Throughput %] | 35.80 | 12175 | 57.4% |
| 5 | DRAM Write Bandwidth [Throughput %] | 4.18 | 12175 | 57.4% |
| 5 | NVLink RX Requests Protocol Data [Throughput %] | 0.01 | 12175 | 57.4% |
| 5 | NVLink RX Requests User Data [Throughput %] | 0.02 | 12175 | 57.4% |
| 5 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12175 | 57.4% |
| 5 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12175 | 57.4% |
| 5 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12175 | 57.4% |
| 5 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12175 | 57.4% |
| 5 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12175 | 57.4% |
| 5 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12175 | 57.4% |
| 2 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.75e+09 (not trustworthy as MHz) | 11329 | 51.7% |
| 2 | SMs Active [Throughput %] | 86.14 | 11329 | 51.7% |
| 2 | SM Issue [Throughput %] | 4.98 | 11329 | 51.7% |
| 2 | Tensor Active [Throughput %] | 55.96 | 11329 | 51.7% |
| 2 | DRAM Read Bandwidth [Throughput %] | 33.96 | 11329 | 51.7% |
| 2 | DRAM Write Bandwidth [Throughput %] | 4.11 | 11329 | 51.7% |
| 2 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11329 | 51.7% |
| 2 | NVLink RX Requests User Data [Throughput %] | 0.00 | 11329 | 51.7% |
| 2 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11329 | 51.7% |
| 2 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11329 | 51.7% |
| 2 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11329 | 51.7% |
| 2 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11329 | 51.7% |
| 2 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11329 | 51.7% |
| 2 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11329 | 51.7% |
| 1 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.73e+09 (not trustworthy as MHz) | 12456 | 57.9% |
| 1 | SMs Active [Throughput %] | 88.38 | 12456 | 57.9% |
| 1 | SM Issue [Throughput %] | 5.18 | 12456 | 57.9% |
| 1 | Tensor Active [Throughput %] | 57.82 | 12456 | 57.9% |
| 1 | DRAM Read Bandwidth [Throughput %] | 34.91 | 12456 | 57.9% |
| 1 | DRAM Write Bandwidth [Throughput %] | 4.10 | 12456 | 57.9% |
| 1 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 12456 | 57.9% |
| 1 | NVLink RX Requests User Data [Throughput %] | 0.00 | 12456 | 57.9% |
| 1 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12456 | 57.9% |
| 1 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12456 | 57.9% |
| 1 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12456 | 57.9% |
| 1 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12456 | 57.9% |
| 1 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12456 | 57.9% |
| 1 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12456 | 57.9% |
| 3 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.77e+09 (not trustworthy as MHz) | 11634 | 53.7% |
| 3 | SMs Active [Throughput %] | 84.99 | 11634 | 53.7% |
| 3 | SM Issue [Throughput %] | 4.89 | 11634 | 53.7% |
| 3 | Tensor Active [Throughput %] | 55.16 | 11634 | 53.7% |
| 3 | DRAM Read Bandwidth [Throughput %] | 33.99 | 11634 | 53.7% |
| 3 | DRAM Write Bandwidth [Throughput %] | 4.04 | 11634 | 53.7% |
| 3 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 11634 | 53.7% |
| 3 | NVLink RX Requests User Data [Throughput %] | 0.01 | 11634 | 53.7% |
| 3 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 11634 | 53.7% |
| 3 | NVLink RX Responses User Data [Throughput %] | 0.00 | 11634 | 53.7% |
| 3 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 11634 | 53.7% |
| 3 | NVLink TX Requests User Data [Throughput %] | 0.00 | 11634 | 53.7% |
| 3 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 11634 | 53.7% |
| 3 | NVLink TX Responses User Data [Throughput %] | 0.00 | 11634 | 53.7% |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
