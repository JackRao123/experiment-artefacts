# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/nsys_workflow_20260911/devbox-ep8-te-metadata-v2/metrics.sqlite`
- Ranks: 0, 3, 7, 1, 6, 5, 2, 4; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: 1.242353541224972%.
- Hardware metrics present: True; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 10420.62 | 10213.67 | 206.95 | 3546.80 | 342.30 | 3402.66 | 3325.31 | 3052.01 |
| 3 | 1 | 10418.37 | 10210.48 | 207.89 | 3368.05 | 316.94 | 3224.58 | 3145.67 | 2896.34 |
| 7 | 1 | 10415.62 | 10203.69 | 211.93 | 3267.83 | 273.99 | 3120.62 | 3041.75 | 2843.20 |
| 1 | 1 | 10408.60 | 10208.97 | 199.63 | 3245.06 | 261.14 | 3108.45 | 3030.95 | 2833.34 |
| 6 | 1 | 10407.20 | 10209.84 | 197.36 | 3092.81 | 296.09 | 2960.21 | 2881.44 | 2646.24 |
| 5 | 1 | 10415.19 | 10202.00 | 213.19 | 3338.21 | 273.65 | 3185.85 | 3111.01 | 2903.06 |
| 2 | 1 | 10406.54 | 10203.26 | 203.28 | 3331.61 | 290.87 | 3190.37 | 3113.97 | 2887.99 |
| 4 | 1 | 10409.91 | 10186.36 | 223.55 | 3297.33 | 275.05 | 3135.35 | 3059.71 | 2860.87 |

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
| 7 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 1 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 6 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 5 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 2 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 4 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |

## Forward expert-wrapper CUDA API costs

Original + recomputed forwards, averaged per FB. API residence can overlap GPU work; it is not additive or automatically recoverable time.

| Rank | API | Calls/FB | Host API ms/FB |
|---|---|---:|---:|
| 0 | cuLaunchKernelEx | 9512 | 44.55 |
| 0 | cudaEventRecord_v3020 | 1500 | 5.61 |
| 0 | cudaStreamWaitEvent_v3020 | 2400 | 2.14 |
| 3 | cuLaunchKernelEx | 9500 | 53.87 |
| 3 | cudaEventRecord_v3020 | 1500 | 6.79 |
| 3 | cuKernelGetAttribute | 19000 | 2.11 |
| 7 | cuLaunchKernelEx | 9512 | 41.22 |
| 7 | cudaEventRecord_v3020 | 1500 | 4.91 |
| 7 | cuKernelGetAttribute | 19024 | 2.08 |
| 1 | cuLaunchKernelEx | 9532 | 53.00 |
| 1 | cudaEventRecord_v3020 | 1500 | 6.69 |
| 1 | cuKernelGetAttribute | 19064 | 2.22 |
| 6 | cuLaunchKernelEx | 9512 | 41.02 |
| 6 | cudaEventRecord_v3020 | 1500 | 5.03 |
| 6 | cudaStreamWaitEvent_v3020 | 2400 | 2.09 |
| 5 | cuLaunchKernelEx | 9512 | 41.22 |
| 5 | cudaEventRecord_v3020 | 1500 | 4.70 |
| 5 | cuKernelGetAttribute | 19024 | 2.03 |
| 2 | cuLaunchKernelEx | 9524 | 53.90 |
| 2 | cudaEventRecord_v3020 | 1500 | 6.77 |
| 2 | cudaStreamWaitEvent_v3020 | 2400 | 2.45 |
| 4 | cuLaunchKernelEx | 9528 | 43.21 |
| 4 | cudaEventRecord_v3020 | 1500 | 5.33 |
| 4 | cuKernelGetAttribute | 19056 | 2.31 |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3960.52 | 4026.22 |
| expert_dispatch_combine | 2799.67 | 2953.07 |
| expert_gemm_path | 1256.42 | 1256.42 |
| other_compute | 706.39 | 716.10 |
| other_gemm | 561.27 | 566.93 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4098.67 | 4163.27 |
| expert_dispatch_combine | 2730.37 | 2881.05 |
| expert_gemm_path | 1295.55 | 1295.55 |
| other_compute | 709.06 | 721.42 |
| other_gemm | 554.77 | 560.37 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4153.38 | 4217.86 |
| expert_dispatch_combine | 2659.78 | 2812.86 |
| expert_gemm_path | 1265.90 | 1265.90 |
| other_compute | 736.76 | 749.99 |
| other_gemm | 577.36 | 582.42 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4151.33 | 4215.37 |
| expert_dispatch_combine | 2662.47 | 2814.94 |
| expert_gemm_path | 1308.09 | 1308.09 |
| other_compute | 748.71 | 760.85 |
| other_gemm | 572.34 | 577.43 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4118.05 | 4181.83 |
| expert_dispatch_combine | 2482.49 | 2634.68 |
| expert_gemm_path | 1447.80 | 1447.80 |
| other_compute | 762.90 | 776.37 |
| other_gemm | 556.58 | 561.99 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4140.49 | 4201.20 |
| expert_dispatch_combine | 2706.71 | 2859.52 |
| expert_gemm_path | 1231.63 | 1231.63 |
| other_compute | 735.73 | 748.63 |
| other_gemm | 558.73 | 563.85 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4112.60 | 4176.11 |
| expert_dispatch_combine | 2718.78 | 2869.55 |
| expert_gemm_path | 1279.40 | 1279.40 |
| other_compute | 706.33 | 717.85 |
| other_gemm | 562.81 | 568.11 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4152.81 | 4214.04 |
| expert_dispatch_combine | 2673.99 | 2825.98 |
| expert_gemm_path | 1248.51 | 1248.51 |
| other_compute | 742.04 | 754.84 |
| other_gemm | 556.85 | 562.40 |

## Expert-GEMM hardware samples

Category-exclusive samples only. Excludes launch preparation and gaps. Device-wide counters are not per-kernel rooflines.

| Rank | Metric | Exclusive mean | Sample count | Exclusive fraction |
|---|---|---:|---:|---:|
| 0 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.57e+09 (not trustworthy as MHz) | 12562 | 100.0% |
| 0 | SMs Active [Throughput %] | 98.61 | 12562 | 100.0% |
| 0 | SM Issue [Throughput %] | 5.04 | 12562 | 100.0% |
| 0 | Tensor Active [Throughput %] | 91.41 | 12562 | 100.0% |
| 0 | DRAM Read Bandwidth [Throughput %] | 18.94 | 12562 | 100.0% |
| 0 | DRAM Write Bandwidth [Throughput %] | 5.84 | 12562 | 100.0% |
| 0 | NVLink RX Requests Protocol Data [Throughput %] | 0.42 | 12562 | 100.0% |
| 0 | NVLink RX Requests User Data [Throughput %] | 0.01 | 12562 | 100.0% |
| 0 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12562 | 100.0% |
| 0 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12562 | 100.0% |
| 0 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12562 | 100.0% |
| 0 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12562 | 100.0% |
| 0 | NVLink TX Responses Protocol Data [Throughput %] | 0.42 | 12562 | 100.0% |
| 0 | NVLink TX Responses User Data [Throughput %] | 0.42 | 12562 | 100.0% |
| 3 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.59e+09 (not trustworthy as MHz) | 12947 | 100.0% |
| 3 | SMs Active [Throughput %] | 98.63 | 12947 | 100.0% |
| 3 | SM Issue [Throughput %] | 4.98 | 12947 | 100.0% |
| 3 | Tensor Active [Throughput %] | 91.69 | 12947 | 100.0% |
| 3 | DRAM Read Bandwidth [Throughput %] | 18.97 | 12947 | 100.0% |
| 3 | DRAM Write Bandwidth [Throughput %] | 5.92 | 12947 | 100.0% |
| 3 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 12947 | 100.0% |
| 3 | NVLink RX Requests User Data [Throughput %] | 0.00 | 12947 | 100.0% |
| 3 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12947 | 100.0% |
| 3 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12947 | 100.0% |
| 3 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12947 | 100.0% |
| 3 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12947 | 100.0% |
| 3 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12947 | 100.0% |
| 3 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12947 | 100.0% |
| 7 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.5e+09 (not trustworthy as MHz) | 12655 | 100.0% |
| 7 | SMs Active [Throughput %] | 98.60 | 12655 | 100.0% |
| 7 | SM Issue [Throughput %] | 5.05 | 12655 | 100.0% |
| 7 | Tensor Active [Throughput %] | 91.45 | 12655 | 100.0% |
| 7 | DRAM Read Bandwidth [Throughput %] | 18.49 | 12655 | 100.0% |
| 7 | DRAM Write Bandwidth [Throughput %] | 5.56 | 12655 | 100.0% |
| 7 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 12655 | 100.0% |
| 7 | NVLink RX Requests User Data [Throughput %] | 0.00 | 12655 | 100.0% |
| 7 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12655 | 100.0% |
| 7 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12655 | 100.0% |
| 7 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12655 | 100.0% |
| 7 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12655 | 100.0% |
| 7 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12655 | 100.0% |
| 7 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12655 | 100.0% |
| 1 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.53e+09 (not trustworthy as MHz) | 13082 | 100.0% |
| 1 | SMs Active [Throughput %] | 98.74 | 13082 | 100.0% |
| 1 | SM Issue [Throughput %] | 5.09 | 13082 | 100.0% |
| 1 | Tensor Active [Throughput %] | 91.72 | 13082 | 100.0% |
| 1 | DRAM Read Bandwidth [Throughput %] | 18.51 | 13082 | 100.0% |
| 1 | DRAM Write Bandwidth [Throughput %] | 5.76 | 13082 | 100.0% |
| 1 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 13082 | 100.0% |
| 1 | NVLink RX Requests User Data [Throughput %] | 0.00 | 13082 | 100.0% |
| 1 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 13082 | 100.0% |
| 1 | NVLink RX Responses User Data [Throughput %] | 0.00 | 13082 | 100.0% |
| 1 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 13082 | 100.0% |
| 1 | NVLink TX Requests User Data [Throughput %] | 0.00 | 13082 | 100.0% |
| 1 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 13082 | 100.0% |
| 1 | NVLink TX Responses User Data [Throughput %] | 0.00 | 13082 | 100.0% |
| 6 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.57e+09 (not trustworthy as MHz) | 14477 | 100.0% |
| 6 | SMs Active [Throughput %] | 98.87 | 14477 | 100.0% |
| 6 | SM Issue [Throughput %] | 4.86 | 14477 | 100.0% |
| 6 | Tensor Active [Throughput %] | 92.59 | 14477 | 100.0% |
| 6 | DRAM Read Bandwidth [Throughput %] | 18.46 | 14477 | 100.0% |
| 6 | DRAM Write Bandwidth [Throughput %] | 5.90 | 14477 | 100.0% |
| 6 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 14477 | 100.0% |
| 6 | NVLink RX Requests User Data [Throughput %] | 0.00 | 14477 | 100.0% |
| 6 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 14477 | 100.0% |
| 6 | NVLink RX Responses User Data [Throughput %] | 0.00 | 14477 | 100.0% |
| 6 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 14477 | 100.0% |
| 6 | NVLink TX Requests User Data [Throughput %] | 0.00 | 14477 | 100.0% |
| 6 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 14477 | 100.0% |
| 6 | NVLink TX Responses User Data [Throughput %] | 0.00 | 14477 | 100.0% |
| 5 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.57e+09 (not trustworthy as MHz) | 12319 | 100.0% |
| 5 | SMs Active [Throughput %] | 98.53 | 12319 | 100.0% |
| 5 | SM Issue [Throughput %] | 5.06 | 12319 | 100.0% |
| 5 | Tensor Active [Throughput %] | 91.10 | 12319 | 100.0% |
| 5 | DRAM Read Bandwidth [Throughput %] | 19.25 | 12319 | 100.0% |
| 5 | DRAM Write Bandwidth [Throughput %] | 5.82 | 12319 | 100.0% |
| 5 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 12319 | 100.0% |
| 5 | NVLink RX Requests User Data [Throughput %] | 0.00 | 12319 | 100.0% |
| 5 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12319 | 100.0% |
| 5 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12319 | 100.0% |
| 5 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12319 | 100.0% |
| 5 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12319 | 100.0% |
| 5 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12319 | 100.0% |
| 5 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12319 | 100.0% |
| 2 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.55e+09 (not trustworthy as MHz) | 12793 | 100.0% |
| 2 | SMs Active [Throughput %] | 98.70 | 12793 | 100.0% |
| 2 | SM Issue [Throughput %] | 5.01 | 12793 | 100.0% |
| 2 | Tensor Active [Throughput %] | 91.67 | 12793 | 100.0% |
| 2 | DRAM Read Bandwidth [Throughput %] | 18.81 | 12793 | 100.0% |
| 2 | DRAM Write Bandwidth [Throughput %] | 5.79 | 12793 | 100.0% |
| 2 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 12793 | 100.0% |
| 2 | NVLink RX Requests User Data [Throughput %] | 0.00 | 12793 | 100.0% |
| 2 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12793 | 100.0% |
| 2 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12793 | 100.0% |
| 2 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12793 | 100.0% |
| 2 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12793 | 100.0% |
| 2 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12793 | 100.0% |
| 2 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12793 | 100.0% |
| 4 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.58e+09 (not trustworthy as MHz) | 12492 | 100.0% |
| 4 | SMs Active [Throughput %] | 98.59 | 12492 | 100.0% |
| 4 | SM Issue [Throughput %] | 5.07 | 12492 | 100.0% |
| 4 | Tensor Active [Throughput %] | 91.20 | 12492 | 100.0% |
| 4 | DRAM Read Bandwidth [Throughput %] | 18.88 | 12492 | 100.0% |
| 4 | DRAM Write Bandwidth [Throughput %] | 5.85 | 12492 | 100.0% |
| 4 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 12492 | 100.0% |
| 4 | NVLink RX Requests User Data [Throughput %] | 0.00 | 12492 | 100.0% |
| 4 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12492 | 100.0% |
| 4 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12492 | 100.0% |
| 4 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12492 | 100.0% |
| 4 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12492 | 100.0% |
| 4 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12492 | 100.0% |
| 4 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12492 | 100.0% |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
