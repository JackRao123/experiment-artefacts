# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/nsys_workflow_20260911/devbox-ep8-grouped/metrics.sqlite`
- Ranks: 0, 2, 3, 7, 4, 6, 5, 1; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: 1.1375110387303211%.
- Hardware metrics present: True; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 11133.53 | 10828.81 | 304.72 | 4055.87 | 785.29 | 3814.14 | 3736.08 | 3007.13 |
| 2 | 1 | 11123.61 | 10841.85 | 281.75 | 3877.95 | 742.39 | 3663.78 | 3581.30 | 2875.45 |
| 3 | 1 | 11126.99 | 10836.16 | 290.84 | 3843.51 | 769.47 | 3621.26 | 3537.67 | 2825.63 |
| 7 | 1 | 11125.04 | 10827.69 | 297.35 | 3774.31 | 719.19 | 3542.40 | 3462.24 | 2814.32 |
| 4 | 1 | 11121.13 | 10831.23 | 289.90 | 3809.42 | 704.84 | 3587.02 | 3504.89 | 2840.44 |
| 6 | 1 | 11120.06 | 10830.44 | 289.62 | 3541.97 | 797.52 | 3320.92 | 3237.63 | 2530.02 |
| 5 | 1 | 11118.12 | 10825.77 | 292.35 | 3869.37 | 721.41 | 3641.01 | 3562.40 | 2891.52 |
| 1 | 1 | 11121.66 | 10828.34 | 293.32 | 3722.84 | 712.89 | 3492.11 | 3414.52 | 2749.65 |

### Capture diagnostics

- 25× on other/helper processes: Not all NVTX events might have been collected.
- 25× on other/helper processes: No NVTX events collected. Does the process use NVTX?
- 8× on captured trainer ranks: Not all NVTX events might have been collected.
- 8× on other/helper processes: CUDA profiling might have not been started correctly.
- 25× on other/helper processes: No CUDA events collected. Does the process use CUDA?
- 17× on other/helper processes: Not all CUDA events might have been collected.
- 8× on captured trainer ranks: Not all CUDA events might have been collected.

Capture completeness is qualified by these warnings. Complete step anchors and runtime links do not prove zero event loss.

## Expert-weight prefetch arrivals

Arrival relative to preceding-block compute completion; not a zero-communication speedup estimate.

| Rank | Step | Attributed gathers | Ready by preceding block end | Status |
|---|---:|---:|---|---|
| 0 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 2 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 3 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 7 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 4 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 6 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 5 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 1 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3985.09 | 4051.21 |
| expert_dispatch_combine | 3095.57 | 3249.18 |
| expert_gemm_path | 1426.07 | 1426.07 |
| other_compute | 710.82 | 720.99 |
| other_gemm | 563.14 | 568.67 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4119.78 | 4186.47 |
| expert_dispatch_combine | 3041.08 | 3192.01 |
| expert_gemm_path | 1433.95 | 1433.95 |
| other_compute | 706.61 | 720.61 |
| other_gemm | 563.24 | 569.00 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4132.89 | 4199.55 |
| expert_dispatch_combine | 3009.27 | 3159.94 |
| expert_gemm_path | 1480.67 | 1480.67 |
| other_compute | 713.49 | 727.80 |
| other_gemm | 555.69 | 562.10 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4197.24 | 4262.82 |
| expert_dispatch_combine | 2979.67 | 3133.12 |
| expert_gemm_path | 1409.20 | 1409.20 |
| other_compute | 745.69 | 758.70 |
| other_gemm | 579.82 | 585.40 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4177.63 | 4242.58 |
| expert_dispatch_combine | 2988.01 | 3140.25 |
| expert_gemm_path | 1409.19 | 1409.19 |
| other_compute | 745.40 | 760.20 |
| other_gemm | 557.96 | 564.28 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4155.76 | 4223.15 |
| expert_dispatch_combine | 2731.63 | 2884.06 |
| expert_gemm_path | 1653.90 | 1653.90 |
| other_compute | 770.99 | 785.11 |
| other_gemm | 559.62 | 565.42 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4161.13 | 4224.36 |
| expert_dispatch_combine | 3038.20 | 3191.18 |
| expert_gemm_path | 1371.30 | 1371.30 |
| other_compute | 739.98 | 753.63 |
| other_gemm | 560.05 | 565.75 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4191.24 | 4256.69 |
| expert_dispatch_combine | 2958.21 | 3110.51 |
| expert_gemm_path | 1478.87 | 1478.87 |
| other_compute | 764.09 | 774.34 |
| other_gemm | 576.33 | 582.06 |

## Expert-GEMM hardware samples

Category-exclusive samples only. Excludes launch preparation and gaps. Device-wide counters are not per-kernel rooflines.

| Rank | Metric | Exclusive mean | Sample count | Exclusive fraction |
|---|---|---:|---:|---:|
| 0 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.42e+09 (not trustworthy as MHz) | 14261 | 100.0% |
| 0 | SMs Active [Throughput %] | 97.21 | 14261 | 100.0% |
| 0 | SM Issue [Throughput %] | 5.93 | 14261 | 100.0% |
| 0 | Tensor Active [Throughput %] | 91.75 | 14261 | 100.0% |
| 0 | DRAM Read Bandwidth [Throughput %] | 53.63 | 14261 | 100.0% |
| 0 | DRAM Write Bandwidth [Throughput %] | 5.15 | 14261 | 100.0% |
| 0 | NVLink RX Requests Protocol Data [Throughput %] | 0.45 | 14261 | 100.0% |
| 0 | NVLink RX Requests User Data [Throughput %] | 0.01 | 14261 | 100.0% |
| 0 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 14261 | 100.0% |
| 0 | NVLink RX Responses User Data [Throughput %] | 0.00 | 14261 | 100.0% |
| 0 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 14261 | 100.0% |
| 0 | NVLink TX Requests User Data [Throughput %] | 0.00 | 14261 | 100.0% |
| 0 | NVLink TX Responses Protocol Data [Throughput %] | 0.45 | 14261 | 100.0% |
| 0 | NVLink TX Responses User Data [Throughput %] | 0.45 | 14261 | 100.0% |
| 2 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.4e+09 (not trustworthy as MHz) | 14332 | 100.0% |
| 2 | SMs Active [Throughput %] | 98.43 | 14332 | 100.0% |
| 2 | SM Issue [Throughput %] | 5.92 | 14332 | 100.0% |
| 2 | Tensor Active [Throughput %] | 92.88 | 14332 | 100.0% |
| 2 | DRAM Read Bandwidth [Throughput %] | 52.68 | 14332 | 100.0% |
| 2 | DRAM Write Bandwidth [Throughput %] | 5.12 | 14332 | 100.0% |
| 2 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 14332 | 100.0% |
| 2 | NVLink RX Requests User Data [Throughput %] | 0.00 | 14332 | 100.0% |
| 2 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 14332 | 100.0% |
| 2 | NVLink RX Responses User Data [Throughput %] | 0.00 | 14332 | 100.0% |
| 2 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 14332 | 100.0% |
| 2 | NVLink TX Requests User Data [Throughput %] | 0.00 | 14332 | 100.0% |
| 2 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 14332 | 100.0% |
| 2 | NVLink TX Responses User Data [Throughput %] | 0.00 | 14332 | 100.0% |
| 3 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.44e+09 (not trustworthy as MHz) | 14822 | 100.0% |
| 3 | SMs Active [Throughput %] | 97.76 | 14822 | 100.0% |
| 3 | SM Issue [Throughput %] | 5.88 | 14822 | 100.0% |
| 3 | Tensor Active [Throughput %] | 90.66 | 14822 | 100.0% |
| 3 | DRAM Read Bandwidth [Throughput %] | 55.37 | 14822 | 100.0% |
| 3 | DRAM Write Bandwidth [Throughput %] | 5.19 | 14822 | 100.0% |
| 3 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 14822 | 100.0% |
| 3 | NVLink RX Requests User Data [Throughput %] | 0.00 | 14822 | 100.0% |
| 3 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 14822 | 100.0% |
| 3 | NVLink RX Responses User Data [Throughput %] | 0.00 | 14822 | 100.0% |
| 3 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 14822 | 100.0% |
| 3 | NVLink TX Requests User Data [Throughput %] | 0.00 | 14822 | 100.0% |
| 3 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 14822 | 100.0% |
| 3 | NVLink TX Responses User Data [Throughput %] | 0.00 | 14822 | 100.0% |
| 7 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.33e+09 (not trustworthy as MHz) | 14080 | 100.0% |
| 7 | SMs Active [Throughput %] | 98.41 | 14080 | 100.0% |
| 7 | SM Issue [Throughput %] | 6.00 | 14080 | 100.0% |
| 7 | Tensor Active [Throughput %] | 94.62 | 14080 | 100.0% |
| 7 | DRAM Read Bandwidth [Throughput %] | 50.44 | 14080 | 100.0% |
| 7 | DRAM Write Bandwidth [Throughput %] | 4.96 | 14080 | 100.0% |
| 7 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 14080 | 100.0% |
| 7 | NVLink RX Requests User Data [Throughput %] | 0.00 | 14080 | 100.0% |
| 7 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 14080 | 100.0% |
| 7 | NVLink RX Responses User Data [Throughput %] | 0.00 | 14080 | 100.0% |
| 7 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 14080 | 100.0% |
| 7 | NVLink TX Requests User Data [Throughput %] | 0.00 | 14080 | 100.0% |
| 7 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 14080 | 100.0% |
| 7 | NVLink TX Responses User Data [Throughput %] | 0.00 | 14080 | 100.0% |
| 4 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.43e+09 (not trustworthy as MHz) | 14078 | 100.0% |
| 4 | SMs Active [Throughput %] | 98.05 | 14078 | 100.0% |
| 4 | SM Issue [Throughput %] | 5.84 | 14078 | 100.0% |
| 4 | Tensor Active [Throughput %] | 91.63 | 14078 | 100.0% |
| 4 | DRAM Read Bandwidth [Throughput %] | 53.17 | 14078 | 100.0% |
| 4 | DRAM Write Bandwidth [Throughput %] | 5.18 | 14078 | 100.0% |
| 4 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 14078 | 100.0% |
| 4 | NVLink RX Requests User Data [Throughput %] | 0.00 | 14078 | 100.0% |
| 4 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 14078 | 100.0% |
| 4 | NVLink RX Responses User Data [Throughput %] | 0.00 | 14078 | 100.0% |
| 4 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 14078 | 100.0% |
| 4 | NVLink TX Requests User Data [Throughput %] | 0.00 | 14078 | 100.0% |
| 4 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 14078 | 100.0% |
| 4 | NVLink TX Responses User Data [Throughput %] | 0.00 | 14078 | 100.0% |
| 6 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.42e+09 (not trustworthy as MHz) | 16536 | 100.0% |
| 6 | SMs Active [Throughput %] | 98.69 | 16536 | 100.0% |
| 6 | SM Issue [Throughput %] | 5.76 | 16536 | 100.0% |
| 6 | Tensor Active [Throughput %] | 91.33 | 16536 | 100.0% |
| 6 | DRAM Read Bandwidth [Throughput %] | 56.25 | 16536 | 100.0% |
| 6 | DRAM Write Bandwidth [Throughput %] | 5.16 | 16536 | 100.0% |
| 6 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 16536 | 100.0% |
| 6 | NVLink RX Requests User Data [Throughput %] | 0.00 | 16536 | 100.0% |
| 6 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 16536 | 100.0% |
| 6 | NVLink RX Responses User Data [Throughput %] | 0.00 | 16536 | 100.0% |
| 6 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 16536 | 100.0% |
| 6 | NVLink TX Requests User Data [Throughput %] | 0.00 | 16536 | 100.0% |
| 6 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 16536 | 100.0% |
| 6 | NVLink TX Responses User Data [Throughput %] | 0.00 | 16536 | 100.0% |
| 5 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.42e+09 (not trustworthy as MHz) | 13717 | 100.0% |
| 5 | SMs Active [Throughput %] | 98.74 | 13717 | 100.0% |
| 5 | SM Issue [Throughput %] | 6.01 | 13717 | 100.0% |
| 5 | Tensor Active [Throughput %] | 92.42 | 13717 | 100.0% |
| 5 | DRAM Read Bandwidth [Throughput %] | 51.20 | 13717 | 100.0% |
| 5 | DRAM Write Bandwidth [Throughput %] | 5.24 | 13717 | 100.0% |
| 5 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 13717 | 100.0% |
| 5 | NVLink RX Requests User Data [Throughput %] | 0.00 | 13717 | 100.0% |
| 5 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 13717 | 100.0% |
| 5 | NVLink RX Responses User Data [Throughput %] | 0.00 | 13717 | 100.0% |
| 5 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 13717 | 100.0% |
| 5 | NVLink TX Requests User Data [Throughput %] | 0.00 | 13717 | 100.0% |
| 5 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 13717 | 100.0% |
| 5 | NVLink TX Responses User Data [Throughput %] | 0.00 | 13717 | 100.0% |
| 1 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.37e+09 (not trustworthy as MHz) | 14801 | 100.0% |
| 1 | SMs Active [Throughput %] | 97.87 | 14801 | 100.0% |
| 1 | SM Issue [Throughput %] | 6.00 | 14801 | 100.0% |
| 1 | Tensor Active [Throughput %] | 93.08 | 14801 | 100.0% |
| 1 | DRAM Read Bandwidth [Throughput %] | 53.43 | 14801 | 100.0% |
| 1 | DRAM Write Bandwidth [Throughput %] | 5.07 | 14801 | 100.0% |
| 1 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 14801 | 100.0% |
| 1 | NVLink RX Requests User Data [Throughput %] | 0.00 | 14801 | 100.0% |
| 1 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 14801 | 100.0% |
| 1 | NVLink RX Responses User Data [Throughput %] | 0.00 | 14801 | 100.0% |
| 1 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 14801 | 100.0% |
| 1 | NVLink TX Requests User Data [Throughput %] | 0.00 | 14801 | 100.0% |
| 1 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 14801 | 100.0% |
| 1 | NVLink TX Responses User Data [Throughput %] | 0.00 | 14801 | 100.0% |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
