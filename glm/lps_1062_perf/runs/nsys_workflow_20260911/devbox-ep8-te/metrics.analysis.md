# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `devbox-ep8-te/metrics.sqlite`
- Ranks: 0, 1, 4, 5, 7, 2, 3, 6; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: 1.107204497560721%.
- Hardware metrics present: True; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 10461.67 | 10219.40 | 242.27 | 3585.09 | 334.05 | 3411.46 | 3328.33 | 3054.46 |
| 1 | 1 | 10457.62 | 10208.08 | 249.54 | 3313.34 | 264.70 | 3126.72 | 3049.34 | 2851.25 |
| 4 | 1 | 10459.46 | 10202.25 | 257.21 | 3358.38 | 252.96 | 3161.11 | 3087.07 | 2897.90 |
| 5 | 1 | 10457.97 | 10200.23 | 257.74 | 3375.27 | 264.98 | 3179.31 | 3103.52 | 2906.17 |
| 7 | 1 | 10456.38 | 10205.49 | 250.89 | 3304.31 | 264.56 | 3115.15 | 3039.19 | 2845.76 |
| 2 | 1 | 10454.16 | 10207.17 | 247.00 | 3383.25 | 284.20 | 3198.58 | 3121.91 | 2901.08 |
| 3 | 1 | 10461.95 | 10198.38 | 263.57 | 3411.91 | 311.14 | 3213.01 | 3133.83 | 2890.89 |
| 6 | 1 | 10455.20 | 10194.26 | 260.94 | 3136.35 | 302.27 | 2938.23 | 2861.33 | 2628.41 |

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
| 0 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 1 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 4 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 5 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 7 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 2 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 3 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |
| 6 | 0 | 0 | unknown | not applicable: expert-DP size one, no expert-weight gather expected |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3960.80 | 4029.86 |
| expert_dispatch_combine | 2805.07 | 2957.18 |
| expert_gemm_path | 1256.58 | 1256.58 |
| other_compute | 704.01 | 715.93 |
| other_gemm | 560.17 | 566.01 |

### Rank 1

Runtime correlation: 99.99%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4133.91 | 4198.33 |
| expert_dispatch_combine | 2666.40 | 2818.50 |
| expert_gemm_path | 1307.75 | 1307.75 |
| other_compute | 748.60 | 759.58 |
| other_gemm | 571.87 | 577.62 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4143.39 | 4204.12 |
| expert_dispatch_combine | 2684.97 | 2836.70 |
| expert_gemm_path | 1248.89 | 1248.89 |
| other_compute | 740.99 | 752.35 |
| other_gemm | 557.22 | 563.05 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4141.97 | 4203.73 |
| expert_dispatch_combine | 2702.76 | 2855.09 |
| expert_gemm_path | 1232.67 | 1232.67 |
| other_compute | 737.23 | 749.40 |
| other_gemm | 558.60 | 564.35 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4160.38 | 4224.26 |
| expert_dispatch_combine | 2671.37 | 2824.07 |
| expert_gemm_path | 1263.95 | 1263.95 |
| other_compute | 738.90 | 749.76 |
| other_gemm | 577.15 | 582.29 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4104.49 | 4169.02 |
| expert_dispatch_combine | 2721.09 | 2871.81 |
| expert_gemm_path | 1281.42 | 1281.42 |
| other_compute | 708.24 | 718.52 |
| other_gemm | 562.49 | 568.32 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4098.54 | 4163.41 |
| expert_dispatch_combine | 2725.63 | 2875.80 |
| expert_gemm_path | 1296.56 | 1296.56 |
| other_compute | 707.63 | 719.80 |
| other_gemm | 554.07 | 559.86 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4122.57 | 4185.01 |
| expert_dispatch_combine | 2478.02 | 2628.96 |
| expert_gemm_path | 1448.57 | 1448.57 |
| other_compute | 763.92 | 776.54 |
| other_gemm | 557.00 | 562.77 |

## Expert-GEMM hardware samples

Category-exclusive samples only. Excludes launch preparation and gaps. Device-wide counters are not per-kernel rooflines.

| Rank | Metric | Exclusive mean | Sample count | Exclusive fraction |
|---|---|---:|---:|---:|
| 0 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.57e+09 (not trustworthy as MHz) | 12564 | 100.0% |
| 0 | SMs Active [Throughput %] | 98.63 | 12564 | 100.0% |
| 0 | SM Issue [Throughput %] | 5.08 | 12564 | 100.0% |
| 0 | Tensor Active [Throughput %] | 91.37 | 12564 | 100.0% |
| 0 | DRAM Read Bandwidth [Throughput %] | 18.94 | 12564 | 100.0% |
| 0 | DRAM Write Bandwidth [Throughput %] | 5.86 | 12564 | 100.0% |
| 0 | NVLink RX Requests Protocol Data [Throughput %] | 0.42 | 12564 | 100.0% |
| 0 | NVLink RX Requests User Data [Throughput %] | 0.01 | 12564 | 100.0% |
| 0 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12564 | 100.0% |
| 0 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12564 | 100.0% |
| 0 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12564 | 100.0% |
| 0 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12564 | 100.0% |
| 0 | NVLink TX Responses Protocol Data [Throughput %] | 0.42 | 12564 | 100.0% |
| 0 | NVLink TX Responses User Data [Throughput %] | 0.42 | 12564 | 100.0% |
| 1 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.53e+09 (not trustworthy as MHz) | 13083 | 100.0% |
| 1 | SMs Active [Throughput %] | 98.63 | 13083 | 100.0% |
| 1 | SM Issue [Throughput %] | 5.08 | 13083 | 100.0% |
| 1 | Tensor Active [Throughput %] | 91.60 | 13083 | 100.0% |
| 1 | DRAM Read Bandwidth [Throughput %] | 18.46 | 13083 | 100.0% |
| 1 | DRAM Write Bandwidth [Throughput %] | 5.75 | 13083 | 100.0% |
| 1 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 13083 | 100.0% |
| 1 | NVLink RX Requests User Data [Throughput %] | 0.00 | 13083 | 100.0% |
| 1 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 13083 | 100.0% |
| 1 | NVLink RX Responses User Data [Throughput %] | 0.00 | 13083 | 100.0% |
| 1 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 13083 | 100.0% |
| 1 | NVLink TX Requests User Data [Throughput %] | 0.00 | 13083 | 100.0% |
| 1 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 13083 | 100.0% |
| 1 | NVLink TX Responses User Data [Throughput %] | 0.00 | 13083 | 100.0% |
| 4 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.58e+09 (not trustworthy as MHz) | 12485 | 100.0% |
| 4 | SMs Active [Throughput %] | 98.68 | 12485 | 100.0% |
| 4 | SM Issue [Throughput %] | 5.06 | 12485 | 100.0% |
| 4 | Tensor Active [Throughput %] | 91.29 | 12485 | 100.0% |
| 4 | DRAM Read Bandwidth [Throughput %] | 18.89 | 12485 | 100.0% |
| 4 | DRAM Write Bandwidth [Throughput %] | 5.85 | 12485 | 100.0% |
| 4 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 12485 | 100.0% |
| 4 | NVLink RX Requests User Data [Throughput %] | 0.00 | 12485 | 100.0% |
| 4 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12485 | 100.0% |
| 4 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12485 | 100.0% |
| 4 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12485 | 100.0% |
| 4 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12485 | 100.0% |
| 4 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12485 | 100.0% |
| 4 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12485 | 100.0% |
| 5 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.57e+09 (not trustworthy as MHz) | 12330 | 100.0% |
| 5 | SMs Active [Throughput %] | 98.61 | 12330 | 100.0% |
| 5 | SM Issue [Throughput %] | 5.07 | 12330 | 100.0% |
| 5 | Tensor Active [Throughput %] | 91.18 | 12330 | 100.0% |
| 5 | DRAM Read Bandwidth [Throughput %] | 19.21 | 12330 | 100.0% |
| 5 | DRAM Write Bandwidth [Throughput %] | 5.82 | 12330 | 100.0% |
| 5 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 12330 | 100.0% |
| 5 | NVLink RX Requests User Data [Throughput %] | 0.00 | 12330 | 100.0% |
| 5 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12330 | 100.0% |
| 5 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12330 | 100.0% |
| 5 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12330 | 100.0% |
| 5 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12330 | 100.0% |
| 5 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12330 | 100.0% |
| 5 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12330 | 100.0% |
| 7 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.5e+09 (not trustworthy as MHz) | 12633 | 100.0% |
| 7 | SMs Active [Throughput %] | 98.67 | 12633 | 100.0% |
| 7 | SM Issue [Throughput %] | 5.03 | 12633 | 100.0% |
| 7 | Tensor Active [Throughput %] | 91.50 | 12633 | 100.0% |
| 7 | DRAM Read Bandwidth [Throughput %] | 18.58 | 12633 | 100.0% |
| 7 | DRAM Write Bandwidth [Throughput %] | 5.56 | 12633 | 100.0% |
| 7 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 12633 | 100.0% |
| 7 | NVLink RX Requests User Data [Throughput %] | 0.00 | 12633 | 100.0% |
| 7 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12633 | 100.0% |
| 7 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12633 | 100.0% |
| 7 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12633 | 100.0% |
| 7 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12633 | 100.0% |
| 7 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12633 | 100.0% |
| 7 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12633 | 100.0% |
| 2 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.55e+09 (not trustworthy as MHz) | 12820 | 100.0% |
| 2 | SMs Active [Throughput %] | 98.71 | 12820 | 100.0% |
| 2 | SM Issue [Throughput %] | 5.03 | 12820 | 100.0% |
| 2 | Tensor Active [Throughput %] | 91.63 | 12820 | 100.0% |
| 2 | DRAM Read Bandwidth [Throughput %] | 18.84 | 12820 | 100.0% |
| 2 | DRAM Write Bandwidth [Throughput %] | 5.78 | 12820 | 100.0% |
| 2 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 12820 | 100.0% |
| 2 | NVLink RX Requests User Data [Throughput %] | 0.00 | 12820 | 100.0% |
| 2 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12820 | 100.0% |
| 2 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12820 | 100.0% |
| 2 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12820 | 100.0% |
| 2 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12820 | 100.0% |
| 2 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12820 | 100.0% |
| 2 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12820 | 100.0% |
| 3 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.59e+09 (not trustworthy as MHz) | 12977 | 100.0% |
| 3 | SMs Active [Throughput %] | 98.60 | 12977 | 100.0% |
| 3 | SM Issue [Throughput %] | 5.03 | 12977 | 100.0% |
| 3 | Tensor Active [Throughput %] | 91.60 | 12977 | 100.0% |
| 3 | DRAM Read Bandwidth [Throughput %] | 18.96 | 12977 | 100.0% |
| 3 | DRAM Write Bandwidth [Throughput %] | 5.92 | 12977 | 100.0% |
| 3 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 12977 | 100.0% |
| 3 | NVLink RX Requests User Data [Throughput %] | 0.00 | 12977 | 100.0% |
| 3 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12977 | 100.0% |
| 3 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12977 | 100.0% |
| 3 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12977 | 100.0% |
| 3 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12977 | 100.0% |
| 3 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12977 | 100.0% |
| 3 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12977 | 100.0% |
| 6 | GPC Clock Frequency [MHz] | unit mismatch: raw 1.57e+09 (not trustworthy as MHz) | 14504 | 100.0% |
| 6 | SMs Active [Throughput %] | 98.83 | 14504 | 100.0% |
| 6 | SM Issue [Throughput %] | 4.88 | 14504 | 100.0% |
| 6 | Tensor Active [Throughput %] | 92.43 | 14504 | 100.0% |
| 6 | DRAM Read Bandwidth [Throughput %] | 18.38 | 14504 | 100.0% |
| 6 | DRAM Write Bandwidth [Throughput %] | 5.89 | 14504 | 100.0% |
| 6 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 14504 | 100.0% |
| 6 | NVLink RX Requests User Data [Throughput %] | 0.00 | 14504 | 100.0% |
| 6 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 14504 | 100.0% |
| 6 | NVLink RX Responses User Data [Throughput %] | 0.00 | 14504 | 100.0% |
| 6 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 14504 | 100.0% |
| 6 | NVLink TX Requests User Data [Throughput %] | 0.00 | 14504 | 100.0% |
| 6 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 14504 | 100.0% |
| 6 | NVLink TX Responses User Data [Throughput %] | 0.00 | 14504 | 100.0% |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
