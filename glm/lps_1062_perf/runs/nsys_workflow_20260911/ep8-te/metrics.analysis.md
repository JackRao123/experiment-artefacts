# Nsight runtime report

## Verdict

Ranked observed costs below are candidates for investigation, not guaranteed speedups.

## Capture quality

- Input: `/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/nsys_workflow_20260911/ep8-te/metrics.sqlite`
- Ranks: 0, 1, 5, 6, 3, 4, 7, 2; explicit NVTX rank/forward-backward anchors.
- Rank coverage: {'captured': 8, 'world_size': 8}; observed CUDA graph launch calls: 0.
- Same-run capture slowdown: 1.9323538153886544%.
- Hardware metrics present: True; this timing report does not infer saturation.
- Warning/error records: 116; see diagnostics below before trusting event completeness.
- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.
- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.

## Per-step budget

All times ms, means across captured forward/backward requests. Optimizer excluded.

| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 10516.50 | 10257.19 | 259.31 | 3536.82 | 311.31 | 3339.71 | 3262.93 | 3017.55 |
| 1 | 1 | 10507.28 | 10249.08 | 258.20 | 3325.95 | 267.14 | 3128.46 | 3053.35 | 2858.80 |
| 5 | 1 | 10505.19 | 10244.67 | 260.53 | 3407.76 | 249.10 | 3207.10 | 3133.13 | 2954.92 |
| 6 | 1 | 10508.86 | 10251.39 | 257.47 | 3143.14 | 259.72 | 2951.71 | 2871.64 | 2675.28 |
| 3 | 1 | 10507.70 | 10246.72 | 260.98 | 3380.17 | 280.14 | 3184.70 | 3104.60 | 2887.22 |
| 4 | 1 | 10502.31 | 10245.72 | 256.59 | 3437.93 | 242.98 | 3242.46 | 3167.29 | 2988.09 |
| 7 | 1 | 10505.90 | 10246.66 | 259.24 | 3486.16 | 279.77 | 3294.04 | 3212.99 | 2998.55 |
| 2 | 1 | 10507.36 | 10247.74 | 259.62 | 3456.07 | 275.66 | 3262.05 | 3181.90 | 2966.91 |

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
| 0 | 0 | 0 | unknown | no attributed expert-weight gathers; check group annotation coverage |
| 1 | 0 | 0 | unknown | no attributed expert-weight gathers; check group annotation coverage |
| 5 | 0 | 0 | unknown | no attributed expert-weight gathers; check group annotation coverage |
| 6 | 0 | 0 | unknown | no attributed expert-weight gathers; check group annotation coverage |
| 3 | 0 | 0 | unknown | no attributed expert-weight gathers; check group annotation coverage |
| 4 | 0 | 0 | unknown | no attributed expert-weight gathers; check group annotation coverage |
| 7 | 0 | 0 | unknown | no attributed expert-weight gathers; check group annotation coverage |
| 2 | 0 | 0 | unknown | no attributed expert-weight gathers; check group annotation coverage |

## Ranked observed costs

Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate.

### Rank 0

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 3995.63 | 4056.53 |
| expert_dispatch_combine | 2767.72 | 2917.25 |
| expert_gemm_path | 1286.59 | 1286.59 |
| other_compute | 706.48 | 719.40 |
| other_gemm | 570.00 | 575.53 |

### Rank 1

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4137.10 | 4200.23 |
| expert_dispatch_combine | 2679.76 | 2830.30 |
| expert_gemm_path | 1307.94 | 1307.94 |
| other_compute | 751.45 | 760.44 |
| other_gemm | 569.44 | 575.13 |

### Rank 5

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4150.06 | 4209.98 |
| expert_dispatch_combine | 2734.53 | 2887.54 |
| expert_gemm_path | 1236.18 | 1236.18 |
| other_compute | 738.47 | 749.92 |
| other_gemm | 563.40 | 568.80 |

### Rank 6

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4139.31 | 4202.35 |
| expert_dispatch_combine | 2489.41 | 2641.99 |
| expert_gemm_path | 1465.64 | 1465.64 |
| other_compute | 768.10 | 780.75 |
| other_gemm | 562.65 | 569.70 |

### Rank 3

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4106.82 | 4169.65 |
| expert_dispatch_combine | 2695.76 | 2848.29 |
| expert_gemm_path | 1321.20 | 1321.20 |
| other_compute | 746.71 | 759.78 |
| other_gemm | 558.64 | 565.61 |

### Rank 4

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4148.82 | 4207.05 |
| expert_dispatch_combine | 2762.65 | 2915.57 |
| expert_gemm_path | 1243.98 | 1243.98 |
| other_compute | 739.85 | 752.99 |
| other_gemm | 554.93 | 561.48 |

### Rank 7

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4114.46 | 4177.83 |
| expert_dispatch_combine | 2793.19 | 2944.79 |
| expert_gemm_path | 1204.24 | 1204.24 |
| other_compute | 731.66 | 744.62 |
| other_gemm | 554.64 | 562.00 |

### Rank 2

Runtime correlation: 100.00%.

| Category | Exclusive ms | Union ms |
|---|---:|---:|
| attention | 4095.49 | 4158.90 |
| expert_dispatch_combine | 2760.45 | 2912.17 |
| expert_gemm_path | 1248.39 | 1248.39 |
| other_compute | 738.51 | 751.63 |
| other_gemm | 551.98 | 558.44 |

## Expert-GEMM hardware samples

Category-exclusive samples only. Excludes launch preparation and gaps. Device-wide counters are not per-kernel rooflines.

| Rank | Metric | Exclusive mean | Sample count | Exclusive fraction |
|---|---|---:|---:|---:|
| 0 | GPC Clock Frequency [MHz] | 1535289111.39 | 12860 | 100.0% |
| 0 | SMs Active [Throughput %] | 98.64 | 12860 | 100.0% |
| 0 | SM Issue [Throughput %] | 4.89 | 12860 | 100.0% |
| 0 | Tensor Active [Throughput %] | 91.71 | 12860 | 100.0% |
| 0 | DRAM Read Bandwidth [Throughput %] | 18.74 | 12860 | 100.0% |
| 0 | DRAM Write Bandwidth [Throughput %] | 5.69 | 12860 | 100.0% |
| 0 | NVLink RX Requests Protocol Data [Throughput %] | 0.45 | 12860 | 100.0% |
| 0 | NVLink RX Requests User Data [Throughput %] | 0.01 | 12860 | 100.0% |
| 0 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12860 | 100.0% |
| 0 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12860 | 100.0% |
| 0 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12860 | 100.0% |
| 0 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12860 | 100.0% |
| 0 | NVLink TX Responses Protocol Data [Throughput %] | 0.45 | 12860 | 100.0% |
| 0 | NVLink TX Responses User Data [Throughput %] | 0.45 | 12860 | 100.0% |
| 1 | GPC Clock Frequency [MHz] | 1544632010.00 | 13074 | 100.0% |
| 1 | SMs Active [Throughput %] | 98.65 | 13074 | 100.0% |
| 1 | SM Issue [Throughput %] | 4.94 | 13074 | 100.0% |
| 1 | Tensor Active [Throughput %] | 91.78 | 13074 | 100.0% |
| 1 | DRAM Read Bandwidth [Throughput %] | 18.38 | 13074 | 100.0% |
| 1 | DRAM Write Bandwidth [Throughput %] | 5.74 | 13074 | 100.0% |
| 1 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 13074 | 100.0% |
| 1 | NVLink RX Requests User Data [Throughput %] | 0.00 | 13074 | 100.0% |
| 1 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 13074 | 100.0% |
| 1 | NVLink RX Responses User Data [Throughput %] | 0.00 | 13074 | 100.0% |
| 1 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 13074 | 100.0% |
| 1 | NVLink TX Requests User Data [Throughput %] | 0.00 | 13074 | 100.0% |
| 1 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 13074 | 100.0% |
| 1 | NVLink TX Responses User Data [Throughput %] | 0.00 | 13074 | 100.0% |
| 5 | GPC Clock Frequency [MHz] | 1566533884.99 | 12362 | 100.0% |
| 5 | SMs Active [Throughput %] | 98.60 | 12362 | 100.0% |
| 5 | SM Issue [Throughput %] | 5.02 | 12362 | 100.0% |
| 5 | Tensor Active [Throughput %] | 91.30 | 12362 | 100.0% |
| 5 | DRAM Read Bandwidth [Throughput %] | 18.88 | 12362 | 100.0% |
| 5 | DRAM Write Bandwidth [Throughput %] | 5.79 | 12362 | 100.0% |
| 5 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 12362 | 100.0% |
| 5 | NVLink RX Requests User Data [Throughput %] | 0.00 | 12362 | 100.0% |
| 5 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12362 | 100.0% |
| 5 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12362 | 100.0% |
| 5 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12362 | 100.0% |
| 5 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12362 | 100.0% |
| 5 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12362 | 100.0% |
| 5 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12362 | 100.0% |
| 6 | GPC Clock Frequency [MHz] | 1553841175.41 | 14647 | 100.0% |
| 6 | SMs Active [Throughput %] | 98.81 | 14647 | 100.0% |
| 6 | SM Issue [Throughput %] | 4.81 | 14647 | 100.0% |
| 6 | Tensor Active [Throughput %] | 92.62 | 14647 | 100.0% |
| 6 | DRAM Read Bandwidth [Throughput %] | 18.01 | 14647 | 100.0% |
| 6 | DRAM Write Bandwidth [Throughput %] | 5.82 | 14647 | 100.0% |
| 6 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 14647 | 100.0% |
| 6 | NVLink RX Requests User Data [Throughput %] | 0.00 | 14647 | 100.0% |
| 6 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 14647 | 100.0% |
| 6 | NVLink RX Responses User Data [Throughput %] | 0.00 | 14647 | 100.0% |
| 6 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 14647 | 100.0% |
| 6 | NVLink TX Requests User Data [Throughput %] | 0.00 | 14647 | 100.0% |
| 6 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 14647 | 100.0% |
| 6 | NVLink TX Responses User Data [Throughput %] | 0.00 | 14647 | 100.0% |
| 3 | GPC Clock Frequency [MHz] | 1561530460.69 | 13203 | 100.0% |
| 3 | SMs Active [Throughput %] | 98.65 | 13203 | 100.0% |
| 3 | SM Issue [Throughput %] | 4.95 | 13203 | 100.0% |
| 3 | Tensor Active [Throughput %] | 91.83 | 13203 | 100.0% |
| 3 | DRAM Read Bandwidth [Throughput %] | 18.60 | 13203 | 100.0% |
| 3 | DRAM Write Bandwidth [Throughput %] | 5.82 | 13203 | 100.0% |
| 3 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 13203 | 100.0% |
| 3 | NVLink RX Requests User Data [Throughput %] | 0.00 | 13203 | 100.0% |
| 3 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 13203 | 100.0% |
| 3 | NVLink RX Responses User Data [Throughput %] | 0.00 | 13203 | 100.0% |
| 3 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 13203 | 100.0% |
| 3 | NVLink TX Requests User Data [Throughput %] | 0.00 | 13203 | 100.0% |
| 3 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 13203 | 100.0% |
| 3 | NVLink TX Responses User Data [Throughput %] | 0.00 | 13203 | 100.0% |
| 4 | GPC Clock Frequency [MHz] | 1582568785.49 | 12432 | 100.0% |
| 4 | SMs Active [Throughput %] | 98.60 | 12432 | 100.0% |
| 4 | SM Issue [Throughput %] | 5.08 | 12432 | 100.0% |
| 4 | Tensor Active [Throughput %] | 91.40 | 12432 | 100.0% |
| 4 | DRAM Read Bandwidth [Throughput %] | 19.36 | 12432 | 100.0% |
| 4 | DRAM Write Bandwidth [Throughput %] | 5.90 | 12432 | 100.0% |
| 4 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 12432 | 100.0% |
| 4 | NVLink RX Requests User Data [Throughput %] | 0.00 | 12432 | 100.0% |
| 4 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12432 | 100.0% |
| 4 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12432 | 100.0% |
| 4 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12432 | 100.0% |
| 4 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12432 | 100.0% |
| 4 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12432 | 100.0% |
| 4 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12432 | 100.0% |
| 7 | GPC Clock Frequency [MHz] | 1583167703.22 | 12056 | 100.0% |
| 7 | SMs Active [Throughput %] | 98.58 | 12056 | 100.0% |
| 7 | SM Issue [Throughput %] | 5.15 | 12056 | 100.0% |
| 7 | Tensor Active [Throughput %] | 90.93 | 12056 | 100.0% |
| 7 | DRAM Read Bandwidth [Throughput %] | 19.33 | 12056 | 100.0% |
| 7 | DRAM Write Bandwidth [Throughput %] | 5.84 | 12056 | 100.0% |
| 7 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 12056 | 100.0% |
| 7 | NVLink RX Requests User Data [Throughput %] | 0.00 | 12056 | 100.0% |
| 7 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12056 | 100.0% |
| 7 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12056 | 100.0% |
| 7 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12056 | 100.0% |
| 7 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12056 | 100.0% |
| 7 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12056 | 100.0% |
| 7 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12056 | 100.0% |
| 2 | GPC Clock Frequency [MHz] | 1591144835.53 | 12472 | 100.0% |
| 2 | SMs Active [Throughput %] | 98.65 | 12472 | 100.0% |
| 2 | SM Issue [Throughput %] | 4.99 | 12472 | 100.0% |
| 2 | Tensor Active [Throughput %] | 91.46 | 12472 | 100.0% |
| 2 | DRAM Read Bandwidth [Throughput %] | 19.24 | 12472 | 100.0% |
| 2 | DRAM Write Bandwidth [Throughput %] | 5.88 | 12472 | 100.0% |
| 2 | NVLink RX Requests Protocol Data [Throughput %] | 0.00 | 12472 | 100.0% |
| 2 | NVLink RX Requests User Data [Throughput %] | 0.00 | 12472 | 100.0% |
| 2 | NVLink RX Responses Protocol Data [Throughput %] | 0.00 | 12472 | 100.0% |
| 2 | NVLink RX Responses User Data [Throughput %] | 0.00 | 12472 | 100.0% |
| 2 | NVLink TX Requests Protocol Data [Throughput %] | 0.00 | 12472 | 100.0% |
| 2 | NVLink TX Requests User Data [Throughput %] | 0.00 | 12472 | 100.0% |
| 2 | NVLink TX Responses Protocol Data [Throughput %] | 0.00 | 12472 | 100.0% |
| 2 | NVLink TX Responses User Data [Throughput %] | 0.00 | 12472 | 100.0% |

## Claim status

- Interval budgets: trace-measured; overlapping category unions do not add to step time.
- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.
- Dispatch/combine includes packing and synchronization, not just network transfer.
- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.
- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.
- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.
- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON.
