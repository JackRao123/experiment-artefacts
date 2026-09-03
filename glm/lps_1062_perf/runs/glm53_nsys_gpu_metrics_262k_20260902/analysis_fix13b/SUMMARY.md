# Attribution summary

sqlite: `fix13b.sqlite`

## Step window per GPU (s)

gpu0: 24.812, gpu1: 24.794, gpu2: 24.795, gpu3: 24.790, gpu4: 24.797, gpu5: 24.797, gpu6: 24.794, gpu7: 24.796

## Kernel time by category (mean per GPU, s)

| category | s/GPU | % of window |
|---|---|---|
| gemm | 5.010 | 20.2 |
| hybridep_sync | 4.384 | 17.7 |
| dsa_backward | 3.709 | 15.0 |
| cat_copy | 2.931 | 11.8 |
| elementwise | 2.174 | 8.8 |
| dsa_forward | 1.238 | 5.0 |
| dsa_indexer | 1.095 | 4.4 |
| hybridep_dispatch | 0.888 | 3.6 |
| hybridep_combine | 0.869 | 3.5 |
| moe_permute | 0.780 | 3.1 |
| nccl | 0.627 | 2.5 |
| activation | 0.374 | 1.5 |
| norm | 0.305 | 1.2 |
| topk_router | 0.144 | 0.6 |
| hybridep_meta | 0.085 | 0.3 |
| cub_scan_sort | 0.076 | 0.3 |
| other | 0.014 | 0.1 |
| cross_entropy | 0.006 | 0.0 |
| nonzero_cub | 0.001 | 0.0 |

## GPU idle (no kernel resident) inside window

| gpu | idle s | idle % | gaps >1ms | max gap ms |
|---|---|---|---|---|
| 0 | 0.331 | 1.33 | 5 | 5.881 |
| 1 | 0.615 | 2.48 | 3 | 258.317 |
| 2 | 0.364 | 1.47 | 5 | 15.138 |
| 3 | 0.374 | 1.51 | 7 | 15.471 |
| 4 | 0.34 | 1.37 | 4 | 3.461 |
| 5 | 0.342 | 1.38 | 1 | 3.514 |
| 6 | 1.021 | 4.12 | 3 | 673.925 |
| 7 | 0.326 | 1.31 | 3 | 3.422 |

## NVTX phases (wall s on host thread / GPU kernel s attributed by launch time), mean over GPUs

| range | instances/GPU | wall s | gpu kernel s |
|---|---|---|---|
| forward | 1 | 7.352 | 7.314 |
| backward | 1 | 17.365 | 17.393 |
| optimizer | 1 | 0.000 | 0.000 |
| attention | 156 | 3.232 | 3.116 |
| moe | 150 | 11.706 | 8.347 |
| hybridep_dispatch | 150 | 0.163 | 1.058 |
| hybridep_combine | 150 | 0.022 | 3.767 |
| recompute | 78 | 17.250 | 17.270 |
| lm_head | 1 | 0.065 | 0.070 |
| dsa_indexer | 42 | 2.804 | 0.111 |
| dsa_core | 156 | 0.063 | 1.525 |
| layer_forward | 78 | 7.211 | 7.242 |
| layer_recompute_forward | 78 | 8.616 | 6.593 |
| backward_minus_recompute_forward | 1 | nan | 10.800 |

## HybridEP sync waits >= 5.0 ms: who is everyone waiting for?

long waits total (all GPUs): 33.31 s of 35.07 s total sync time

| group | key | wait s (all GPUs) |
|---|---|---|
| by_laggard | gpu0 | 3.349 |
| by_laggard | gpu1 | 6.558 |
| by_laggard | gpu2 | 1.991 |
| by_laggard | gpu3 | 2.645 |
| by_laggard | gpu4 | 4.097 |
| by_laggard | gpu5 | 2.107 |
| by_laggard | gpu6 | 10.744 |
| by_laggard | gpu7 | 1.815 |
| by_kind | laggard_gpu_busy | 26.373 |
| by_kind | laggard_gpu_idle(host) | 6.923 |
| by_kind | mixed | 0.009 |
| by_laggard_top_category | gemm | 31.574 |
| by_laggard_top_category | elementwise | 1.688 |
| by_laggard_top_category | moe_permute | 0.043 |
| total | long_waits>=5.0ms_all_gpus_s | 33.305 |
| total | all_sync_all_gpus_s | 35.073 |

## Host-side blocking API calls inside window (per GPU)

| gpu | api | calls | total s | max ms |
|---|---|---|---|---|
| 0 | cudaStreamSynchronize | 533 | 20.609 | 782.133 |
| 1 | cudaStreamSynchronize | 531 | 19.961 | 782.164 |
| 2 | cudaStreamSynchronize | 531 | 20.038 | 781.368 |
| 3 | cudaStreamSynchronize | 531 | 20.057 | 782.478 |
| 4 | cudaStreamSynchronize | 531 | 20.403 | 782.734 |
| 5 | cudaStreamSynchronize | 531 | 20.270 | 781.553 |
| 6 | cudaStreamSynchronize | 531 | 19.577 | 376.862 |
| 7 | cudaStreamSynchronize | 531 | 20.489 | 780.673 |

## Top call chains for blocking syncs

- 1 calls, 0.000 s: `0x7f0b1dcc37db < 0x7f0b1dcc39ab < 0x7f0b1dcc0f05 < 0x7f0b1dcc10ff < 0x7f0bd90ae536 < cudaStreamSynchronize < at::native::_local_scalar_dense_cuda(at::Tensor const&)::{la < at::native::_local_scalar_dense_cuda(at::Tensor const&) < c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::W < at::_ops::_local_scalar_dense::redispatch(c10::DispatchKeySe < c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::W < at::_ops::_local_scalar_dense::call(at::Tensor const&)`

## Top 'other' kernels (s per GPU)

| kernel | instances (all GPUs) | s/GPU | max ms |
|---|---|---|---|
| `scan` | 1200 | 0.012 | 0.09 |
| `searchsorted_cuda_kernel` | 4416 | 0.001 | 0.003 |
| `compute_cuda_kernel` | 151 | 0.0 | 0.03 |
| `indexing_backward_kernel` | 1 | 0.0 | 0.06 |
| `ln_fwd_general_kernel` | 336 | 0.0 | 0.011 |
| `pad_tokens_per_expert_kernel` | 1200 | 0.0 | 0.004 |
| `thd_partition_indices_kernel` | 96 | 0.0 | 0.002 |
| `triton_poi_fused_all_reduce_lift_fresh_0` | 1200 | 0.0 | 0.002 |
| `write_indices` | 5 | 0.0 | 0.003 |

## GPU hardware metrics by phase (mean over GPUs)

| phase | metric | avg |
|---|---|---|
| step | GR Active [Throughput %] | 98.54 |
| step | SMs Active [Throughput %] | 69.88 |
| step | SM Issue [Throughput %] | 20.20 |
| step | Tensor Active [Throughput %] | 25.17 |
| step | DRAM Read Bandwidth [Throughput %] | 11.42 |
| step | DRAM Write Bandwidth [Throughput %] | 7.88 |
| step | NVLink RX Requests Protocol Data [Throughput %] | 0.69 |
| step | NVLink RX Requests User Data [Throughput %] | 1.90 |
| step | NVLink RX Responses Protocol Data [Throughput %] | 0.40 |
| step | NVLink RX Responses User Data [Throughput %] | 1.75 |
| step | NVLink TX Requests Protocol Data [Throughput %] | 0.81 |
| step | NVLink TX Requests User Data [Throughput %] | 1.89 |
| step | NVLink TX Responses Protocol Data [Throughput %] | 0.29 |
| step | NVLink TX Responses User Data [Throughput %] | 1.64 |
| forward | GR Active [Throughput %] | 98.46 |
| forward | SMs Active [Throughput %] | 69.09 |
| forward | SM Issue [Throughput %] | 24.32 |
| forward | Tensor Active [Throughput %] | 27.24 |
| forward | DRAM Read Bandwidth [Throughput %] | 11.96 |
| forward | DRAM Write Bandwidth [Throughput %] | 8.73 |
| forward | NVLink RX Requests Protocol Data [Throughput %] | 0.78 |
| forward | NVLink RX Requests User Data [Throughput %] | 2.21 |
| forward | NVLink RX Responses Protocol Data [Throughput %] | 0.40 |
| forward | NVLink RX Responses User Data [Throughput %] | 1.88 |
| forward | NVLink TX Requests Protocol Data [Throughput %] | 0.87 |
| forward | NVLink TX Requests User Data [Throughput %] | 2.20 |
| forward | NVLink TX Responses Protocol Data [Throughput %] | 0.31 |
| forward | NVLink TX Responses User Data [Throughput %] | 1.78 |
| backward | GR Active [Throughput %] | 98.58 |
| backward | SMs Active [Throughput %] | 70.15 |
| backward | SM Issue [Throughput %] | 18.45 |
| backward | Tensor Active [Throughput %] | 24.32 |
| backward | DRAM Read Bandwidth [Throughput %] | 11.21 |
| backward | DRAM Write Bandwidth [Throughput %] | 7.54 |
| backward | NVLink RX Requests Protocol Data [Throughput %] | 0.66 |
| backward | NVLink RX Requests User Data [Throughput %] | 1.77 |
| backward | NVLink RX Responses Protocol Data [Throughput %] | 0.40 |
| backward | NVLink RX Responses User Data [Throughput %] | 1.70 |
| backward | NVLink TX Requests Protocol Data [Throughput %] | 0.79 |
| backward | NVLink TX Requests User Data [Throughput %] | 1.77 |
| backward | NVLink TX Responses Protocol Data [Throughput %] | 0.28 |
| backward | NVLink TX Responses User Data [Throughput %] | 1.58 |
| optimizer | GR Active [Throughput %] | 27.76 |
| optimizer | SMs Active [Throughput %] | 10.56 |
| optimizer | SM Issue [Throughput %] | 1.28 |
| optimizer | Tensor Active [Throughput %] | 0.00 |
| optimizer | DRAM Read Bandwidth [Throughput %] | 1.93 |
| optimizer | DRAM Write Bandwidth [Throughput %] | 1.60 |
| optimizer | NVLink RX Requests Protocol Data [Throughput %] | 0.01 |
| optimizer | NVLink RX Requests User Data [Throughput %] | 0.01 |
| optimizer | NVLink RX Responses Protocol Data [Throughput %] | 0.00 |
| optimizer | NVLink RX Responses User Data [Throughput %] | 0.00 |
| optimizer | NVLink TX Requests Protocol Data [Throughput %] | 0.01 |
| optimizer | NVLink TX Requests User Data [Throughput %] | 0.01 |
| optimizer | NVLink TX Responses Protocol Data [Throughput %] | 0.00 |
| optimizer | NVLink TX Responses User Data [Throughput %] | 0.00 |
