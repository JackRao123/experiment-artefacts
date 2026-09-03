# Attribution summary

sqlite: `fix1348.sqlite`

## Step window per GPU (s)

gpu0: 23.329, gpu1: 23.307, gpu2: 23.303, gpu3: 23.264, gpu4: 23.306, gpu5: 23.307, gpu6: 23.292, gpu7: 23.299

## Kernel time by category (mean per GPU, s)

| category | s/GPU | % of window |
|---|---|---|
| gemm | 5.105 | 21.9 |
| dsa_backward | 3.758 | 16.1 |
| hybridep_sync | 3.121 | 13.4 |
| cat_copy | 2.684 | 11.5 |
| dsa_forward | 1.250 | 5.4 |
| elementwise | 1.236 | 5.3 |
| nccl | 1.192 | 5.1 |
| dsa_indexer | 1.110 | 4.8 |
| hybridep_dispatch | 0.930 | 4.0 |
| hybridep_combine | 0.871 | 3.7 |
| moe_permute | 0.787 | 3.4 |
| topk_router | 0.366 | 1.6 |
| norm | 0.307 | 1.3 |
| activation | 0.184 | 0.8 |
| hybridep_meta | 0.176 | 0.8 |
| cub_scan_sort | 0.076 | 0.3 |
| other | 0.014 | 0.1 |
| cross_entropy | 0.006 | 0.0 |
| nonzero_cub | 0.001 | 0.0 |

## GPU idle (no kernel resident) inside window

| gpu | idle s | idle % | gaps >1ms | max gap ms |
|---|---|---|---|---|
| 0 | 0.718 | 3.08 | 13 | 100.984 |
| 1 | 0.497 | 2.13 | 14 | 100.285 |
| 2 | 0.362 | 1.55 | 4 | 8.042 |
| 3 | 0.885 | 3.81 | 16 | 91.198 |
| 4 | 0.442 | 1.9 | 10 | 43.788 |
| 5 | 0.408 | 1.75 | 7 | 27.057 |
| 6 | 0.414 | 1.78 | 7 | 34.9 |
| 7 | 0.35 | 1.5 | 6 | 6.496 |

## NVTX phases (wall s on host thread / GPU kernel s attributed by launch time), mean over GPUs

| range | instances/GPU | wall s | gpu kernel s |
|---|---|---|---|
| forward | 1 | 6.994 | 6.939 |
| backward | 1 | 16.226 | 16.233 |
| optimizer | 1 | 0.000 | 0.000 |
| attention | 156 | 3.228 | 3.274 |
| moe | 150 | 11.876 | 6.892 |
| hybridep_dispatch | 150 | 0.330 | 1.567 |
| hybridep_combine | 150 | 0.022 | 2.472 |
| recompute | 78 | 16.110 | 16.109 |
| lm_head | 1 | 0.053 | 0.071 |
| dsa_indexer | 42 | 2.791 | 0.112 |
| dsa_core | 156 | 0.060 | 1.539 |
| layer_forward | 78 | 6.864 | 6.865 |
| layer_recompute_forward | 78 | 9.116 | 5.692 |
| backward_minus_recompute_forward | 1 | nan | 10.541 |

## HybridEP sync waits >= 5.0 ms: who is everyone waiting for?

long waits total (all GPUs): 23.11 s of 24.97 s total sync time

| group | key | wait s (all GPUs) |
|---|---|---|
| by_laggard | gpu0 | 4.845 |
| by_laggard | gpu1 | 3.696 |
| by_laggard | gpu2 | 1.663 |
| by_laggard | gpu3 | 3.013 |
| by_laggard | gpu4 | 2.975 |
| by_laggard | gpu5 | 1.629 |
| by_laggard | gpu6 | 4.034 |
| by_laggard | gpu7 | 1.26 |
| by_kind | laggard_gpu_busy | 19.26 |
| by_kind | laggard_gpu_idle(host) | 3.817 |
| by_kind | mixed | 0.036 |
| by_laggard_top_category | gemm | 19.791 |
| by_laggard_top_category | hybridep_meta | 3.186 |
| by_laggard_top_category | moe_permute | 0.137 |
| total | long_waits>=5.0ms_all_gpus_s | 23.114 |
| total | all_sync_all_gpus_s | 24.965 |

## Host-side blocking API calls inside window (per GPU)

| gpu | api | calls | total s | max ms |
|---|---|---|---|---|
| 0 | cudaStreamSynchronize | 533 | 18.757 | 163.084 |
| 1 | cudaStreamSynchronize | 531 | 18.538 | 173.278 |
| 2 | cudaStreamSynchronize | 531 | 18.537 | 168.397 |
| 3 | cudaStreamSynchronize | 531 | 18.360 | 167.192 |
| 4 | cudaStreamSynchronize | 531 | 18.566 | 171.946 |
| 5 | cudaStreamSynchronize | 531 | 18.612 | 170.864 |
| 6 | cudaStreamSynchronize | 532 | 18.518 | 172.178 |
| 7 | cudaStreamSynchronize | 532 | 18.734 | 172.674 |

## Top call chains for blocking syncs

- 1 calls, 0.000 s: `0x7f3485cc37db < 0x7f3485cc39ab < 0x7f3485cc0f05 < 0x7f3485cc10ff < 0x7f353baae536 < cudaStreamSynchronize < at::native::_local_scalar_dense_cuda(at::Tensor const&)::{la < at::native::_local_scalar_dense_cuda(at::Tensor const&) < c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::W < at::_ops::_local_scalar_dense::redispatch(c10::DispatchKeySe < c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::W < at::_ops::_local_scalar_dense::call(at::Tensor const&)`

## Top 'other' kernels (s per GPU)

| kernel | instances (all GPUs) | s/GPU | max ms |
|---|---|---|---|
| `scan` | 1200 | 0.012 | 0.108 |
| `searchsorted_cuda_kernel` | 4416 | 0.001 | 0.003 |
| `compute_cuda_kernel` | 144 | 0.0 | 0.006 |
| `indexing_backward_kernel` | 1 | 0.0 | 0.06 |
| `ln_fwd_general_kernel` | 336 | 0.0 | 0.012 |
| `pad_tokens_per_expert_kernel` | 1200 | 0.0 | 0.004 |
| `thd_partition_indices_kernel` | 64 | 0.0 | 0.002 |
| `triton_poi_fused__to_copy_all_reduce_scalar_tensor_stack_0` | 1200 | 0.0 | 0.002 |
| `write_indices` | 5 | 0.0 | 0.003 |

## GPU hardware metrics by phase (mean over GPUs)

| phase | metric | avg |
|---|---|---|
| step | GR Active [Throughput %] | 98.20 |
| step | SMs Active [Throughput %] | 70.54 |
| step | SM Issue [Throughput %] | 19.56 |
| step | Tensor Active [Throughput %] | 27.26 |
| step | DRAM Read Bandwidth [Throughput %] | 11.17 |
| step | DRAM Write Bandwidth [Throughput %] | 7.47 |
| step | NVLink RX Requests Protocol Data [Throughput %] | 0.74 |
| step | NVLink RX Requests User Data [Throughput %] | 2.03 |
| step | NVLink RX Responses Protocol Data [Throughput %] | 0.39 |
| step | NVLink RX Responses User Data [Throughput %] | 1.83 |
| step | NVLink TX Requests Protocol Data [Throughput %] | 0.83 |
| step | NVLink TX Requests User Data [Throughput %] | 2.03 |
| step | NVLink TX Responses Protocol Data [Throughput %] | 0.30 |
| step | NVLink TX Responses User Data [Throughput %] | 1.75 |
| forward | GR Active [Throughput %] | 97.91 |
| forward | SMs Active [Throughput %] | 68.19 |
| forward | SM Issue [Throughput %] | 23.05 |
| forward | Tensor Active [Throughput %] | 29.13 |
| forward | DRAM Read Bandwidth [Throughput %] | 11.82 |
| forward | DRAM Write Bandwidth [Throughput %] | 8.39 |
| forward | NVLink RX Requests Protocol Data [Throughput %] | 0.82 |
| forward | NVLink RX Requests User Data [Throughput %] | 2.34 |
| forward | NVLink RX Responses Protocol Data [Throughput %] | 0.40 |
| forward | NVLink RX Responses User Data [Throughput %] | 1.96 |
| forward | NVLink TX Requests Protocol Data [Throughput %] | 0.90 |
| forward | NVLink TX Requests User Data [Throughput %] | 2.34 |
| forward | NVLink TX Responses Protocol Data [Throughput %] | 0.33 |
| forward | NVLink TX Responses User Data [Throughput %] | 1.88 |
| backward | GR Active [Throughput %] | 98.31 |
| backward | SMs Active [Throughput %] | 71.50 |
| backward | SM Issue [Throughput %] | 18.07 |
| backward | Tensor Active [Throughput %] | 26.48 |
| backward | DRAM Read Bandwidth [Throughput %] | 10.90 |
| backward | DRAM Write Bandwidth [Throughput %] | 7.09 |
| backward | NVLink RX Requests Protocol Data [Throughput %] | 0.70 |
| backward | NVLink RX Requests User Data [Throughput %] | 1.91 |
| backward | NVLink RX Responses Protocol Data [Throughput %] | 0.38 |
| backward | NVLink RX Responses User Data [Throughput %] | 1.79 |
| backward | NVLink TX Requests Protocol Data [Throughput %] | 0.80 |
| backward | NVLink TX Requests User Data [Throughput %] | 1.90 |
| backward | NVLink TX Responses Protocol Data [Throughput %] | 0.29 |
| backward | NVLink TX Responses User Data [Throughput %] | 1.70 |
| optimizer | GR Active [Throughput %] | 31.66 |
| optimizer | SMs Active [Throughput %] | 9.96 |
| optimizer | SM Issue [Throughput %] | 1.26 |
| optimizer | Tensor Active [Throughput %] | 0.00 |
| optimizer | DRAM Read Bandwidth [Throughput %] | 1.92 |
| optimizer | DRAM Write Bandwidth [Throughput %] | 1.64 |
| optimizer | NVLink RX Requests Protocol Data [Throughput %] | 0.00 |
| optimizer | NVLink RX Requests User Data [Throughput %] | 0.00 |
| optimizer | NVLink RX Responses Protocol Data [Throughput %] | 0.00 |
| optimizer | NVLink RX Responses User Data [Throughput %] | 0.00 |
| optimizer | NVLink TX Requests Protocol Data [Throughput %] | 0.00 |
| optimizer | NVLink TX Requests User Data [Throughput %] | 0.00 |
| optimizer | NVLink TX Responses Protocol Data [Throughput %] | 0.00 |
| optimizer | NVLink TX Responses User Data [Throughput %] | 0.00 |
