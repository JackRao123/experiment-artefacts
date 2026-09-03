# Attribution summary

sqlite: `fix134.sqlite`

## Step window per GPU (s)

gpu0: 22.610, gpu1: 22.590, gpu2: 22.589, gpu3: 22.588, gpu4: 22.550, gpu5: 22.549, gpu6: 22.549, gpu7: 22.550

## Kernel time by category (mean per GPU, s)

| category | s/GPU | % of window |
|---|---|---|
| gemm | 5.022 | 22.2 |
| dsa_backward | 3.749 | 16.6 |
| cat_copy | 2.678 | 11.9 |
| hybridep_sync | 2.660 | 11.8 |
| dsa_forward | 1.248 | 5.5 |
| elementwise | 1.235 | 5.5 |
| nccl | 1.219 | 5.4 |
| dsa_indexer | 1.108 | 4.9 |
| hybridep_dispatch | 0.889 | 3.9 |
| hybridep_combine | 0.869 | 3.9 |
| moe_permute | 0.786 | 3.5 |
| topk_router | 0.357 | 1.6 |
| norm | 0.307 | 1.4 |
| activation | 0.184 | 0.8 |
| hybridep_meta | 0.083 | 0.4 |
| cub_scan_sort | 0.077 | 0.3 |
| other | 0.014 | 0.1 |
| cross_entropy | 0.006 | 0.0 |
| nonzero_cub | 0.001 | 0.0 |

## GPU idle (no kernel resident) inside window

| gpu | idle s | idle % | gaps >1ms | max gap ms |
|---|---|---|---|---|
| 0 | 0.775 | 3.43 | 7 | 444.01 |
| 1 | 0.333 | 1.48 | 6 | 3.099 |
| 2 | 0.789 | 3.49 | 5 | 447.066 |
| 3 | 0.354 | 1.57 | 8 | 5.47 |
| 4 | 0.347 | 1.54 | 7 | 3.867 |
| 5 | 0.346 | 1.54 | 8 | 5.44 |
| 6 | 0.341 | 1.51 | 5 | 5.538 |
| 7 | 0.353 | 1.57 | 5 | 5.834 |

## NVTX phases (wall s on host thread / GPU kernel s attributed by launch time), mean over GPUs

| range | instances/GPU | wall s | gpu kernel s |
|---|---|---|---|
| forward | 1 | 7.291 | 7.144 |
| backward | 1 | 15.203 | 15.347 |
| optimizer | 1 | 0.000 | 0.000 |
| attention | 156 | 3.111 | 3.181 |
| moe | 150 | 11.430 | 6.291 |
| hybridep_dispatch | 150 | 0.161 | 1.058 |
| hybridep_combine | 150 | 0.021 | 2.390 |
| recompute | 78 | 15.090 | 15.225 |
| lm_head | 1 | 0.172 | 0.070 |
| dsa_indexer | 42 | 2.697 | 0.112 |
| dsa_core | 156 | 0.060 | 1.537 |
| layer_forward | 78 | 6.719 | 6.739 |
| layer_recompute_forward | 78 | 8.678 | 5.108 |
| backward_minus_recompute_forward | 1 | nan | 10.239 |

## HybridEP sync waits >= 5.0 ms: who is everyone waiting for?

long waits total (all GPUs): 19.20 s of 21.28 s total sync time

| group | key | wait s (all GPUs) |
|---|---|---|
| by_laggard | gpu0 | 2.4 |
| by_laggard | gpu1 | 3.436 |
| by_laggard | gpu2 | 1.525 |
| by_laggard | gpu3 | 1.959 |
| by_laggard | gpu4 | 2.95 |
| by_laggard | gpu5 | 1.419 |
| by_laggard | gpu6 | 4.228 |
| by_laggard | gpu7 | 1.286 |
| by_kind | laggard_gpu_busy | 19.192 |
| by_kind | mixed | 0.012 |
| by_laggard_top_category | gemm | 19.079 |
| by_laggard_top_category | moe_permute | 0.124 |
| total | long_waits>=5.0ms_all_gpus_s | 19.204 |
| total | all_sync_all_gpus_s | 21.283 |

## Host-side blocking API calls inside window (per GPU)

| gpu | api | calls | total s | max ms |
|---|---|---|---|---|
| 0 | cudaStreamSynchronize | 533 | 18.081 | 113.680 |
| 1 | cudaStreamSynchronize | 531 | 18.316 | 508.070 |
| 2 | cudaStreamSynchronize | 531 | 17.576 | 111.645 |
| 3 | cudaStreamSynchronize | 531 | 18.089 | 507.786 |
| 4 | cudaStreamSynchronize | 534 | 18.082 | 507.905 |
| 5 | cudaStreamSynchronize | 534 | 18.039 | 507.669 |
| 6 | cudaStreamSynchronize | 534 | 18.088 | 507.599 |
| 7 | cudaStreamSynchronize | 534 | 18.052 | 507.632 |

## Top call chains for blocking syncs

- 1 calls, 0.000 s: `0x7face4bc27db < 0x7face4bc29ab < 0x7face4bbff05 < 0x7face4bc00ff < 0x7fad946ae536 < cudaStreamSynchronize < at::native::_local_scalar_dense_cuda(at::Tensor const&)::{la < at::native::_local_scalar_dense_cuda(at::Tensor const&) < c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::W < at::_ops::_local_scalar_dense::redispatch(c10::DispatchKeySe < c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::W < at::_ops::_local_scalar_dense::call(at::Tensor const&)`

## Top 'other' kernels (s per GPU)

| kernel | instances (all GPUs) | s/GPU | max ms |
|---|---|---|---|
| `scan` | 1200 | 0.012 | 0.108 |
| `searchsorted_cuda_kernel` | 4416 | 0.001 | 0.003 |
| `compute_cuda_kernel` | 144 | 0.0 | 0.006 |
| `indexing_backward_kernel` | 1 | 0.0 | 0.055 |
| `ln_fwd_general_kernel` | 336 | 0.0 | 0.012 |
| `pad_tokens_per_expert_kernel` | 1200 | 0.0 | 0.004 |
| `thd_partition_indices_kernel` | 64 | 0.0 | 0.002 |
| `triton_poi_fused__to_copy_all_reduce_scalar_tensor_stack_0` | 1200 | 0.0 | 0.002 |
| `write_indices` | 5 | 0.0 | 0.003 |

## GPU hardware metrics by phase (mean over GPUs)

| phase | metric | avg |
|---|---|---|
| step | GR Active [Throughput %] | 98.33 |
| step | SMs Active [Throughput %] | 72.01 |
| step | SM Issue [Throughput %] | 20.07 |
| step | Tensor Active [Throughput %] | 27.78 |
| step | DRAM Read Bandwidth [Throughput %] | 11.47 |
| step | DRAM Write Bandwidth [Throughput %] | 7.68 |
| step | NVLink RX Requests Protocol Data [Throughput %] | 0.75 |
| step | NVLink RX Requests User Data [Throughput %] | 2.09 |
| step | NVLink RX Responses Protocol Data [Throughput %] | 0.37 |
| step | NVLink RX Responses User Data [Throughput %] | 1.87 |
| step | NVLink TX Requests Protocol Data [Throughput %] | 0.83 |
| step | NVLink TX Requests User Data [Throughput %] | 2.09 |
| step | NVLink TX Responses Protocol Data [Throughput %] | 0.30 |
| step | NVLink TX Responses User Data [Throughput %] | 1.80 |
| forward | GR Active [Throughput %] | 96.66 |
| forward | SMs Active [Throughput %] | 64.94 |
| forward | SM Issue [Throughput %] | 22.07 |
| forward | Tensor Active [Throughput %] | 27.58 |
| forward | DRAM Read Bandwidth [Throughput %] | 11.29 |
| forward | DRAM Write Bandwidth [Throughput %] | 8.05 |
| forward | NVLink RX Requests Protocol Data [Throughput %] | 0.78 |
| forward | NVLink RX Requests User Data [Throughput %] | 2.24 |
| forward | NVLink RX Responses Protocol Data [Throughput %] | 0.37 |
| forward | NVLink RX Responses User Data [Throughput %] | 1.86 |
| forward | NVLink TX Requests Protocol Data [Throughput %] | 0.85 |
| forward | NVLink TX Requests User Data [Throughput %] | 2.24 |
| forward | NVLink TX Responses Protocol Data [Throughput %] | 0.31 |
| forward | NVLink TX Responses User Data [Throughput %] | 1.79 |
| backward | GR Active [Throughput %] | 99.14 |
| backward | SMs Active [Throughput %] | 75.35 |
| backward | SM Issue [Throughput %] | 19.12 |
| backward | Tensor Active [Throughput %] | 27.91 |
| backward | DRAM Read Bandwidth [Throughput %] | 11.58 |
| backward | DRAM Write Bandwidth [Throughput %] | 7.51 |
| backward | NVLink RX Requests Protocol Data [Throughput %] | 0.74 |
| backward | NVLink RX Requests User Data [Throughput %] | 2.03 |
| backward | NVLink RX Responses Protocol Data [Throughput %] | 0.38 |
| backward | NVLink RX Responses User Data [Throughput %] | 1.88 |
| backward | NVLink TX Requests Protocol Data [Throughput %] | 0.82 |
| backward | NVLink TX Requests User Data [Throughput %] | 2.02 |
| backward | NVLink TX Responses Protocol Data [Throughput %] | 0.30 |
| backward | NVLink TX Responses User Data [Throughput %] | 1.81 |
| optimizer | GR Active [Throughput %] | 26.56 |
| optimizer | SMs Active [Throughput %] | 11.06 |
| optimizer | SM Issue [Throughput %] | 1.32 |
| optimizer | Tensor Active [Throughput %] | 0.00 |
| optimizer | DRAM Read Bandwidth [Throughput %] | 1.98 |
| optimizer | DRAM Write Bandwidth [Throughput %] | 1.65 |
| optimizer | NVLink RX Requests Protocol Data [Throughput %] | 0.00 |
| optimizer | NVLink RX Requests User Data [Throughput %] | 0.00 |
| optimizer | NVLink RX Responses Protocol Data [Throughput %] | 0.00 |
| optimizer | NVLink RX Responses User Data [Throughput %] | 0.00 |
| optimizer | NVLink TX Requests Protocol Data [Throughput %] | 0.00 |
| optimizer | NVLink TX Requests User Data [Throughput %] | 0.00 |
| optimizer | NVLink TX Responses Protocol Data [Throughput %] | 0.00 |
| optimizer | NVLink TX Responses User Data [Throughput %] | 0.00 |
