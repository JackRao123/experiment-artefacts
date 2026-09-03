# Attribution summary

sqlite: `full-glm53-nvtx.sqlite`

## Step window per GPU (s)

gpu0: 27.420, gpu1: 27.398, gpu2: 27.397, gpu3: 27.399, gpu4: 27.398, gpu5: 27.397, gpu6: 27.397, gpu7: 27.398

## Kernel time by category (mean per GPU, s)

| category | s/GPU | % of window |
|---|---|---|
| gemm | 4.903 | 17.9 |
| hybridep_sync | 3.869 | 14.1 |
| dsa_backward | 3.705 | 13.5 |
| cat_copy | 2.948 | 10.8 |
| elementwise | 2.193 | 8.0 |
| fp32_simt_head | 1.853 | 6.8 |
| dsa_forward | 1.233 | 4.5 |
| dsa_indexer | 1.090 | 4.0 |
| hybridep_dispatch | 0.886 | 3.2 |
| hybridep_combine | 0.868 | 3.2 |
| moe_permute | 0.777 | 2.8 |
| nccl | 0.720 | 2.6 |
| activation | 0.372 | 1.4 |
| norm | 0.305 | 1.1 |
| topk_router | 0.143 | 0.5 |
| cub_scan_sort | 0.101 | 0.4 |
| hybridep_meta | 0.080 | 0.3 |
| nonzero_cub | 0.037 | 0.1 |
| other | 0.030 | 0.1 |
| cross_entropy | 0.006 | 0.0 |

## GPU idle (no kernel resident) inside window

| gpu | idle s | idle % | gaps >1ms | max gap ms |
|---|---|---|---|---|
| 0 | 1.426 | 5.2 | 6 | 47.163 |
| 1 | 1.931 | 7.05 | 10 | 321.904 |
| 2 | 1.648 | 6.02 | 11 | 6.885 |
| 3 | 1.711 | 6.25 | 8 | 4.725 |
| 4 | 1.548 | 5.65 | 6 | 7.059 |
| 5 | 1.78 | 6.5 | 6 | 128.417 |
| 6 | 1.553 | 5.67 | 7 | 6.359 |
| 7 | 1.631 | 5.95 | 7 | 5.499 |

## NVTX phases (wall s on host thread / GPU kernel s attributed by launch time), mean over GPUs

| range | instances/GPU | wall s | gpu kernel s |
|---|---|---|---|
| forward | 1 | 8.977 | 8.302 |
| backward | 1 | 18.345 | 17.817 |
| optimizer | 1 | 0.000 | 0.000 |
| attention | 156 | 11.306 | 3.381 |
| moe | 150 | 5.133 | 7.802 |
| hybridep_dispatch | 150 | 0.160 | 1.045 |
| hybridep_combine | 150 | 0.022 | 3.251 |
| recompute | 78 | 17.501 | 16.861 |
| lm_head | 1 | 0.907 | 0.957 |
| dsa_indexer | 42 | 0.205 | 0.110 |
| dsa_core | 156 | 0.064 | 1.520 |
| layer_forward | 78 | 7.946 | 7.339 |
| layer_recompute_forward | 78 | 9.371 | 6.213 |
| backward_minus_recompute_forward | 1 | nan | 11.604 |

## HybridEP sync waits >= 5.0 ms: who is everyone waiting for?

long waits total (all GPUs): 29.24 s of 30.95 s total sync time

| group | key | wait s (all GPUs) |
|---|---|---|
| by_laggard | gpu0 | 3.686 |
| by_laggard | gpu1 | 6.474 |
| by_laggard | gpu2 | 2.004 |
| by_laggard | gpu3 | 2.668 |
| by_laggard | gpu4 | 4.168 |
| by_laggard | gpu5 | 2.08 |
| by_laggard | gpu6 | 6.337 |
| by_laggard | gpu7 | 1.827 |
| by_kind | laggard_gpu_busy | 26.49 |
| by_kind | laggard_gpu_idle(host) | 2.754 |
| by_laggard_top_category | gemm | 28.083 |
| by_laggard_top_category | elementwise | 1.088 |
| by_laggard_top_category | moe_permute | 0.073 |
| total | long_waits>=5.0ms_all_gpus_s | 29.244 |
| total | all_sync_all_gpus_s | 30.953 |

## Host-side blocking API calls inside window (per GPU)

| gpu | api | calls | total s | max ms |
|---|---|---|---|---|
| 0 | cudaStreamSynchronize | 7523 | 22.077 | 436.620 |
| 1 | cudaStreamSynchronize | 7521 | 21.141 | 131.268 |
| 2 | cudaStreamSynchronize | 7521 | 21.495 | 434.570 |
| 3 | cudaStreamSynchronize | 7521 | 21.325 | 435.304 |
| 4 | cudaStreamSynchronize | 7521 | 21.185 | 131.372 |
| 5 | cudaStreamSynchronize | 7521 | 21.230 | 279.794 |
| 6 | cudaStreamSynchronize | 7521 | 21.696 | 436.831 |
| 7 | cudaStreamSynchronize | 7521 | 21.452 | 436.244 |

## Top call chains for blocking syncs

- 1 calls, 0.000 s: `0x7fcc646c17db < 0x7fcc646c19ab < 0x7fcc646bef05 < 0x7fcc646bf0ff < 0x7fcd150ae536 < cudaStreamSynchronize < at::native::_local_scalar_dense_cuda(at::Tensor const&)::{la < at::native::_local_scalar_dense_cuda(at::Tensor const&) < c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::W < at::_ops::_local_scalar_dense::redispatch(c10::DispatchKeySe < c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::W < at::_ops::_local_scalar_dense::call(at::Tensor const&)`

## Top 'other' kernels (s per GPU)

| kernel | instances (all GPUs) | s/GPU | max ms |
|---|---|---|---|
| `compute_cuda_kernel` | 22464 | 0.016 | 0.008 |
| `scan` | 1200 | 0.011 | 0.088 |
| `searchsorted_cuda_kernel` | 4416 | 0.001 | 0.003 |
| `indexing_backward_kernel` | 1 | 0.0 | 0.057 |
| `ln_fwd_general_kernel` | 336 | 0.0 | 0.011 |
| `pad_tokens_per_expert_kernel` | 1200 | 0.0 | 0.004 |
| `thd_partition_indices_kernel` | 64 | 0.0 | 0.002 |
| `triton_poi_fused__to_copy_all_reduce_scalar_tensor_stack_0` | 1200 | 0.0 | 0.002 |
| `write_indices` | 5 | 0.0 | 0.003 |

## GPU hardware metrics by phase (mean over GPUs)

| phase | metric | avg |
|---|---|---|
| step | GR Active [Throughput %] | 94.77 |
| step | SMs Active [Throughput %] | 69.51 |
| step | SM Issue [Throughput %] | 22.19 |
| step | Tensor Active [Throughput %] | 22.41 |
| step | DRAM Read Bandwidth [Throughput %] | 11.61 |
| step | DRAM Write Bandwidth [Throughput %] | 7.21 |
| step | NVLink RX Requests Protocol Data [Throughput %] | 0.62 |
| step | NVLink RX Requests User Data [Throughput %] | 1.71 |
| step | NVLink RX Responses Protocol Data [Throughput %] | 0.34 |
| step | NVLink RX Responses User Data [Throughput %] | 1.56 |
| step | NVLink TX Requests Protocol Data [Throughput %] | 0.72 |
| step | NVLink TX Requests User Data [Throughput %] | 1.70 |
| step | NVLink TX Responses Protocol Data [Throughput %] | 0.26 |
| step | NVLink TX Responses User Data [Throughput %] | 1.47 |
| forward | GR Active [Throughput %] | 92.28 |
| forward | SMs Active [Throughput %] | 66.47 |
| forward | SM Issue [Throughput %] | 25.89 |
| forward | Tensor Active [Throughput %] | 21.89 |
| forward | DRAM Read Bandwidth [Throughput %] | 11.20 |
| forward | DRAM Write Bandwidth [Throughput %] | 7.33 |
| forward | NVLink RX Requests Protocol Data [Throughput %] | 0.63 |
| forward | NVLink RX Requests User Data [Throughput %] | 1.80 |
| forward | NVLink RX Responses Protocol Data [Throughput %] | 0.33 |
| forward | NVLink RX Responses User Data [Throughput %] | 1.53 |
| forward | NVLink TX Requests Protocol Data [Throughput %] | 0.71 |
| forward | NVLink TX Requests User Data [Throughput %] | 1.79 |
| forward | NVLink TX Responses Protocol Data [Throughput %] | 0.26 |
| forward | NVLink TX Responses User Data [Throughput %] | 1.45 |
| backward | GR Active [Throughput %] | 95.98 |
| backward | SMs Active [Throughput %] | 70.94 |
| backward | SM Issue [Throughput %] | 20.40 |
| backward | Tensor Active [Throughput %] | 22.67 |
| backward | DRAM Read Bandwidth [Throughput %] | 11.82 |
| backward | DRAM Write Bandwidth [Throughput %] | 7.15 |
| backward | NVLink RX Requests Protocol Data [Throughput %] | 0.62 |
| backward | NVLink RX Requests User Data [Throughput %] | 1.67 |
| backward | NVLink RX Responses Protocol Data [Throughput %] | 0.35 |
| backward | NVLink RX Responses User Data [Throughput %] | 1.58 |
| backward | NVLink TX Requests Protocol Data [Throughput %] | 0.72 |
| backward | NVLink TX Requests User Data [Throughput %] | 1.67 |
| backward | NVLink TX Responses Protocol Data [Throughput %] | 0.26 |
| backward | NVLink TX Responses User Data [Throughput %] | 1.48 |
| optimizer | GR Active [Throughput %] | 23.69 |
| optimizer | SMs Active [Throughput %] | 9.78 |
| optimizer | SM Issue [Throughput %] | 1.17 |
| optimizer | Tensor Active [Throughput %] | 0.00 |
| optimizer | DRAM Read Bandwidth [Throughput %] | 1.79 |
| optimizer | DRAM Write Bandwidth [Throughput %] | 1.51 |
| optimizer | NVLink RX Requests Protocol Data [Throughput %] | 0.01 |
| optimizer | NVLink RX Requests User Data [Throughput %] | 0.01 |
| optimizer | NVLink RX Responses Protocol Data [Throughput %] | 0.00 |
| optimizer | NVLink RX Responses User Data [Throughput %] | 0.00 |
| optimizer | NVLink TX Requests Protocol Data [Throughput %] | 0.00 |
| optimizer | NVLink TX Requests User Data [Throughput %] | 0.00 |
| optimizer | NVLink TX Responses Protocol Data [Throughput %] | 0.00 |
| optimizer | NVLink TX Responses User Data [Throughput %] | 0.00 |
