# Attribution summary

sqlite: `fix1.sqlite`

## Step window per GPU (s)

gpu0: 26.587, gpu1: 26.564, gpu2: 26.565, gpu3: 26.564, gpu4: 26.564, gpu5: 26.566, gpu6: 26.565, gpu7: 26.564

## Kernel time by category (mean per GPU, s)

| category | s/GPU | % of window |
|---|---|---|
| gemm | 4.904 | 18.5 |
| hybridep_sync | 4.500 | 16.9 |
| dsa_backward | 3.705 | 13.9 |
| cat_copy | 2.908 | 10.9 |
| elementwise | 2.166 | 8.2 |
| fp32_simt_head | 1.853 | 7.0 |
| dsa_forward | 1.237 | 4.7 |
| dsa_indexer | 1.095 | 4.1 |
| hybridep_dispatch | 0.888 | 3.3 |
| hybridep_combine | 0.869 | 3.3 |
| moe_permute | 0.781 | 2.9 |
| nccl | 0.581 | 2.2 |
| activation | 0.374 | 1.4 |
| norm | 0.305 | 1.1 |
| topk_router | 0.144 | 0.5 |
| hybridep_meta | 0.082 | 0.3 |
| cub_scan_sort | 0.076 | 0.3 |
| other | 0.014 | 0.1 |
| cross_entropy | 0.006 | 0.0 |
| nonzero_cub | 0.001 | 0.0 |

## GPU idle (no kernel resident) inside window

| gpu | idle s | idle % | gaps >1ms | max gap ms |
|---|---|---|---|---|
| 0 | 0.332 | 1.25 | 5 | 3.851 |
| 1 | 0.988 | 3.72 | 4 | 659.754 |
| 2 | 0.323 | 1.21 | 1 | 6.369 |
| 3 | 0.326 | 1.23 | 1 | 6.178 |
| 4 | 0.331 | 1.25 | 1 | 6.131 |
| 5 | 0.323 | 1.22 | 5 | 5.944 |
| 6 | 0.702 | 2.64 | 3 | 383.395 |
| 7 | 0.325 | 1.22 | 2 | 6.226 |

## NVTX phases (wall s on host thread / GPU kernel s attributed by launch time), mean over GPUs

| range | instances/GPU | wall s | gpu kernel s |
|---|---|---|---|
| forward | 1 | 8.220 | 8.199 |
| backward | 1 | 18.274 | 18.287 |
| optimizer | 1 | 0.000 | 0.000 |
| attention | 156 | 3.188 | 3.099 |
| moe | 150 | 11.814 | 8.450 |
| hybridep_dispatch | 150 | 0.155 | 1.054 |
| hybridep_combine | 150 | 0.021 | 3.872 |
| recompute | 78 | 17.429 | 17.331 |
| lm_head | 1 | 0.907 | 0.957 |
| dsa_indexer | 42 | 2.795 | 0.111 |
| dsa_core | 156 | 0.060 | 1.525 |
| layer_forward | 78 | 7.191 | 7.238 |
| layer_recompute_forward | 78 | 8.661 | 6.682 |
| backward_minus_recompute_forward | 1 | nan | 11.605 |

## HybridEP sync waits >= 5.0 ms: who is everyone waiting for?

long waits total (all GPUs): 34.26 s of 36.00 s total sync time

| group | key | wait s (all GPUs) |
|---|---|---|
| by_laggard | gpu0 | 3.39 |
| by_laggard | gpu1 | 9.123 |
| by_laggard | gpu2 | 2.007 |
| by_laggard | gpu3 | 2.631 |
| by_laggard | gpu4 | 4.147 |
| by_laggard | gpu5 | 2.086 |
| by_laggard | gpu6 | 9.056 |
| by_laggard | gpu7 | 1.824 |
| by_kind | laggard_gpu_busy | 26.576 |
| by_kind | laggard_gpu_idle(host) | 7.688 |
| by_laggard_top_category | gemm | 31.849 |
| by_laggard_top_category | elementwise | 2.363 |
| by_laggard_top_category | moe_permute | 0.052 |
| total | long_waits>=5.0ms_all_gpus_s | 34.264 |
| total | all_sync_all_gpus_s | 36.001 |

## Host-side blocking API calls inside window (per GPU)

| gpu | api | calls | total s | max ms |
|---|---|---|---|---|
| 0 | cudaStreamSynchronize | 548 | 22.558 | 773.063 |
| 1 | cudaStreamSynchronize | 278 | 15.039 | 489.736 |
| 2 | cudaStreamSynchronize | 278 | 15.493 | 774.436 |
| 3 | cudaStreamSynchronize | 278 | 15.702 | 774.710 |
| 4 | cudaStreamSynchronize | 278 | 15.608 | 775.277 |
| 5 | cudaStreamSynchronize | 278 | 15.669 | 774.698 |
| 6 | cudaStreamSynchronize | 278 | 15.257 | 776.035 |
| 7 | cudaStreamSynchronize | 278 | 15.601 | 776.366 |

## Top call chains for blocking syncs

- 1 calls, 0.000 s: `0x7f30984c17db < 0x7f30984c19ab < 0x7f30984bef05 < 0x7f30984bf0ff < 0x7f31478ae536 < cudaStreamSynchronize < at::native::_local_scalar_dense_cuda(at::Tensor const&)::{la < at::native::_local_scalar_dense_cuda(at::Tensor const&) < c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::W < at::_ops::_local_scalar_dense::redispatch(c10::DispatchKeySe < c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::W < at::_ops::_local_scalar_dense::call(at::Tensor const&)`

## Top 'other' kernels (s per GPU)

| kernel | instances (all GPUs) | s/GPU | max ms |
|---|---|---|---|
| `scan` | 1200 | 0.012 | 0.096 |
| `searchsorted_cuda_kernel` | 4416 | 0.001 | 0.003 |
| `compute_cuda_kernel` | 144 | 0.0 | 0.006 |
| `indexing_backward_kernel` | 1 | 0.0 | 0.056 |
| `ln_fwd_general_kernel` | 336 | 0.0 | 0.012 |
| `pad_tokens_per_expert_kernel` | 1200 | 0.0 | 0.004 |
| `thd_partition_indices_kernel` | 64 | 0.0 | 0.002 |
| `triton_poi_fused__to_copy_all_reduce_scalar_tensor_stack_0` | 1200 | 0.0 | 0.002 |
| `write_indices` | 5 | 0.0 | 0.004 |

## GPU hardware metrics by phase (mean over GPUs)

| phase | metric | avg |
|---|---|---|
| step | GR Active [Throughput %] | 98.65 |
| step | SMs Active [Throughput %] | 71.61 |
| step | SM Issue [Throughput %] | 22.91 |
| step | Tensor Active [Throughput %] | 23.13 |
| step | DRAM Read Bandwidth [Throughput %] | 11.94 |
| step | DRAM Write Bandwidth [Throughput %] | 7.38 |
| step | NVLink RX Requests Protocol Data [Throughput %] | 0.65 |
| step | NVLink RX Requests User Data [Throughput %] | 1.77 |
| step | NVLink RX Responses Protocol Data [Throughput %] | 0.38 |
| step | NVLink RX Responses User Data [Throughput %] | 1.63 |
| step | NVLink TX Requests Protocol Data [Throughput %] | 0.76 |
| step | NVLink TX Requests User Data [Throughput %] | 1.77 |
| step | NVLink TX Responses Protocol Data [Throughput %] | 0.27 |
| step | NVLink TX Responses User Data [Throughput %] | 1.53 |
| forward | GR Active [Throughput %] | 98.77 |
| forward | SMs Active [Throughput %] | 72.48 |
| forward | SM Issue [Throughput %] | 28.28 |
| forward | Tensor Active [Throughput %] | 23.95 |
| forward | DRAM Read Bandwidth [Throughput %] | 12.17 |
| forward | DRAM Write Bandwidth [Throughput %] | 7.92 |
| forward | NVLink RX Requests Protocol Data [Throughput %] | 0.70 |
| forward | NVLink RX Requests User Data [Throughput %] | 1.97 |
| forward | NVLink RX Responses Protocol Data [Throughput %] | 0.36 |
| forward | NVLink RX Responses User Data [Throughput %] | 1.68 |
| forward | NVLink TX Requests Protocol Data [Throughput %] | 0.78 |
| forward | NVLink TX Requests User Data [Throughput %] | 1.97 |
| forward | NVLink TX Responses Protocol Data [Throughput %] | 0.28 |
| forward | NVLink TX Responses User Data [Throughput %] | 1.59 |
| backward | GR Active [Throughput %] | 98.61 |
| backward | SMs Active [Throughput %] | 71.16 |
| backward | SM Issue [Throughput %] | 20.51 |
| backward | Tensor Active [Throughput %] | 22.75 |
| backward | DRAM Read Bandwidth [Throughput %] | 11.86 |
| backward | DRAM Write Bandwidth [Throughput %] | 7.15 |
| backward | NVLink RX Requests Protocol Data [Throughput %] | 0.63 |
| backward | NVLink RX Requests User Data [Throughput %] | 1.69 |
| backward | NVLink RX Responses Protocol Data [Throughput %] | 0.39 |
| backward | NVLink RX Responses User Data [Throughput %] | 1.62 |
| backward | NVLink TX Requests Protocol Data [Throughput %] | 0.75 |
| backward | NVLink TX Requests User Data [Throughput %] | 1.68 |
| backward | NVLink TX Responses Protocol Data [Throughput %] | 0.26 |
| backward | NVLink TX Responses User Data [Throughput %] | 1.50 |
| optimizer | GR Active [Throughput %] | 24.96 |
| optimizer | SMs Active [Throughput %] | 11.63 |
| optimizer | SM Issue [Throughput %] | 1.36 |
| optimizer | Tensor Active [Throughput %] | 0.00 |
| optimizer | DRAM Read Bandwidth [Throughput %] | 2.08 |
| optimizer | DRAM Write Bandwidth [Throughput %] | 1.74 |
| optimizer | NVLink RX Requests Protocol Data [Throughput %] | 0.00 |
| optimizer | NVLink RX Requests User Data [Throughput %] | 0.00 |
| optimizer | NVLink RX Responses Protocol Data [Throughput %] | 0.00 |
| optimizer | NVLink RX Responses User Data [Throughput %] | 0.00 |
| optimizer | NVLink TX Requests Protocol Data [Throughput %] | 0.00 |
| optimizer | NVLink TX Requests User Data [Throughput %] | 0.00 |
| optimizer | NVLink TX Responses Protocol Data [Throughput %] | 0.00 |
| optimizer | NVLink TX Responses User Data [Throughput %] | 0.00 |
