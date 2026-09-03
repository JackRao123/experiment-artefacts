# Attribution summary

sqlite: `full-glm53.sqlite`

## Step window per GPU (s)

gpu0: 27.666, gpu1: 27.666, gpu2: 27.666, gpu3: 27.666, gpu4: 27.666, gpu5: 27.666, gpu6: 27.666, gpu7: 27.666

## Kernel time by category (mean per GPU, s)

| category | s/GPU | % of window |
|---|---|---|
| gemm | 4.955 | 17.9 |
| cat_copy | 4.201 | 15.2 |
| hybridep_sync | 3.707 | 13.4 |
| dsa_backward | 3.706 | 13.4 |
| fp32_simt_head | 1.855 | 6.7 |
| elementwise | 1.326 | 4.8 |
| dsa_forward | 1.236 | 4.5 |
| nccl | 1.219 | 4.4 |
| dsa_indexer | 1.091 | 3.9 |
| hybridep_dispatch | 0.921 | 3.3 |
| hybridep_combine | 0.866 | 3.1 |
| moe_permute | 0.777 | 2.8 |
| norm | 0.305 | 1.1 |
| topk_router | 0.141 | 0.5 |
| cub_scan_sort | 0.101 | 0.4 |
| hybridep_meta | 0.077 | 0.3 |
| nonzero_cub | 0.037 | 0.1 |
| other | 0.030 | 0.1 |
| cross_entropy | 0.006 | 0.0 |
| optimizer | 0.003 | 0.0 |

## GPU idle (no kernel resident) inside window

| gpu | idle s | idle % | gaps >1ms | max gap ms |
|---|---|---|---|---|
| 0 | 1.409 | 5.09 | 11 | 121.191 |
| 1 | 1.483 | 5.36 | 11 | 69.408 |
| 2 | 1.852 | 6.7 | 15 | 408.581 |
| 3 | 1.367 | 4.94 | 14 | 69.354 |
| 4 | 1.333 | 4.82 | 9 | 69.219 |
| 5 | 1.525 | 5.51 | 13 | 179.192 |
| 6 | 1.383 | 5.0 | 14 | 67.843 |
| 7 | 1.534 | 5.55 | 14 | 203.902 |

## HybridEP sync waits >= 5.0 ms: who is everyone waiting for?

long waits total (all GPUs): 28.23 s of 29.66 s total sync time

| group | key | wait s (all GPUs) |
|---|---|---|
| by_laggard | gpu0 | 3.297 |
| by_laggard | gpu1 | 4.171 |
| by_laggard | gpu2 | 1.997 |
| by_laggard | gpu3 | 2.613 |
| by_laggard | gpu4 | 4.124 |
| by_laggard | gpu5 | 3.489 |
| by_laggard | gpu6 | 6.59 |
| by_laggard | gpu7 | 1.951 |
| by_kind | laggard_gpu_busy | 26.378 |
| by_kind | laggard_gpu_idle(host) | 1.446 |
| by_kind | mixed | 0.408 |
| by_laggard_top_category | gemm | 23.771 |
| by_laggard_top_category | cat_copy | 4.417 |
| by_laggard_top_category | moe_permute | 0.043 |
| total | long_waits>=5.0ms_all_gpus_s | 28.231 |
| total | all_sync_all_gpus_s | 29.656 |

## Host-side blocking API calls inside window (per GPU)

| gpu | api | calls | total s | max ms |
|---|---|---|---|---|
| 0 | cudaStreamSynchronize_v3020 | 7538 | 22.830 | 405.062 |
| 1 | cudaStreamSynchronize_v3020 | 7537 | 22.207 | 524.986 |
| 2 | cudaStreamSynchronize_v3020 | 7537 | 21.974 | 285.577 |
| 3 | cudaStreamSynchronize_v3020 | 7537 | 22.712 | 525.555 |
| 4 | cudaStreamSynchronize_v3020 | 7537 | 22.459 | 526.306 |
| 5 | cudaStreamSynchronize_v3020 | 7537 | 22.234 | 519.657 |
| 6 | cudaStreamSynchronize_v3020 | 7537 | 22.514 | 525.652 |
| 7 | cudaStreamSynchronize_v3020 | 7537 | 22.248 | 286.573 |

(no CUDA API callchains in this export; capture with --cudabacktrace=sync to get them)

## Top 'other' kernels (s per GPU)

| kernel | instances (all GPUs) | s/GPU | max ms |
|---|---|---|---|
| `compute_cuda_kernel` | 22472 | 0.016 | 0.03 |
| `scan` | 1200 | 0.011 | 0.087 |
| `searchsorted_cuda_kernel` | 4416 | 0.001 | 0.003 |
| `cleanup` | 8 | 0.0 | 0.002 |
| `indexing_backward_kernel` | 1 | 0.0 | 0.057 |
| `ln_fwd_general_kernel` | 336 | 0.0 | 0.011 |
| `pad_tokens_per_expert_kernel` | 1200 | 0.0 | 0.005 |
| `thd_partition_indices_kernel` | 96 | 0.0 | 0.002 |
| `triton_poi_fused__to_copy_all_reduce_scalar_tensor_stack_0` | 1200 | 0.0 | 0.002 |
| `write_indices` | 5 | 0.0 | 0.003 |
