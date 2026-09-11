# GLM-5.3 1d2m, 262144 tokens: topology results

All results: one warmup, five Kineto-unprofiled controls with lightweight CUDA-event timers, one memory capture, one runtime capture. LoRA rank 32, full one-block recompute, native-FP8 expert weights. Timing is forward/backward only unless specified. Same main pin as prior full-model run.

## Measured three-block proxy

| Topology | GPUs | FB mean ± SD (s) | Tokens/s total | Tokens/s/GPU | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|---:|
| cp8ep8 | 8 | 1.2932 ± 0.0491 | 202,703 | 25,338 | 37.14 | 38.59 |
| cp8ep1 | 8 | 1.4510 ± 0.2408 | 180,668 | 22,584 | 65.35 | 68.67 |
| cp1ep1 | 1 | 58.1208 ± 0.2402 | 4,510 | 4,510 | 227.79 | 254.19 |

Memory is the maximum allocator peak across the distributed ranks and five controls, not total physical HBM used by every library.

## Whole-block layer timings

GPU CUDA-event elapsed milliseconds, rank 0, mean ± sample SD over all five controls. Each block includes attention and MLP. Actual backward excludes recomputation; minor checkpoint boundary setup is included. All-rank raw records and conservative max-rank summaries are in each summary.json.

| Topology | Block type | Forward ms | Recompute ms | Actual backward ms | Backward incl. recompute ms |
|---|---|---:|---:|---:|---:|
| cp8ep8 | Dense / indexer | 111.14 ± 1.38 | 39.56 ± 0.47 | 88.78 ± 0.97 | 128.34 ± 1.22 |
| cp8ep8 | MoE / shared indices | 76.41 ± 0.46 | 87.49 ± 47.11 | 117.27 ± 3.70 | 204.76 ± 44.41 |
| cp8ep8 | MoE / indexer | 127.79 ± 1.98 | 70.52 ± 1.16 | 105.50 ± 0.47 | 176.01 ± 0.93 |
| cp8ep1 | Dense / indexer | 112.96 ± 14.29 | 40.06 ± 1.17 | 89.21 ± 1.31 | 129.27 ± 2.00 |
| cp8ep1 | MoE / shared indices | 74.63 ± 7.02 | 107.92 ± 106.95 | 188.85 ± 175.71 | 296.78 ± 182.68 |
| cp8ep1 | MoE / indexer | 135.54 ± 11.15 | 86.11 ± 39.65 | 129.93 ± 36.65 | 216.04 ± 76.29 |
| cp1ep1 | Dense / indexer | 26016.36 ± 8.64 | 329.43 ± 2.29 | 741.05 ± 20.16 | 1070.48 ± 22.28 |
| cp1ep1 | MoE / shared indices | 407.22 ± 4.28 | 397.69 ± 2.24 | 800.88 ± 7.32 | 1198.57 ± 9.14 |
| cp1ep1 | MoE / indexer | 26116.07 ± 6.30 | 501.94 ± 181.90 | 810.90 ± 4.70 | 1312.83 ± 179.20 |

## Weighted full-model estimate

Actual checkpoint counts: 3 dense/indexer + 57 MoE/shared + 18 MoE/indexer. No dense/shared blocks exist in the full model. The following is a timing extrapolation, not a claim that the full model fits the EP1 topologies.

| Topology | Forward s | Recompute s | Actual backward s | Blocks FB mean ± SD (s) | FB including proxy non-block residual (s) |
|---|---:|---:|---:|---:|---:|
| cp8ep8 | 6.989 | 6.375 | 8.850 | 22.213 ± 2.521 | 22.682 |
| cp8ep1 | 7.033 | 7.822 | 13.371 | 28.225 ± 11.202 | 28.711 |
| cp1ep1 | 571.350 | 32.691 | 62.470 | 666.511 ± 2.798 | 668.510 |

The residual is each control's measured FB time minus its measured three-block GPU time. Adding that residual assumes embedding/head/loss, orchestration, and gradient-finalization costs remain comparable; optimizer time is excluded.
All five controls are retained. CP8EP1 has large outliers and its weighted estimate is correspondingly noisy; do not treat its mean as a reliable speed ranking against CP8EP8.
The previous full CP8EP8 model measured 22.133 s FB. Agreement with the proxy is a sanity check, not an accuracy guarantee. Depth-dependent routing, index-cache lifetimes, communication overlap and memory pressure can change.

## Communication evidence from the separate runtime captures

Kernel-duration sums in ms, rank 0, one profiled step including optimizer. These are not additive critical-path penalties or pure network-transfer time. HybridEP fuses permutation, dispatch/combine, and communication; local routing work can remain at EP1.

| Topology | NCCL AllGather | NCCL ReduceScatter | NCCL AllReduce | Dispatch/combine-named kernels | Total GPU kernel union / span ms |
|---|---:|---:|---:|---:|---:|
| cp8ep8 | 22.416 | 15.412 | 8.053 | 49.655 | 980.5 / 1322.4 |
| cp8ep1 | 24.195 | 137.842 | 6.236 | 0.000 | 1057.5 / 1397.3 |
| cp1ep1 | 0.000 | 0.000 | 0.000 | 0.000 | 58050.4 / 58710.3 |

- CP8EP8 uses HybridEP. HybridEP EP1 failed at startup with a cooperative-launch-too-large CUDA error; failure log is retained. CP8EP1 and CP1EP1 use the existing alltoall dispatcher's EP1 local path. This is therefore not a dispatcher-controlled communication-only ablation.
- EP1 removes inter-GPU expert dispatch/combine transfer, not local expert sorting, permutation, gathering/scattering, or grouped GEMM overhead.
- CP1 removes context-parallel collectives. It also puts all 262144 tokens on one GPU instead of 32768 tokens per GPU, so raw layer latency is not directly comparable as a communication-only delta.
- CP8 with TP1 uses CP collectives for attention; other NCCL calls can include gradient synchronization and request metadata. Attribution SQL and raw kernel lists are retained.
- No SM/tensor-core saturation claim is made from Kineto; hardware counters were not collected.

## Confirmed CP1 indexer slow path

The CP1 runtime trace contains 16,384 calls to cutlass_80_simt_sgemm_128x64_8x5_nn_align1, totaling 25.596 seconds, and no cuDNN indexer-forward kernels. The three dominant FP32 elementwise kernels add 11.819 + 7.674 + 5.118 seconds of kernel duration. This is an unfused FP32 SIMT score-computation path, not merely eight times more BF16 fused work.
The count matches two indexer blocks × 256 score chunks × 32 heads. Source: dsa_cudnn_kernels.py _indexer_topk_from_score_chunks selects _compute_indexer_scores_chunk_with_global_rows, which loops over heads using FP32 bmm, ReLU, weighting and accumulation. The packed-CP path instead supplies bottom_right_key_start and invokes the cuDNN indexer with causal offsets. Sparse attention itself still uses fused sparse kernels.
CP1 has zero NCCL kernels in this capture, so context/expert inter-GPU communication does disappear. The large slowdown is dominated by the changed indexer execution path. The 666.5-second full-model block estimate describes this current slow path; it is not an estimate of an optimized CP1 implementation, nor a feasible full-model EP1 deployment on one B300.
Source locations in the pinned Megatron-Core tree: transformer/experimental_attention_variant/dsa_cudnn_kernels.py:537 (FP32 head loop), :645 (cuDNN vs fallback branch), :885 (packed-CP causal-offset route), :1186/:1232 (generic chunked entry points). No fix was applied.

## Files

- Config: ../../configs/glm53-debug-1d2m-config.json
- Per topology: benchmark.json, summary.json, layer_timings/rank*.jsonl, runtime/*.pt.trace.json, memory/memory.rank0.pickle, analysis/*.csv and analysis/summary.json.
- Artifacts were copied locally, hashed, and parsed. Original tools/profile_driver.py and tools/mfu.py were not edited for this experiment.

- cp8ep8: [Kineto trace](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_topology_262k_20260909/cp8ep8/runtime/baseten-training-job-32vj99q-multinode-0_78246.1788997399748347765.pt.trace.json) · [Memory snapshot](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_topology_262k_20260909/cp8ep8/memory/memory.rank0.pickle) · [Detailed statistics](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_topology_262k_20260909/cp8ep8/summary.json)
- cp8ep1: [Kineto trace](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_topology_262k_20260909/cp8ep1/runtime/baseten-training-job-32vj99q-multinode-0_92548.1788997879930811953.pt.trace.json) · [Memory snapshot](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_topology_262k_20260909/cp8ep1/memory/memory.rank0.pickle) · [Detailed statistics](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_topology_262k_20260909/cp8ep1/summary.json)
- cp1ep1: [Kineto trace](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_topology_262k_20260909/cp1ep1/runtime/baseten-training-job-32vj99q-multinode-0_97888.1788998518121179457.pt.trace.json) · [Memory snapshot](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_topology_262k_20260909/cp1ep1/memory/memory.rank0.pickle) · [Detailed statistics](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_topology_262k_20260909/cp1ep1/summary.json)
