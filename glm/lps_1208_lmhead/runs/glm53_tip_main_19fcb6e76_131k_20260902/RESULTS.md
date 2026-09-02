# GLM-5.3 tip-of-main 131K memory profile

## Run

- Trainers commit: `19fcb6e769cede699d7856c54349e265f7e89177`
- Megatron-Bridge: `5deb591cb370c488d5cd5593ae4a259fd451bd07`
- Megatron-LM: `84eb19fabfe85cb40663068fc70e205614e10b51`
- Hardware: one 8xB300 node (`tj-w6172jq`; GPUs report as NVIDIA L20D)
- Model: complete GLM-5.3 FP8 checkpoint, all 78 layers
- Shape: TP1 / PP1 / CP8 / EP8, HybridEP, native-FP8 expert storage
- Training: LoRA r32, full uniform recompute, sequence length 131,072
- Protocol: one unprofiled control, followed by one memory-profiled step
- Runtime profile: not collected

Both windows completed with finite loss and gradient norm. The first control
absorbed one-time DSA/cuDNN setup (`77.6s`); the profiled step was stable
(`15.2s`, 1,075 tokens/s/GPU). The trainer was stopped after the snapshot and
all GPUs were freed.

## Result

The global memory peak is caused by the LM-head/loss boundary, not the MoE
sublayer internal activation spike.

| Quantity | Per-GPU memory |
|---|---:|
| Absolute peak allocated | 160.42 GiB |
| Absolute peak reserved | 165.45 GiB |
| Profile-window incremental peak | 37.54-37.58 GiB across ranks |
| LM head live at global peak | 18.91-18.95 GiB |
| Checkpoint residuals live at global peak | 14.63 GiB |
| DSA/attention live at global peak | 3.63 GiB |
| MoE internals live at global peak | 0 GiB |
| Maximum MoE-internal live set anywhere | 14.25-17.64 GiB (mean 15.55 GiB) |
| Total incremental live memory at worst-rank MoE maximum | 34.45 GiB |

The LM-head peak occurs at the end of forward while all checkpoint residuals
remain live. The MoE maximum occurs later during recompute/backward, after the
LM-head buffers have been released. Even the worst-rank MoE moment is about
3.1 GiB below the global incremental peak.

## LM-head breakdown at peak (rank 0)

| Allocation | Memory |
|---|---:|
| Four retained chunk loss/logit buffers plus one in-flight FP32 buffer | 11.82 GiB |
| One prepared FP32 base head weight | 3.55 GiB |
| In-flight FP32 base projection output | 2.36 GiB |
| In-flight BF16 LoRA delta | 1.18 GiB |
| Hidden chunk and small adapter buffers | 0.05 GiB |
| **Direct LM-head total** | **18.95 GiB** |

At CP8, each GPU processes 16,384 tokens. The 4,096-token chunk size creates
four retained full-vocabulary buffers, each approximately 2.36 GiB at TP1.
PR #1265's weight hoist is present on this main commit: only one 3.55 GiB FP32
head-weight copy exists, rather than one per chunk.

## MoE maximum breakdown

MoE memory varies by rank because routing is imbalanced. Rank 1 had the largest
MoE-internal live set at 17.64 GiB:

| Allocation group | Memory |
|---|---:|
| HybridEP dispatch/permutation buffers | 3.71 GiB |
| Routed-expert grouped-linear outputs | 6.18 GiB |
| Routed-expert GLU/intermediate activations | 4.94 GiB |
| Materialized BF16 expert weights from native FP8 storage | 2.25 GiB |
| Shared-expert, router, and small adapter buffers | 0.56 GiB |
| **MoE-internal total** | **17.64 GiB** |

Rank 0's corresponding MoE maximum was 14.39 GiB. Across all ranks the range
was 14.25-17.64 GiB.

## Integrity

Each rank recorded approximately 122,500 allocations and the same number of
free requests/completions. Profile-window active memory returned to baseline;
there is no live-tensor leak in this step.
