# GLM-5.2 LoRA SFT activation annotations

The diagrams show the counterfactual requested: **full activation recomputation disabled**.

## Trainers configuration

- Micro-batch size: `1`
- Activations: BF16 (the FP8 checkpoint is dequantized to BF16 on trainer load)
- Sequence length: `S = 262144`
- B200 recipe: `CP = 32`, `EP = 32`, so `L = S / CP = 8192` local tokens and `A = 8S / EP = 65536` routed token-expert rows per rank on average
- B300 recipe: `CP = 16`, `EP = 16`, so `L = 16384` and `A = 131072`
- Tensor parallelism: `TP = 1`
- Pipeline parallelism: `PP = 1`
- Top-k routed experts per token: `8`
- LoRA rank: `r` is chosen by the training request; the diagrams use `r = 128` only for the example numbers because that is GLM-5.2's configured maximum

## Formula conventions

Let `G = 2^30` bytes per GiB.

- BF16 tensor `[n, d]`: `2nd / G` GiB
- FP32 or INT32 tensor `[n, d]`: `4nd / G` GiB
- BOOL tensor `[n, d]`: `nd / G` GiB
- LoRA low-rank output `[L, r]`: `2Lr / G` GiB

Numbers in diagram callouts are `B200 | B300` at `S = 262144`. They are raw tensor payloads, not allocator reservations.

## What "BWD SAVE" means

The marked tensor must remain available until its backward operation. Aliases are marked so the same storage is not counted twice. CUDA/NCCL workspaces, fragmentation, gradients, parameters, optimizer state, and transient communication scratch buffers are excluded.

The cuDNN DSA save bundle is exact from `ctx.save_for_backward`: absorbed Q, full-sequence compressed KV, top-k indices, attention output, LSE, and a negligible 64-element FP32 attention sink. MoE grouped-GEMM and dispatcher markings are conservative tensor-payload estimates because fused kernels may retain aliases or use additional opaque workspace.

## LoRA targets in a GLM-5.2 MoE layer

- Query down projection
- Query up projection
- KV down projection
- Attention output projection
- Shared-expert packed FC1
- Shared-expert FC2

Routed experts, the DSA indexer, and KV up projection are excluded. The trainer uses absorbed MLA: KV-up weights participate in the math, but the normal KV-up module forward and expanded K/V activations are not materialized.

## Source anchors

- `models/src/loops_models/model_configs/trainer_configs.py`: GLM-5.2 B200/B300 CP and EP recipes
- `server-megatron-bridge/src/trainers_server_megatron_bridge/lora_targets.py`: GLM-5.2 LoRA targets and exclusions
- `server-megatron-bridge/src/trainers_server_megatron_bridge/glm52_dsa.py`: frozen indexer and cuDNN DSA backend
- Megatron-Core `experimental_attention_variant/dsa_cudnn_kernels.py`: exact fused-attention saved tensors
- Megatron-Core `experimental_attention_variant/absorbed_mla.py`: absorbed MLA tensor path
- Megatron-Core `transformer/moe/token_dispatcher.py`: all-to-all dispatch and retained routing metadata
