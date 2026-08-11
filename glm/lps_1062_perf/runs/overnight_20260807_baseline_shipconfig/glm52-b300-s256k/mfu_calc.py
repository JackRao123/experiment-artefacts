#!/usr/bin/env python3
"""MFU math for the GLM-5.2 B300 profile (LPS-1062).

Model dims from zai-org/GLM-5.2-FP8 config.json. FLOP convention:
forward = 2*N_active (matmuls) + attention terms; training step with FULL
activation recompute = 4x forward (fwd + recompute-fwd + 2x bwd).
"""

# ── model dims (HF config) ──────────────────────────────────────────
H = 6144
LAYERS = 78
DENSE_LAYERS = 3          # first_k_dense_replace
MOE_LAYERS = LAYERS - DENSE_LAYERS  # 75
N_HEADS = 64
Q_LORA = 2048
KV_LORA = 512
QK_NOPE = 192
QK_ROPE = 64
V_HEAD = 256
IDX_HEADS = 32
IDX_DIM = 128
IDX_TOPK = 2048
N_EXPERTS = 256
TOPK = 8
MOE_INTER = 2048
DENSE_INTER = 12288
VOCAB = 154880
SEQ = 262144

# ── parameter counts ────────────────────────────────────────────────
attn = H * Q_LORA + Q_LORA * (N_HEADS * (QK_NOPE + QK_ROPE)) \
    + H * (KV_LORA + QK_ROPE) + KV_LORA * (N_HEADS * (QK_NOPE + V_HEAD)) \
    + (N_HEADS * V_HEAD) * H
indexer = H * (IDX_HEADS * IDX_DIM) + H * IDX_DIM  # q proj + k proj (approx)
router = H * N_EXPERTS
expert = 3 * H * MOE_INTER
moe_layer = router + N_EXPERTS * expert + 1 * expert  # +1 shared expert
dense_mlp = 3 * H * DENSE_INTER
embed = VOCAB * H

total_params = MOE_LAYERS * (attn + indexer + moe_layer) \
    + DENSE_LAYERS * (attn + indexer + dense_mlp) + 2 * embed
active_params = MOE_LAYERS * (attn + indexer + router + (TOPK + 1) * expert) \
    + DENSE_LAYERS * (attn + indexer + dense_mlp) + embed  # lm_head counts, embed lookup doesn't

print(f"attn params/layer        : {attn/1e6:10.1f} M")
print(f"indexer params/layer     : {indexer/1e6:10.1f} M")
print(f"MoE params/layer         : {moe_layer/1e9:10.2f} B")
print(f"total params             : {total_params/1e9:10.1f} B")
print(f"active params/token      : {active_params/1e9:10.2f} B")

# ── FLOPs per token (forward) ───────────────────────────────────────
f_matmul = 2 * active_params
# DSA selected attention: QK^T (qk dims) + AV (v dim) over top-2048 keys
f_attn_sel = LAYERS * (2 * IDX_TOPK * (QK_NOPE + QK_ROPE) * N_HEADS
                       + 2 * IDX_TOPK * V_HEAD * N_HEADS)
# DSA indexer scoring: full-history causal on ~every-4th "full" indexer layer
# (indexer_types: 3 initial full, then 1-in-4 full) -> ~22 full layers
FULL_IDX_LAYERS = 22
avg_ctx = SEQ / 2  # causal average
f_indexer = FULL_IDX_LAYERS * 2 * avg_ctx * IDX_DIM * IDX_HEADS
f_forward = f_matmul + f_attn_sel + f_indexer

print(f"\nfwd matmul FLOPs/token   : {f_matmul/1e9:10.1f} GF")
print(f"fwd DSA attn FLOPs/token : {f_attn_sel/1e9:10.1f} GF")
print(f"fwd indexer FLOPs/token  : {f_indexer/1e9:10.1f} GF  (causal-avg, {FULL_IDX_LAYERS} full layers)")
print(f"fwd total FLOPs/token    : {f_forward/1e9:10.1f} GF")

# ── MFU ─────────────────────────────────────────────────────────────
TPS = 7133.96          # measured main-window tokens/s (16 GPUs)
TPS_NO_IDX = TPS       # same measurement, alternate FLOP accounting
train_flops_tok = 4 * f_forward
train_flops_tok_no_idx = 4 * (f_matmul + f_attn_sel)
achieved = TPS * train_flops_tok
achieved_no_idx = TPS_NO_IDX * train_flops_tok_no_idx

for name, peak in [("B200-class 2.25 PF", 2.25e15), ("B300 ~2.5 PF", 2.5e15)]:
    mfu = achieved / (16 * peak)
    mfu_b = achieved_no_idx / (16 * peak)
    print(f"\nvs {name} dense BF16:")
    print(f"  achieved (with indexer) : {achieved/1e15:6.2f} PF/s cluster, {achieved/16/1e12:5.1f} TF/s/GPU -> MFU {100*mfu:5.1f}%")
    print(f"  achieved (no indexer)   : {achieved_no_idx/1e15:6.2f} PF/s cluster, {achieved_no_idx/16/1e12:5.1f} TF/s/GPU -> MFU {100*mfu_b:5.1f}%")
