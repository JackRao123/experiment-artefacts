#!/usr/bin/env python3
"""MFU / HFU for Kimi-K2.7-Code LoRA SFT on B300 — LoRA-corrected, Kimi arch.

Same convention as the GLM-5.2 mfu.py (LPS-1062): under LoRA the frozen base
weights run fwd + dgrad only (no wgrad), adapters run full fwd + 2x bwd:

  useful/token   = 2*F_matmul + 3*F_attn(L) + 3*F_lora(r)
  executed/token = 3*F_matmul + 4*F_attn(L) + 4*F_lora(r)   (full recompute)

Architecture deltas vs GLM-5.2 (verified vs HF config 2026-08-15):
  - 61 layers (1 dense + 60 MoE), H=7168, 64 heads
  - MLA: q_lora 1536, kv_lora 512, qk_nope 128, qk_rope 64, v_head 128
  - 384 routed experts top-8 + 1 shared, moe_inter 2048; dense_inter 18432
  - vocab 163840
  - NO DSA: plain MLA FULL causal attention -> attn FLOPs scale with L/2
    (GLM's DSA is length-independent top-2048 + a small indexer term).

LoRA target set (lora_targets.py KimiK25VL branch): q_down, kv_down, o_proj
in all 61 layers + LM head (untied). NOT q_up/kv_up (vLLM absorb), NOT the
MLPs/experts.
"""

# --- architecture constants (HF config moonshotai/Kimi-K2.7-Code, 2026-08-15) ---
H = 7168
LAYERS = 61
DENSE_LAYERS = 1
MOE_LAYERS = LAYERS - DENSE_LAYERS  # 60
N_HEADS = 64
Q_LORA = 1536
KV_LORA = 512
QK_NOPE = 128
QK_ROPE = 64
V_HEAD = 128
N_EXPERTS = 384
TOPK = 8
N_SHARED = 1
MOE_INTER = 2048
DENSE_INTER = 18432
VOCAB = 163840

PEAK_FLOPS_GPU = 2.5e15  # B300 dense bf16 (LPS-1062 convention)

# --- base-model parameter counts -> F_matmul ---
_attn = H * Q_LORA + Q_LORA * (N_HEADS * (QK_NOPE + QK_ROPE)) \
    + H * (KV_LORA + QK_ROPE) + KV_LORA * (N_HEADS * (QK_NOPE + V_HEAD)) \
    + (N_HEADS * V_HEAD) * H  # 101.1 M/layer
_router = H * N_EXPERTS
_expert = 3 * H * MOE_INTER
_dense_mlp = 3 * H * DENSE_INTER
_embed = VOCAB * H  # counted once: x2 below is exactly the LM-head GEMM

ACTIVE_PARAMS = MOE_LAYERS * (_attn + _router + (TOPK + N_SHARED) * _expert) \
    + DENSE_LAYERS * (_attn + _dense_mlp) \
    + _embed  # 31.69 B

F_MATMUL = 2 * ACTIVE_PARAMS  # 63.37 GF/token fwd, frozen under LoRA


def attn_flops_per_token(seq_len: int) -> float:
    """Forward attention FLOPs/token: plain MLA full causal attention.
    Per layer: QK^T 2*(L/2)*(qk_nope+qk_rope)*n_heads + AV 2*(L/2)*v_head*n_heads."""
    return LAYERS * 2 * (seq_len / 2) * (QK_NOPE + QK_ROPE + V_HEAD) * N_HEADS


# --- LoRA adapter shapes: (in, out) per adapted matmul (target set above) ---
_LORA_ATTN = [(H, Q_LORA), (H, KV_LORA + QK_ROPE), (N_HEADS * V_HEAD, H)]  # x 61
_LORA_HEAD = [(H, VOCAB)]                                                # x 1
_LORA_IO_SUM = sum(i + o for i, o in LAYERS * _LORA_ATTN + _LORA_HEAD)


def lora_fwd_flops_per_token(lora_rank: int) -> float:
    return 2 * lora_rank * _LORA_IO_SUM


def fwd_flops_per_token(seq_len: int, lora_rank: int) -> float:
    return F_MATMUL + attn_flops_per_token(seq_len) + lora_fwd_flops_per_token(lora_rank)


def useful_flops_per_token(seq_len: int, lora_rank: int) -> float:
    return 2 * F_MATMUL + 3 * attn_flops_per_token(seq_len) \
        + 3 * lora_fwd_flops_per_token(lora_rank)


def executed_flops_per_token(seq_len: int, lora_rank: int,
                             full_recompute: bool = True) -> float:
    if not full_recompute:
        return useful_flops_per_token(seq_len, lora_rank)
    return 3 * F_MATMUL + 4 * attn_flops_per_token(seq_len) \
        + 4 * lora_fwd_flops_per_token(lora_rank)


def mfu3x(tps_per_gpu: float, seq_len: int, lora_rank: int,
          peak_flops_gpu: float = PEAK_FLOPS_GPU) -> float:
    return tps_per_gpu * useful_flops_per_token(seq_len, lora_rank) / peak_flops_gpu


def hfu(tps_per_gpu: float, seq_len: int, lora_rank: int,
        full_recompute: bool = True,
        peak_flops_gpu: float = PEAK_FLOPS_GPU) -> float:
    return tps_per_gpu * executed_flops_per_token(
        seq_len, lora_rank, full_recompute) / peak_flops_gpu


if __name__ == "__main__":
    r = 32
    print(f"active params/token : {ACTIVE_PARAMS/1e9:.2f} B")
    print(f"matmul (frozen base): {F_MATMUL/1e9:.1f} GF/token fwd (const)")
    print(f"LoRA adapters (r={r}): {lora_fwd_flops_per_token(r)/1e6:.0f} MF/token fwd")
    for L in (32768, 65536, 131072, 262144):
        print(f"L={L:>7}: attn {attn_flops_per_token(L)/1e9:6.1f} | "
              f"FWD {fwd_flops_per_token(L, r)/1e9:6.1f} | "
              f"useful {useful_flops_per_token(L, r)/1e9:6.1f} | "
              f"executed(fullRC) {executed_flops_per_token(L, r)/1e9:6.1f} GF/token")
