#!/usr/bin/env python3
"""Estimate MFU and HFU for GLM-5.2 LoRA SFT on B300.

Frozen base weights run forward and input-gradient passes, while attention and
LoRA adapters run full forward/backward passes:

  useful/token   = 2*F_matmul + 3*F_attn(L) + 3*F_lora(r)
  executed/token = 3*F_matmul + 4*F_attn(L) + 4*F_lora(r)  # full recompute

For packed data, compute sequence-dependent FLOPs per document length and take
a token-weighted average. The default B300 BF16 peak is 2.5e15 FLOP/s/GPU and
can be overridden with ``peak_flops_gpu``.
"""

# --- architecture constants (verified vs cached HF config 2026-08-09; see
# --- mfu_audit.md for the audit trail) ---
H = 6144
LAYERS = 78
DENSE_LAYERS = 3
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
FULL_IDX_LAYERS = 21  # indexer_types: layers 0,1,2 then every 4th from 6

PEAK_FLOPS_GPU = 2.5e15  # B300 dense bf16 (convention, see header)

# --- base-model parameter counts -> F_matmul ---
_attn = H * Q_LORA + Q_LORA * (N_HEADS * (QK_NOPE + QK_ROPE)) \
    + H * (KV_LORA + QK_ROPE) + KV_LORA * (N_HEADS * (QK_NOPE + V_HEAD)) \
    + (N_HEADS * V_HEAD) * H  # 165.0 M/layer
# indexer exists ONLY on the 21 full layers; wq_b projects from q_lora_rank
_indexer_full_layer = Q_LORA * (IDX_HEADS * IDX_DIM) + H * IDX_DIM + H * IDX_HEADS  # 9.37 M
_router = H * N_EXPERTS
_expert = 3 * H * MOE_INTER
_dense_mlp = 3 * H * DENSE_INTER
_embed = VOCAB * H  # counted once: x2 below is exactly the LM-head GEMM (lookup is free)

ACTIVE_PARAMS = MOE_LAYERS * (_attn + _router + (TOPK + 1) * _expert) \
    + DENSE_LAYERS * (_attn + _dense_mlp) \
    + FULL_IDX_LAYERS * _indexer_full_layer + _embed  # 40.30 B

F_MATMUL = 2 * ACTIVE_PARAMS  # 80.60 GF/token fwd, frozen under LoRA
F_DSA_ATTN = LAYERS * (2 * IDX_TOPK * (QK_NOPE + QK_ROPE) * N_HEADS
                       + 2 * IDX_TOPK * V_HEAD * N_HEADS)  # 10.47 GF/token, length-indep
F_INDEXER_PER_CTX = FULL_IDX_LAYERS * 2 * IDX_DIM * IDX_HEADS  # per ctx token (x L/2)

# --- LoRA adapter shapes: (in, out) per adapted matmul (target set above) ---
_LORA_ATTN = [(H, Q_LORA), (Q_LORA, N_HEADS * (QK_NOPE + QK_ROPE)),
              (H, KV_LORA + QK_ROPE), (N_HEADS * V_HEAD, H)]  # x 78 layers
_LORA_DENSE_MLP = [(H, 2 * DENSE_INTER), (DENSE_INTER, H)]    # x 3 layers
_LORA_SHARED = [(H, 2 * MOE_INTER), (MOE_INTER, H)]           # x 75 layers
_LORA_HEAD = [(H, VOCAB)]                                     # x 1
_LORA_IO_SUM = sum(i + o for i, o in
                   LAYERS * _LORA_ATTN + DENSE_LAYERS * _LORA_DENSE_MLP
                   + MOE_LAYERS * _LORA_SHARED + _LORA_HEAD)


def attn_flops_per_token(seq_len: int) -> float:
    """Forward attention-side FLOPs/token: DSA selected attention (constant)
    + indexer scoring on the 21 full layers (causal, average context L/2)."""
    return F_DSA_ATTN + F_INDEXER_PER_CTX * (seq_len / 2)


def lora_fwd_flops_per_token(lora_rank: int) -> float:
    """Forward FLOPs/token of the LoRA adapters: x@A (in->r) + x@A@B (r->out)
    per adapted matmul = 2*r*(in+out). ~12.1 MF/token per unit of rank."""
    return 2 * lora_rank * _LORA_IO_SUM


def fwd_flops_per_token(seq_len: int, lora_rank: int) -> float:
    """Forward FLOPs/token (base + adapters) for one causal sequence of
    length seq_len. Exact for a single synthetic doc; see header for packing."""
    return F_MATMUL + attn_flops_per_token(seq_len) + lora_fwd_flops_per_token(lora_rank)


def useful_flops_per_token(seq_len: int, lora_rank: int) -> float:
    """Algorithm-required FLOPs/token per step (fwd + bwd) under LoRA:
    2x matmul (fwd + dgrad) + 3x attn (full bwd) + 3x adapters."""
    return 2 * F_MATMUL + 3 * attn_flops_per_token(seq_len) \
        + 3 * lora_fwd_flops_per_token(lora_rank)


def executed_flops_per_token(seq_len: int, lora_rank: int,
                             full_recompute: bool = True) -> float:
    """Pass-model executed FLOPs/token per step. Full recompute adds one more
    forward: 3x matmul + 4x attn + 4x adapters. With full_recompute=False
    (none) executed == useful; selective recompute sits between — measure it."""
    if not full_recompute:
        return useful_flops_per_token(seq_len, lora_rank)
    return 3 * F_MATMUL + 4 * attn_flops_per_token(seq_len) \
        + 4 * lora_fwd_flops_per_token(lora_rank)


def mfu3x(tps_per_gpu: float, seq_len: int, lora_rank: int,
          peak_flops_gpu: float = PEAK_FLOPS_GPU) -> float:
    """Useful-FLOPs MFU fraction. The '3x' name is kept for continuity with
    the LPS-1062 tables; under LoRA the multiplier is structural (2x matmul +
    3x attn + 3x adapters), not a flat 3x — see header for the convention break."""
    return tps_per_gpu * useful_flops_per_token(seq_len, lora_rank) / peak_flops_gpu


def hfu(tps_per_gpu: float, seq_len: int, lora_rank: int,
        full_recompute: bool = True,
        peak_flops_gpu: float = PEAK_FLOPS_GPU) -> float:
    """HFU fraction — hardware FLOPs actually run.

    Computed here as an ANALYTIC ESTIMATE (the pass model above). It can also
    be measured empirically, which is the exact method: torch.profiler(
    with_flops=True) over one real step — hand-adding custom kernels the
    formulas miss (DSA indexer, flash MLA, fused-MoE grouped GEMMs, chunked
    CE) — or Nsight Compute tensor-core counters. Executed FLOPs/token is
    stable per (config, seq_len), so measure once and reuse as a calibrated
    constant. Assumptions of the analytic estimate: base-weight wgrad skipped
    (verified — see header), flash-attn backward == 2x fwd (kernels are often
    ~2.5x), no EP capacity padding, negligible CP ring-attention boundary
    waste. A measured HFU materially above this estimate localizes execution
    waste; the gap between mfu3x and hfu is the recompute overhead by design."""
    return tps_per_gpu * executed_flops_per_token(
        seq_len, lora_rank, full_recompute) / peak_flops_gpu


if __name__ == "__main__":
    r = 32  # all LPS-1062 configs use LoRA r32
    print(f"active params/token : {ACTIVE_PARAMS/1e9:.2f} B")
    print(f"matmul (frozen base): {F_MATMUL/1e9:.1f} GF/token fwd (const)")
    print(f"DSA selected attn   : {F_DSA_ATTN/1e9:.1f} GF/token fwd (const)")
    print(f"LoRA adapters (r={r}): {lora_fwd_flops_per_token(r)/1e6:.0f} MF/token fwd")
    for L in (32768, 65536, 131072, 262144, 524288):
        print(f"L={L:>7}: FWD {fwd_flops_per_token(L, r)/1e9:6.1f} | "
              f"useful {useful_flops_per_token(L, r)/1e9:6.1f} | "
              f"executed(fullRC) {executed_flops_per_token(L, r)/1e9:6.1f} GF/token")
