#!/usr/bin/env python3
"""ARCHIVAL COPY — mfu.py as it existed before the 2026-08-09 LoRA correction.

This is the pre-rewrite (full-FT convention: flat 3x/4x of forward) version,
preserved verbatim for integrity verification after the unversioned-tree
rewrite. Do not import from this file; the live module is mfu.py. Conversion
between conventions: mfu_lora_correction.md.

--- original docstring below ---

MFU math for GLM-5.2 LoRA SFT on B300 (LPS-1062) — length-dependent revision.

Method (unchanged from mfu_calc.py, 2026-08-07, except the indexer term is now
a function of sequence length instead of fixed at 262,144):

  forward FLOPs/token = matmul (2 * active_params)
                      + DSA selected attention (topk-2048, length-independent)
                      + DSA indexer scoring (causal, scales with context)

  mfu3x = tok/s/GPU * 3 * FWD(L) / PEAK   — "useful" FLOPs (fwd + 2x bwd),
          the convention used in all LPS-1062 notebook tables so far.
  hfu   = tok/s/GPU * hw_passes * FWD(L) / PEAK — hardware passes actually
          run (4 = fwd + recompute-fwd + 2x bwd under full recompute).

Why the length dependence matters: the indexer term is
FULL_IDX_LAYERS * 2 * (L/2) * IDX_DIM * IDX_HEADS — causal-average context L/2.
At 262,144 it is 23.6 GF/token; at 131,072 it halves; at 65,536 it quarters.
The old constant 118.3 GF/token is only correct at L=262,144. With benches now
running at 32K-262K (customer regime), FWD(L) must be evaluated per length:

  L        FWD(L) GF/token
  32,768    97.7
  65,536   100.7
  131,072  106.6
  262,144  118.3  (== old constant)

Caveat (real data vs synthetic): the indexer term assumes one causal document
of length L. Under THD packing of shorter docs, each doc's indexer context is
bounded by its own length, so real-data indexer FLOPs/token are LOWER than
FWD(packed_L). Benches here use single synthetic sequences of length L, for
which FWD(L) is exact (modulo the ~22-full-indexer-layer approximation).
MFU comparisons across lengths use FWD(L); comparisons at fixed L are
unaffected by any of this.

PEAK: B300 dense bf16 = 2.5e15 FLOP/s/GPU (same convention as before).
"""

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

PEAK_FLOPS_GPU = 2.5e15  # B300 dense bf16

# parameter counts (identical to mfu_calc.py)
_attn = H * Q_LORA + Q_LORA * (N_HEADS * (QK_NOPE + QK_ROPE)) \
    + H * (KV_LORA + QK_ROPE) + KV_LORA * (N_HEADS * (QK_NOPE + V_HEAD)) \
    + (N_HEADS * V_HEAD) * H
_indexer = H * (IDX_HEADS * IDX_DIM) + H * IDX_DIM
_router = H * N_EXPERTS
_expert = 3 * H * MOE_INTER
_moe_layer = _router + N_EXPERTS * _expert + 1 * _expert
_dense_mlp = 3 * H * DENSE_INTER
_embed = VOCAB * H

ACTIVE_PARAMS = MOE_LAYERS * (_attn + _indexer + _router + (TOPK + 1) * _expert) \
    + DENSE_LAYERS * (_attn + _indexer + _dense_mlp) + _embed

F_MATMUL = 2 * ACTIVE_PARAMS  # 84.3 GF
F_DSA_ATTN = LAYERS * (2 * IDX_TOPK * (QK_NOPE + QK_ROPE) * N_HEADS
                       + 2 * IDX_TOPK * V_HEAD * N_HEADS)  # 10.5 GF
FULL_IDX_LAYERS = 22  # 3 initial full + 1-in-4 full (indexer_types), approx
F_INDEXER_PER_TOKEN_CTX = FULL_IDX_LAYERS * 2 * IDX_DIM * IDX_HEADS  # x (L/2)


def fwd_flops_per_token(seq_len: int) -> float:
    """Forward FLOPs/token for one causal sequence of length seq_len."""
    f_indexer = F_INDEXER_PER_TOKEN_CTX * (seq_len / 2)
    return F_MATMUL + F_DSA_ATTN + f_indexer


def mfu(tps_per_gpu: float, seq_len: int, hw_passes: float = 3.0) -> float:
    """MFU fraction. hw_passes=3 -> mfu3x (useful FLOPs); 4 -> HFU (full recompute)."""
    return tps_per_gpu * hw_passes * fwd_flops_per_token(seq_len) / PEAK_FLOPS_GPU


if __name__ == "__main__":
    print(f"active params/token : {ACTIVE_PARAMS/1e9:.2f} B")
    print(f"matmul              : {F_MATMUL/1e9:.1f} GF/token (const)")
    print(f"DSA selected attn   : {F_DSA_ATTN/1e9:.1f} GF/token (const)")
    for L in (32768, 65536, 131072, 262144, 524288):
        f = fwd_flops_per_token(L)
        print(f"L={L:>7}: indexer {f - F_MATMUL - F_DSA_ATTN:>6.2e} GF | "
              f"FWD {f/1e9:6.1f} GF/token")
