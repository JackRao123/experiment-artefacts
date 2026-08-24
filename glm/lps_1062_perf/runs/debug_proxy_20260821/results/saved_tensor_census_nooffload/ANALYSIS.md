# Saved-Tensor Census Analysis — debug proxy, no offload

Source data: `saved_tensors.rank0.pid55676.call2.json` (also call0/call1 in this dir).
Capture: 1 microbatch, seq 8192, **1 transformer layer** (debug proxy) + chunked loss head.
Rank 0, no offload, no recompute. 576 activation saves, 16.24 GB logical
(unique-storage bytes ≈ logical → no aliasing/double-count).

Note: the seq-4096/4095 pair in the head is **one microbatch whose loss is chunked
into 4096+4095** (= 8191 targets after the causal shift), not two microbatches.

## Top-level buckets

| GB | % | Bucket | Scales with |
|---|---|---|---|
| 7.61 | 46.9% | A. fp32 LM-head weight saved as activation (`chunked_lm_head._project_logits` → `linear_with_frozen_weight`; LoRA frozen base weight), 2 × `[154880,6144]` fp32 (once per loss chunk) | ×1 w.r.t. layers |
| 5.07 | 31.2% | B. fp32 full-vocab logits saved by TE `parallel_cross_entropy`: `[1,4096,154880]` + `[1,4095,154880]` | ×1 |
| 1.79 | 11.0% | C. MLA/DSA attention saves (23 events) | ×~75 layers |
| 1.47 | 9.0% | D. MoE saves (539 events; 512 are empty `[0]` placeholders, zero bytes) | ×~75 layers |
| 0.20 | 1.2% | E. layer input `[8192,6144]` (×75) + final block output to head (×1) | mixed |
| 0.10 | 0.6% | F. head-chunk hidden states + loss misc | ×1 |

Real-model scaling (same seq, no recompute): per-layer (C+D+½E) ≈ **3.4 GB/layer × 75
≈ 250 GB**; head (A+B+F+½E) ≈ **12.8 GB fixed**. The debug proxy inverts the real
profile (head = 78% here vs ~5% at scale). A 2-MoE-layer proxy without the head
measures the right regime (per-layer dominance) only because it omits the head.

Head-side notes (not the optimization target here, but recorded):
- A is the fp32 LM-head *weight* saved once per loss chunk (2 chunks → 2 × 3.8 GB).
- B is what `chunked_lm_head` is supposed to avoid materializing; TE CE saved the
  full `[seq, 154880]` fp32 logits per chunk (chunked_lm_head.py:326).

## C — Attention decomposition (1.79 GB)

**1.22 GB (68%) is a single `save_for_backward`** in the fused DSA kernel
(`FusedSparseAttentionFunc`, dsa_cudnn_kernels.py:2613):

| MB | Tensor | Shape | What it is |
|---|---|---|---|
| 604 | `q_flat` | `[8192, 64, 576]` bf16 | absorbed queries, 64 heads × (512 + 64 rope) |
| 537 | `out_flat` | `[8192, 64, 512]` bf16 | attention output before V up-projection |
| 67 | `global_idxs` | `[8192, 2048]` int32 | DSA top-k indices (2048 keys/query) |
| 9.4 | `kv_flat` | `[8192, 576]` bf16 | shared KV latent (tiny — the MLA win) |
| 2.1 | `lse` | `[8192, 64]` fp32 | log-sum-exp |
| ~0 | `attn_sink` | `[64]` fp32 | per-head sink |

Remaining 0.57 GB — projection plumbing in absorbed_mla.py:

| MB | What | Shape / site |
|---|---|---|
| 268 | o_proj input (64 heads × 256 v-dim flattened) | `[8192,1,16384]`, absorbed_mla.py:927 |
| 201 | hidden states saved **twice**: q_down_proj + kv_down_proj inputs | 2 × `[8192,1,6144]`, lines 459/480 |
| 34 | q_proj input (post-norm q_compressed) | `[8192,1,2048]`, line 538 |
| 42 | q/kv layernorm inputs | `[8192,2048]` + `[8192,512]`, lines 516/518 |
| 17 | `up_v_weight` saved as activation by the V-up einsum (dsa.py:173 — einsum is not a fused linear) | `[64,512,256]` |
| ~6 | RoPE pairs 4 × `[8192,1,1,64]`, LoRA rank-32 intermediates, sinks | lines 629/639 |

## D — MoE decomposition (1.47 GB real bytes)

**Routed experts: 1.07 GB (73%)** — 4 × `[65536, 2048]` bf16 from the SwiGLU in
experts.py (`glu` / `bias_act_func`). 65536 = 8192 tokens × top-8; 2048 = expert ffn
hidden per EP rank. The four saves are the GLU chunk pair (`x_glu`, `x_linear` from
`torch.chunk`) plus activation in/out kept for backward.

**Shared expert: 134 MB** — fc1 input `[8192,1,6144]` (100.7 MB,
shared_experts.py:262) + fc2 input `[8192,1,2048]` (33.6 MB, :323).

**Router/dispatch: ~260 MB** — MoE layer input `[8192,1,6144]` (100.7 MB), router
scores 2 × `[8192,256]` fp32 + top-k bool mask, dispatch saves 3 × `[8192,1,2048]`
(moe_layer.py:498), routing metadata 2 × `[8192,513]` int32.

The 512 empty `[0]` events come from the grouped-GEMM expert path (zero-token
experts / dispatch placeholders). Zero bytes, but they are per-save *events* —
relevant if offload is dispatched per event (queue overhead, not memory).

## Optimization levers (at ×75: C ≈ 134 GB, D ≈ 110 GB)

1. **Routed-expert GLU (~80 GB at ×75) is the biggest single lever.** The hook
   already exists: experts.py:790–805 wraps `bias_act_func` in
   `CheckpointWithoutOutput` + `moe_act_manager` (offload delayed until after
   linear_fc2) when `activation_recompute` is on. This census is the no-recompute
   baseline where the full 1.07 GB/layer stays saved.
2. **Hidden states `[8192,1,6144]` saved ≥3×/layer** (q_down in, kv_down in, MoE in;
   4× with shared-expert fc1) → ~25 GB at ×75 of the same tensor. Dedup by storage
   identity if the offload machinery keys on it.
3. **Kernel `q_flat` + `out_flat` (~85 GB at ×75)** are inherent to the fused DSA
   backward (re-read there); offloading those two is the attention-side lever.
   Everything else in attention is small change.
