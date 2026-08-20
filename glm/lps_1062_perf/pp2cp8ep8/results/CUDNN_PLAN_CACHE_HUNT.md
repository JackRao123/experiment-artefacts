# CUDNN PLAN/GRAPH CACHE HUNT — the fix-campaign critical path (gauss, 2026-08-13 ~07:4x CDT)

Jack-ordered hunt. Target: cauchy's refined hypothesis — cuDNN frontend
execution-plan/graph caching keyed WITHOUT the per-call segment layout.
Source: the exact packages on box w56lorq's venv (snapshotted Mac-side at
`pp2cp8ep8/kernel_src_snapshot/`, sha a94233ff6ca8b3db; cudnn frontend
1.26.0+dsatopk1, flash_mla 1.0.0+b7643bd). Read-only throughout.

## Verdict up front

**The coarse-key/stale-plan hypothesis is NOT confirmed — it is refuted at
the design level.** Every plan/compile cache in the path keys on
code-generating parameters only, and that is CORRECT by construction: the
per-call segment layout (cu_seqlens contents, max_seqlen) flows as *runtime
kernel inputs*, and the kernels are layout-dynamic by design. A later
larger-or-different-layout microbatch does NOT reuse a plan built for a
predecessor's layout — it either reuses a layout-dynamic kernel (indexer
forward) or gets a freshly compiled one (top-k, keyed on shape). The defect,
if it lives in these packages, is in the dynamic handling itself (in-kernel
varlen arithmetic or scheduling), not in any cache key. The cache-key
extension / cache-disable fixes are therefore NOT indicated.

## 1. The cache inventory and exact keys (all of them)

| cache | site | exact key | layout contents in key? |
|---|---|---|---|
| `IndexerTopK` object cache | `indexer_top_k/api.py:150`, key at :174-184 | (dtype, **n_rows**, **num_cols**, seq_lens.numel, top_k, next_n, return_val, num_copy_bits) | NO — seq_lens contents are runtime |
| cute_dsl top-k inner cache | api.py:59-60, 122-125 | next-power-of-2(num_cols) | NO |
| compactify compile cache | `indexer_top_k/compactify.py:123-129`, key at :159 | (cols,) — rows are a runtime Int32 | NO |
| indexer fwd SM100 compile cache | `indexer_forward/_interface.py:29`, key at :150-164 | (q/k/w/out dtypes, head_dim, qhead_per_kv_head, ratio, m/n_block_size, q/kv_stage, is_varlen, has_q_causal_offsets) | NO — explicit comment: "sm_scale/seqlens are runtime args, excluded to avoid spurious recompiles" (:146-149) |
| indexer fwd SM90 compile cache | `_interface_sm90.py:25,123` | same pattern | NO (not tonight's hardware — B300 is SM100) |
| sparse-attn bwd SM100 compile cache | `sparse_attention_backward/_interface_sm100.py:133-134` | (dtype, head_dim, head_dim_v, num_head, block_tile, max_topk, has_topk_length) | NO |
| flash_mla sparse prefill | `flash_mla_interface.py:176-220` | **none** — `sparse_prefill_fwd` computes scheduling per call | n/a |
| cudnn `graph_cache` decorator | `graph.py:8-29` | user-supplied key_fn | no users in the DSA modules (only experimental moe_grouped_matmul, not tonight's path) |

## 2. Why the keys legitimately exclude the layout (the dynamic-by-design chain)

- `to_cute_tensor` marks layouts dynamic before compile
  (`utils/tensor_conversion.py:6-20`).
- The SM100 indexer kernel derives sequence extents at RUNTIME — in the
  varlen path from the runtime `max_seqlen_q/k` Int32 args, else from the
  dynamic tensor shapes — "so that a single compilation works for any
  sequence length" (`indexer_fwd_sm100.py:362-376`).
- The per-document bounds are read from the cu_seqlens CONTENTS in-kernel at
  runtime: `SeqlenInfoQK.create` reads `mCuSeqlensQ[batch_idx]` /
  `mCuSeqlens[batch_idx+1] - offset` per batch (`utils/seqlen.py:47-86`).
- TMA descriptors are built over the dynamic layouts
  (`indexer_fwd_sm100.py:250-291`), so the copies follow the runtime shapes.
- The top-k path RE-COMPILES per shape (n_rows/num_cols in the object key,
  api.py:174-184) — a larger microbatch gets a fresh kernel object, not a
  stale one.
- flash_mla's sparse path has no cross-call cache at all; the sched-meta
  reuse caution (flash_mla_interface.py:80) applies to the kvcache path,
  which tonight's runs don't use.

## 3. What this eliminates / what survives

Eliminated: the "plan reused across layouts because the key lacks
cu_seqlens" mechanism. Note this was already weakened by tonight's padded
evidence: padded runs have uniform shapes across slots, so a shape-keyed
cache cannot produce slot-specific corruption there.

Surviving (reranked):
1. **An in-kernel bug in the dynamic varlen/scheduling path** — the runtime
   cu_seqlens consumption (`SeqlenInfoQK`, utils/seqlen.py:47-86) and the CLC
   persistent tile scheduler (`indexer_fwd_sm100.py:353+`). RESOLVED the one
   open sub-item: `mark_layout_dynamic(leading_dim)` semantics verified in
   the cutlass DSL source (nvidia_cutlass_dsl 4.5.2,
   `cutlass/cute/runtime.py:206-232`) — **all modes' shapes are dynamic, and
   strides are dynamic except the leading dim's (pinned stride-1, which holds
   for the `.contiguous()` tensors the mcore glue passes)**. So no
   static-shape/stride bake exists at the tensor-descriptor layer either.
   What remains unverifiable from source: in-kernel correctness of the
   runtime varlen arithmetic and the CLC scheduler — the empirical
   discriminators (reference leg, cache-clear probe) own those now.
2. **The atomic/work-stealing scheduling class** (e.g.
   indexer_top_k_decode_varlen.py:469's atomic-counter stealing — decode
   path; whether tonight's varlen path uses an atomic scheduler is the open
   sub-question).
3. The visible mcore glue (cleared in SLOT_FOLLOWS §4c) and the
   data/measurement path (poincare's domain).

## 4. Fix proposals, ranked by cost (as assigned)

1. **Reference-path leg (cheapest, config-only, landing):**
   `dsa_kernel_backend` "cudnn"→"none"/unfused (glm52_dsa.py:105-110 sets it
   today) forces BOTH the indexer (pure `fused_qk_topk_naive`,
   dsa.py:563-609) and the attention (`_unfused_absorbed_dsa_fn`,
   dsa.py:147-171) off the compiled packages entirely — no caches in play at
   all. The clean discriminator.
2. **One-shot cache-clear diagnostic (box, doppler's discretion):** clear
   `_compile_cache` / `_cache_of_IndexerTopKObjects` between two same-shape
   microbatches and watch the corruption pattern. Recompile-per-clear is
   minutes — a diagnostic, not a run mode.
3. **Cache-key extension: NOT indicated.** The keys already cover every
   codegen-relevant parameter; adding layout contents would only multiply
   recompiles with zero correctness benefit if the kernels are
   layout-dynamic (they are, per §2).
4. **Env-level cache-disable switch: NONE EXISTS TODAY** in the DSA path
   (the only env toggles in the package are `CUDNN_FE_GROUPED_GEMM_*` in the
   grouped-GEMM modules — a different, unused-tonight path — and
   `DENSE3WG_USE_WARP_SPLIT_PROD` in an SM90 score-recompute module). Adding
   one is a trivial upstream change (guard the two dict reads) but is a code
   change, not a "today" switch. The fastest cache-free execution that exists
   today is item 1's config knob.

## 5. Handoff note

If the reference leg convicts the packages, the defect is in the in-kernel
dynamic varlen/scheduling handling (§3.1/3.2) — that is an upstream
(cudnn-frontend `+dsatopk1` build / FlashMLA b7643bd) debug item, and the
snapshot at `kernel_src_snapshot/` is the exact source. The two-line gate
split (indexer-only vs attention-only reference) noted in
SLOT_FOLLOWS_SOURCE_MEMO.md §4d separates the two packages if needed.
