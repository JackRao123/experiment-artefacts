# mfu.py audit vs zai-org/GLM-5.2-FP8 (2026-08-09)

Sources: HF `config.json` (zai-org/GLM-5.2-FP8), transformers
`modeling_glm_moe_dsa.py` / `configuration_glm_moe_dsa.py` (main).

## Verdict

**FWD(L) is systematically overstated by ~4.0–4.2% at all lengths — inside the
~5% tolerance, but the bias is one-directional (MFU numbers are ~4% too high).**
Two errors, both in the indexer:

1. **Indexer params counted in all 78 layers with the wrong wq shape.**
   Only the 21 `"full"` layers have an indexer (`indexer_types`: shared layers
   set `self.indexer = None` and reuse the previous full layer's top-k indices).
   And the indexer's `wq_b` projects from `q_lora_rank` (2048), not `hidden`
   (6144): real per-full-layer indexer params = 2048·4096 (wq_b) + 6144·128 (wk)
   + 6144·32 (weights_proj) = 9.37 M, vs the script's 25.95 M × 78 layers.
   Active params: **40.30 B actual vs 42.13 B in script** (Δ 1.83 B → 3.66 GF/tok).
2. **FULL_IDX_LAYERS = 21, not 22.** `indexer_types` has exactly 21 `"full"`
   entries (layers 0,1,2 then every 4th from 6: 6,10,…,74). `index_topk_freq=4`
   / `index_skip_topk_offset=3` merely generate this pattern; they do NOT make
   any layer use dense attention — all 78 layers do top-k-2048 selected attention.

Everything else checks out (all matmul dims, DSA attention term, causal L/2,
embed/LM-head handling).

## Constants: script vs config

| constant | script | actual (config) | ok |
|---|---|---|---|
| H | 6144 | hidden_size 6144 | ✓ |
| LAYERS | 78 | num_hidden_layers 78 | ✓ |
| DENSE_LAYERS | 3 | first_k_dense_replace 3 (mlp_layer_types: 3 dense + 75 sparse) | ✓ |
| N_HEADS | 64 | num_attention_heads 64 | ✓ |
| Q_LORA | 2048 | q_lora_rank 2048 | ✓ |
| KV_LORA | 512 | kv_lora_rank 512 | ✓ |
| QK_NOPE / QK_ROPE | 192 / 64 | 192 / 64 (qk_head_dim 256) | ✓ |
| V_HEAD | 256 | v_head_dim 256 | ✓ |
| IDX_HEADS / IDX_DIM / IDX_TOPK | 32 / 128 / 2048 | index_n_heads 32 / index_head_dim 128 / index_topk 2048 | ✓ |
| N_EXPERTS / TOPK / shared | 256 / 8 / 1 | n_routed_experts 256 / num_experts_per_tok 8 / n_shared_experts 1 | ✓ |
| MOE_INTER / DENSE_INTER | 2048 / 12288 | moe_intermediate_size / intermediate_size | ✓ |
| VOCAB | 154880 | vocab_size 154880 | ✓ |
| FULL_IDX_LAYERS | 22 | **21** (count of "full" in indexer_types) | ✗ |
| `_indexer` params | H·(32·128)+H·128 = 25.95 M ×78 layers | **wq_b: Q_LORA·(32·128) + wk: H·128 + weights_proj: H·32 = 9.37 M ×21 layers** | ✗ |

## Formula checks

- **Embed/LM head**: tie_word_embeddings=false (untied). Script counts
  `_embed = VOCAB·H` once in ACTIVE_PARAMS; the ×2 makes it exactly the LM-head
  matmul (1.90 GF/tok), and the embedding lookup is FLOP-free. Correct as is.
- **`_attn` (MLA)**: q_a (H→2048), q_b (2048→64·256), kv_a+k_rope (H→576),
  kv_b (512→64·448), o (64·256→H) = 165.0 M/layer. Matches the modeling code. ✓
- **DSA selected attention**: 2·topk·256·64 (QKᵀ) + 2·topk·256·64 (AV)
  = 134.2 M/layer ×78 = 10.47 GF, length-independent. Structure correct.
  (Tiny overcount for positions < 2048 that attend to fewer keys: ~0.3 GF at
  L=32K, less at longer L. Ignore.)
- **Indexer scoring**: full layers score q_idx·k_idx over the whole causal
  context → 2·IDX_DIM·IDX_HEADS per ctx token, avg L/2. Structure correct;
  only the layer count (21 vs 22) is off. Omitted weights-combine matmul adds
  21·2·32·(L/2) ≈ 0.18 GF at 262K — negligible.
- **MTP**: num_nextn_predict_layers=1 exists in the checkpoint but is unused
  in SFT — correctly excluded.
- **LoRA note (convention, not FWD error)**: with frozen base weights, backward
  is ~1× fwd (dgrad only; wgrad only for tiny adapters), so "useful passes" for
  LoRA is ≈2, not 3, and HFU under full recompute ≈3, not 4. mfu3x is fine as a
  stated cross-table convention but overstates truly-useful FLOPs for LoRA.

## PEAK = 2.5e15

Internally consistent (all LPS-1062 tables use it) and clearly documented in
the docstring, so ratios/comparisons are unaffected. But note: the commonly
published HGX B300 dense BF16 figure is **2.25 PF/GPU** (same as B200;
Blackwell Ultra's uplift is FP4-only); public sources for GB300 scatter over
2.25–2.8 PF. Under a 2.25e15 convention every MFU here would read ×1.11
higher. Keep 2.5e15 for continuity, but it is a chosen convention, not the
datasheet number (moderate confidence — B300 BF16 reporting is inconsistent).

## Corrected numbers

Active params/token: **40.30 B** (script: 42.13 B).
F_matmul = **80.60 GF** (script 84.25); F_dsa_attn = 10.47 GF (unchanged);
indexer = 21·2·128·32·(L/2) = 172,032·(L/2) FLOPs/tok.

| L | script FWD (GF/tok) | corrected FWD | script over by |
|---|---|---|---|
| 32,768 | 97.67 | **93.88** | +4.0% |
| 65,536 | 100.63 | **96.70** | +4.1% |
| 131,072 | 106.53 | **102.34** | +4.1% |
| 262,144 | 118.34 | **113.61** | +4.2% |

MFU values computed with mfu.py are therefore ~4% high (multiply by ~0.96 to
correct); cross-length and cross-config comparisons are essentially unaffected
since the bias is nearly constant in L.
