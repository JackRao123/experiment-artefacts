# TAIL-PAD × DSA SOURCE MEMO — where padding enters, and what can corrupt earlier docs (gauss, 2026-08-13 ~04:3x CDT)

Question from cauchy: where does tail padding (BT_PACK_PAD_TO_MAX) enter the
DSA path, and which entry point corrupts EARLIER docs in a partition with
severity growing toward the tail (the sawtooth)? Source read at mcore
`57efae08b` + trainer 73c24b00 lineage (wt-pp2-packing), TE bindings at
~/Documents/TransformerEngine. No box actions, no code changes.

## Verdict up front

- **Hypothesis B (pad-block selection): REFUTED from source** for real
  queries — pads are causally after their doc's real tokens and
  varlen-confined to their doc at every mask construction site (§2).
- **Hypothesis A (layout misalignment): LIVE** — the corruption class that
  matches the sawtooth is a **three-way CP-split convention consistency**
  requirement: TE's data sharder vs mcore's position-map builder vs the cuDNN
  fused indexer's internal split. A per-doc split mismatch shifts all later
  docs in the contiguous packed layout → accumulates toward the tail → worst
  at the tail-filled doc. I could not settle it from local source (TE's
  kernel is bindings-only here) — empirical probe specced in §4.
- **New finding (loss-level, not logprob-level):** pad query rows are NOT
  masked at the DSA kernel level tonight — `real_token_mask_q` is read at
  dsa_masking.py:392 but set nowhere, so `query_valid_rows=None` and pad rows
  contribute garbage terms to the **indexer loss**. If the parity metric is
  loss-level, part of the +0.05 is this accounting artifact, growing with pad
  volume → tail-filled partitions worst. (§3.4)

## 1. What padding exists tonight (both legs)

- Per-document pad to `pad_multiple` (a `2*cp_size` multiple) is ALWAYS on for
  CP (thd_cp.py:258-259, validated :80-85). Pads carry token 0 / label −100 /
  position 0 / padmask True / load-spread routes (thd_cp.py:288-326).
- The flag adds **tail-fill**: the LAST document's padded region is extended
  so the partition hits a uniform length — `cu_padded[-1] += tail_pad`
  (thd_cp.py:353-387). Two boundary series are maintained throughout:
  `cu_seqlens` (real) and `cu_seqlens_padded` (padded) (thd_cp.py:329-330,
  385, 403-404).
- Every DSA consumer prefers the **padded** series:
  `get_packed_qk_cu_seqlens` (dsa_layout.py:295-317), the RoPE path
  (absorbed_mla.py:437-444), the mask builder (dsa_masking.py:442-455), the
  cuDNN metadata (dsa_cudnn_kernels.py:157-176 → :262-275).

## 2. Hypothesis B refuted — pads cannot be selected by real queries

The full mask chain, with citations:

1. Query positions are absolute padded-packed coordinates, built per
   microbatch from the padded cu_seqlens (dsa_layout.py:134-215 via
   dsa.py:1761-1774).
2. The varlen window for a query at absolute position p is
   `[padded_start_of_its_doc, p]`:
   `generate_varlen_mask_params_for_positions` — `starts =
   cu_seqlens[searchsorted(...)]`, `ends = query_positions + 1`
   (dsa_masking.py:68-80), applied via `build_valid_mask_from_starts_ends`
   (dsa_masking.py:82-98).
3. A doc's pad rows sit at padded positions `[real_end, padded_end)` — strictly
   AFTER every real token of that doc (thd_cp.py:288-289 pads append at the
   doc tail; tail-fill extends the last doc's tail further, :359-373). Causal
   (`ends = p+1`) excludes them from every real query's window; varlen
   (`starts`) confines each query to its own doc.
4. The indexer scores over the gathered+reordered full KV
   (dsa.py:1984-1990), and the top-k pool for a real query is therefore
   real-keys-only. Pad tokens' own garbage outputs are loss-masked at
   extraction (packer.py:187-200 slices `padded_start + real_length`, with
   length asserts at :180-197; unshard is the exact TE-index inverse,
   thd_cp.py:468-487).

So pads do not enter real tokens' attention. B is dead at the mask level.

## 3. Hypothesis A live — the three-way split-consistency channel

Three components must agree EXACTLY on the per-doc zigzag split of the padded
layout:

| component | role | site |
|---|---|---|
| TE `tex.thd_get_partitioned_indices` | data sharding (which rows each CP rank holds) | thd_cp.py:422-435 → TE `attention.cpp:957` (kernel `nvte_cp_thd_get_partitioned_indices` — **not in the local TE checkout, bindings only**) |
| mcore `build_packed_allgather_cp_local_positions` | the DSA position map / KV reorder | dsa_layout.py:134-215; split rule `half_seq_lens = (padded_len//cp)//2`, front+mirrored-back (:182-186) |
| cuDNN fused indexer internal split | the fused kernel's own per-rank varlen split from `packed_cu_seqlens_*`/`packed_cp_size` | dsa_cudnn_kernels.py:262-275, metadata from :157-176 |

Why a mismatch here produces THE SIGNATURE: the packed tensor is contiguous,
so if any two conventions disagree on doc k's split, every later doc's rows
are read at shifted positions — the shift accumulates with each doc's pad →
**corruption grows toward the tail**, and the tail-filled last doc (huge
padded segment) shows it worst. The mcore builder's divisibility guard is
**CPU-only** (dsa_layout.py:166-180: "In CUDA training these lengths are
runtime tensors; checking them here would add a sync") — a mismatch is silent
on the training path. This is my top suspect, and it is exactly the class
that the (refuted) TE-DotProductAttention attribution was reaching for.

Secondary channels checked and cleared or discounted:

3.1 **RoPE — cleared.** Freqs are computed for `max_seqlen` (inflated by
tail-fill: thd_cp.py:386 → rotary_pos_embedding.py:238-241) but applied
per-document with restart/offsets from the same padded cu_seqlens
(rope_utils.py:222-263, CASE 2 per-sequence restart); a token's RoPE phase
depends only on its in-doc position — invariant to the tail-fill.

3.2 **Extraction/stitching — cleared.** `thd_logprobs_to_loss_fn_outputs`
slices padded-starts + real lengths with explicit asserts
(packer.py:180-197); unshard inverts with the same TE indices
(thd_cp.py:468-487).

3.3 **MoE capacity displacement — refuted for tonight's config.**
`moe_expert_capacity_factor` default None (transformer_config.py:861), no
capacity key in the trainer JSON → no drop-and-pad. Pads are routed
(load-spread) but only consume bandwidth, not real-token slots.

3.4 **Indexer loss on pad rows — REAL BUG, LATENT tonight.** `query_valid_rows`
comes from `packed_seq_params.real_token_mask_q` (dsa_masking.py:386-395),
which is set nowhere in mcore or the trainer (grep-verified) → None. BUT the
term is multiplied by zero on tonight's path: the GLM-5.2 DSA LoRA provider
sets `dsa_indexer_loss_coeff = 0.0` (glm52_dsa.py:105-110, overriding the
bridge default 0.001 from glm5_bridge.py:149), and `use_indexer_loss` gates
on `indexer_loss_coeff > 0` (dsa.py:1890-1894) — the indexer loss is not even
computed tonight. The bug fires only on coeff>0 configs (full-parameter DSA).
**Probe (c) result (consistency check, pre-registered ~zero excess): HOLDS** —
loss gap vs logprob-aggregated gap: golden +0.046983 vs +0.045373 (excess
+0.0016); PP2 −0.132920 vs −0.128365 (excess −0.0046). |Excess| ≤ 0.005 both
configs, and the padded-PP2 run-to-run noise floor (range 0.030 over 4
same-config samples) dwarfs it anyway. The parity gaps are ~fully
forward-logprob phenomena; no unaccounted loss-level term. (weierstrass has
the latent-bug work-item line for the morning queue.)

3.5 **max_seqlen inflation → kernel-shape ULP — real but signature-mismatched.**
The tail-filled doc inflates `max_seqlen` to the partition length, which feeds
kernel launch metadata (`packed_max_seqlen_*`, dsa_cudnn_kernels.py:272-273)
and can shift cuDNN/FlashMLA internal tiling for the whole partition — but
that produces diffuse ULP noise, not a position-structured sawtooth. Listed
for completeness; not the sawtooth source.

## 4. Needs an empirical probe (ranked)

(a) **Three-way index-convention diff** — the decisive probe for hypothesis A.
On box (read-only, CPU/tiny-GPU): build one synthetic padded multi-doc
partition through the packer, then compare (i) `tex.thd_get_partitioned_indices`
per rank vs (ii) mcore's `build_packed_allgather_cp_local_positions` per rank
vs (iii) the row order the cuDNN fused path actually consumes (or, cheapest:
dump the real run's per-rank shard indices and the position map and check
`data_row[i]` vs `position[i]` agreement doc-by-doc). Mismatch ⇒ the fix is
in the position builder or the cuDNN metadata; match ⇒ A is dead too and the
corruption is in the kernel-internal varlen handling (cuDNN source needed).

(b) **Uniform-doc padded leg** — pack a partition of equal-length docs with
pad-to-max: per-doc pads vanish, only tail-fill remains. If the sawtooth
vanishes with it, the corruption scales with per-doc pad volume (supports A /
3.4); if it persists at the tail, the tail-fill itself is the corruptor
(points at the cuDNN kernel's handling of the inflated last segment).

(c) **Loss-level vs logprob-level separation** — DONE (see §3.4): excess
≤0.005 at both configs, consistent with the confirmed coeff-0.0 reading; the
indexer-loss term is latent tonight. (The one-field fix — set
`real_token_mask_q` from the packer's existing padmask — remains the correct
hygiene item for coeff>0 configs.)

Probe (a) script staged at `tools/probe_a_split_convention.py` (mcore leg
validated Mac-side on 8 cases; TE leg needs a tiny CUDA alloc — box owner
picks the window).

## 4b. PROBE (a) RESULT (doppler, 2026-08-13 ~06:1x CDT): hypothesis A DEAD

`tools/probe_a_split_convention.py` on the box (worker node, isolated from
the booting trainer): **TOTAL MISMATCHED ROWS: 0, exit 0** — TE
`thd_get_partitioned_indices` == mcore
`build_packed_allgather_cp_local_positions` on every rank, every case
(realA/realB/awkward doc lengths × CP8 + CP16 × tail-fill on/off). The
TE-sharder↔mcore-position-map conventions agree exactly. Per the §4 decision
tree, the residual layout suspect is the **cuDNN fused indexer's internal
split** (not introspectable from user space — needs the cuDNN source or a
kernel-level dump). Full output: box `lps1062_pp2/probe_a_out.log`.

Materiality note for the ship call: the padded path's same-config
nondeterminism (4 samples, range 0.030) is the uninit-memory/race class,
which a *deterministic* split convention — matched or mismatched — cannot
produce. With A dead and B refuted, the corruption evidence now points at
the same kernel/scratch/atomic class that the slot-follows hunt
(results/SLOT_FOLLOWS_SOURCE_MEMO.md §3) converged on independently.

## 5. Confidence ranking + the ship-padded question

1. B refuted (pads never selected by real queries) — high, full chain cited.
2. Extraction/RoPE/capacity cleared — high, cited.
3. Indexer-loss-on-pads real at loss level (3.4) — high from source
   (never-set field); its SIZE needs probe (c).
4. Three-way split mismatch (A) — moderate-to-high as the sawtooth
   candidate: it is the only remaining channel with the accumulate-toward-
   tail structure; unsettled because TE's kernel is bindings-only locally.
   Probe (a) decides.
5. Kernel-shape ULP from max_seqlen inflation — high that it exists, low
   that it explains the sawtooth (diffuse, not position-structured).

**Input to the fix-vs-ship-unpadded call (FINAL, post-wheel-bump):** the slot
defect that dominated tonight's corruption was one raced kernel wheel (the
retired 1.26.0+dsatopk1 frontend's #396 TMEM WAR) — CONFIRMED by the 1.27.0
reproducer (SLOT_FOLLOWS_SOURCE_MEMO.md §4e resolution: PP1/CP8 ×3 mean
12.30481 vs golden 12.30418, spread 0.00077, 25× collapse; variance and
slot-mean died together). With A and B dead and the race convicted, the
padded-vs-unpadded residual (+0.036, borderline) reduces to the padded-leg
re-check on the fixed wheel — if the tail-pad component survives there, it
is small and lands in the latent indexer-loss / kernel-shape-ULP territory
(§3.4/§3.5), not a layout bug. The ship-padded-vs-unpadded call now turns on
the fixed-wheel re-check plus the dynamic-shape p2p deviceSync cost
(p2p_communication.py:263/:419) for unpadded — a much smaller decision than
tonight's hunt suggested.
