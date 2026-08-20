# GATE-1 SOURCE MEMO — DSA top-k CP-geometry dependence + M=N per-microbatch state (gauss, 2026-08-13 ~03:4x CDT)

Source read at the exact pins tonight's boxes run: vendored mcore
`57efae08b` (clean tree = tonight's behavior; the CPFS dirty-stack hunks are
all env-gated and were verified DISABLED ×16 ranks in tonight's boot logs) and
the M=N runner at the 73c24b00 lineage (wt-pp2-packing @ 8b0ef108).
File:line citations throughout. Nothing executed on boxes; no code changed.

## Verdict summary

- **Q1 (primary):** The DSA top-k selection machinery is **geometry-normalized
  by design** — pool, order, and causal mask are all constructed in global
  packed-document order regardless of CP size. There is **no code path** by
  which CP8 vs CP16 directly changes the selection pool, the score formula, or
  tie-breaking. The geometry dependence enters **indirectly**: top-k is a
  discontinuous function of fp32 scores, the CUDA training path applies **no
  tie-break bias**, and upstream ULP-level score noise (shape-dependent kernel
  reduction orders — the credible entry channel, see §3) flips genuine
  near-boundary selections. Each flip changes the layer output by
  block-content magnitude, and every subsequent layer re-amplifies. This
  mechanistically grounds the "structural sensitivity" reading and predicts
  exactly the observed **diffuse, everywhere, large** per-token diffs.
- **Q1b (EP8 vs EP16):** router top-k is per-token and geometry-free given
  identical inputs; the combine reduction order over a token's experts is set
  by the routing-map permutation (EP-count-independent). EP enters through the
  same discontinuity-amplifier channel (router top-k flips on ULP noise), not
  through any reduction-order change in the combine.
- **Q2 (M=N cross-microbatch state):** every per-microbatch state carrier I
  can find in the path is either freshly constructed per microbatch or
  explicitly cleared. **No cross-microbatch state channel found in source.**
  The one latent trap (DSA top-k holder falls back to the *global config
  object* if packed_seq_params and attention_mask are both None) does not fire
  on the THD path. Details §4; residual unprovable-from-source class flagged
  in §5.

## 1. The DSA top-k path is geometry-normalized (the full chain)

1. **Pool = full sequence.** The indexer's key tensor is all-gathered across
   the CP group and reordered to global packed order before scoring:
   `k = gather_from_sequence_parallel_region(k, group=cp_group)` then
   `k = k.index_select(0, kv_reorder_idx)` — dsa.py:1984-1990. The reorder
   index undoes the zigzag rank order:
   `key_reorder_idx = torch.argsort(gathered_key_positions)` —
   dsa_layout.py:249-261, whose docstring states the gathered order is
   rank-concatenated local-packed and the permutation "restores those gathered
   KV tensors to global packed order" (dsa_layout.py:231-238).
2. **Query positions are absolute packed positions**, built per microbatch
   from that microbatch's own cu_seqlens
   (`build_packed_allgather_cp_local_positions`, dsa_layout.py:134-215) —
   front chunk + mirrored back chunk per rank, values in global document
   coordinates. A token at document position p has position p at any CP size.
3. **Causal mask from absolute positions, ratio 1.**
   `_causal_seq_lens(q_positions, ratio, sk)` = `(q_positions+1)//ratio`
   (dsa_cudnn_kernels.py:413-415) with `_INDEXER_RATIO = 1`
   (dsa_cudnn_kernels.py:130) — per-query causal key count over the full
   gathered sequence; geometry-independent given absolute positions.
4. **Score formula fixed-order.** Scores = Σ_heads relu(q_h·k)·w_h in fp32,
   heads accumulated in fixed index order
   (dsa_cudnn_kernels.py:561-573). Deterministic given identical q/k/w values.
5. **Selection over the normalized pool.** `index_scores.topk(topk_k, dim=-1)`
   (naive path, dsa.py:600-603) / the cuDNN wrapper
   (dsa_cudnn_kernels.py:463-486). Same scores + same order ⇒ same selection.

So: **given bitwise-identical layer inputs, CP8 and CP16 produce
bitwise-identical selections.** No chunking/gather-order term enters the
selection semantics.

## 2. Where geometry dependence actually enters: the discontinuity amplifier

- **No tie-break on the CUDA path.** The deterministic position-based
  tie-break bias is explicitly gated OFF on CUDA:
  `_use_dense_indexer_topk_tie_break` returns `... and not scores.is_cuda`
  (dsa_cudnn_kernels.py:440-449, comment: "keep the expensive full-matrix
  tie-break off the CUDA training path"). So exact/near ties resolve by
  kernel-implementation order — and, more importantly, **near-ties resolve
  differently the moment upstream ULP noise perturbs the scores.**
- The selection is discontinuous: a ULP-level score difference at the top-k
  boundary flips a genuine selection, and a flipped selection changes the
  attention output by the block's content magnitude — not by a ULP. Layer L+1
  then computes its own scores from the perturbed output and re-amplifies.
  78 layers of this is a chaos amplifier whose output is diffuse,
  everywhere, order-1e−1 per-token logprob differences — the observed
  signature (mean-clean datums carrying ~0.08 mean |Δlogprob| per decile).
- **EP side:** the same amplifier exists in the MoE router —
  `torch.topk(logits, k=topk, dim=1)` per token (router.py:258,262) flips
  expert assignments on ULP noise. The combine itself is order-stable across
  EP sizes: unpermute restores token order via
  `reversed_local_input_permutation_mapping` + `routing_map`
  (token_dispatcher.py:862-885), i.e. the per-token expert reduction follows
  the global routing map, not the EP rank layout; the a2a moves bytes without
  reducing. So EP8→EP16 changes *which ranks hold an expert* (bytes moved)
  but not the reduction order.

## 3. The entry channel for the ULP noise (flagged: not fully settleable from this repo's source)

The layout logic is normalized, so the seed noise must enter through
**shape-dependent kernel reduction orders**: at CP8 each rank's GEMM/attention
kernels run with m = 16,384 local rows vs 8,192 at CP16, and FP8/TE/cuDNN
kernels re-tile/re-split reductions by shape — producing ULP-different outputs
for identical logical inputs at every linear layer. This is standard
kernel-behaviour territory and **cannot be confirmed from this repo's source**
(it lives inside TE/cuDNN/FlashMLA kernel selection). What I can confirm from
source is that no *algorithmic* path reintroduces geometry: the only
CP-size-dependent construction in the whole DSA path is the **pad-row
position assignment** (`pad_start = cu_seqlens[-1] + cp_rank*output_size`,
dsa_layout.py:195-204) — and tonight's Gate-1 comparison is zero-padding, so
it is not in play (worth remembering for padded comparisons).

Also noted for completeness: the DSA **indexer loss** is reduced over the CP
group (dsa.py:1935-1940) — a loss-accounting geometry dependence, not a
per-token logprob one.

## 4. Q2 — M=N per-microbatch state inventory (73c24b00 lineage)

The M=N path makes one schedule call per step with
`num_microbatches = len(microbatches)` and `data_iterator=iter(microbatches)`
(training_runner.py:370-413). Per-slot state audit:

| candidate | verdict | evidence |
|---|---|---|
| packed_seq_params carrier | **fresh object per microbatch** | constructed inside the per-partition loop, packer.py:134-176 (`PackedSeqParams(...)` at :167, appended per partition at :177-197) |
| DSA top-k sharing holder | per-microbatch, keyed by source layer | carrier = packed_seq_params when present (dsa.py:1595-1599); skip layers read `topk_holder[self.source_layer]` with a loud cross-PP guard (dsa.py:1961-1973). **Latent trap:** if packed_seq_params AND attention_mask are both None the carrier falls back to `self.config` — a process-global object (dsa.py:1599). Does not fire on the THD path (packed_seq_params always present), but any future non-THD DSA caller inherits a cross-microbatch/cross-step top-k leak. Worth an upstream assert. |
| RouterReplay globals | set per op, cleared in `finally` | training_runner.py:394-417; inactive unless R3 replay mode (not tonight) |
| dispatcher forward state | explicitly cleared after combine | `_clear_forward_state(...)` wipes hidden_shape/routing_map/permutation/splits/d2h_event (token_dispatcher.py:894-903); the clearing commit 922ebcbdb is an ancestor of 57efae08b (verified) |
| forward_data_store → partition mapping | per-microbatch, forward order | training_runner.py:418-420 + the count assert at :420-427 |
| grad buffer | accumulates across slots **by design** (grad accumulation); `_sum_over_microbatches` cancels the schedule's /M division | training_runner.py:385-393 |
| RNG | dropout=0 ⇒ no RNG consumed in forward; checkpoint RNG fork/restore is per-chunk exact | — |
| FIX B/F caches (box dirty tree) | gated OFF tonight (all 7 gates verified DISABLED in boot logs); carriers are per-microbatch anyway | DISPATCHER_CACHE_ARCHAEOLOGY.md §7 |

**Conclusion for Q2:** no source channel for slot-dependent systematic drift.
The schedule treats every slot identically (same kernels, same per-slot
params); slot 1's inputs depend only on slot 1's data.

## 5. What this says about the observed signature — and what I cannot settle

- The **diffuse everywhere-diffs** are fully consistent with the
  discontinuity-amplifier (§2): expected, structural, not a bug. This
  corroborates the "structural sensitivity" reading of the ~0.19 gap.
- The **systematically negative, escalating d1→d5 shift concentrated in
  partition/slot 1** is *not* a chaos signature (chaos is ~zero-mean). Source
  constrains the channel: each slot's positions/masks derive from that slot's
  own cu_seqlens built by the packer (packer.py:144-176), so a
  slot-localized, position-structured, one-signed shift points at the
  **data/packing alignment for that slot** (what documents land in slot 1 and
  how their deciles line up between the two runs), not at shared machinery
  state. This is exactly what poincare's permuted-doc pass discriminates —
  my source read says the drift should follow the *documents*, and if it
  follows the *slot* instead, the only remaining suspects are the
  not-settleable-from-source classes below.
- **Flagged as not settleable from source alone:** (a) the §3 kernel-tiling
  ULP entry channel (inside TE/cuDNN); (b) any stale-buffer read through
  caching-allocator reuse across slots (no evidence in source, but absence
  of evidence — a kernel reading recycled memory would produce slot-correlated
  drift and only an empirical pass like poincare's rules it out); (c) whether
  the two compared runs' packers produced datum-identical slot contents
  (a packing/alignment question, not a kernel one).

### Confidence ranking

1. Geometry-normalized selection chain (§1) — high, direct source.
2. Discontinuity amplifier + tie-break-off-on-CUDA (§2) — high, direct source.
3. No cross-microbatch state in the M=N path (§4) — high for the enumerated
   classes; the self.config fallback trap is real but latent-only.
4. Kernel-tiling as the ULP entry channel (§3) — moderate; consistent with
   all evidence, not provable from this repo.
5. Slot-1 signature ⇒ data/packing alignment (§5) — moderate; source
   constrains the channel, poincare's pass decides empirically.
