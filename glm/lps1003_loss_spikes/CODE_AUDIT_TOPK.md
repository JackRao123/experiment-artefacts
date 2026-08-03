# LPS-1003 code audit: the actual defect (branch session, 2026-07-30 afternoon)

> **RESOLVED 2026-08-01: prefill/launch stream race, fixed (PR basetenlabs/trainers#875; wheel now +dsatopk5 after the 08-02 review — PR875_REVIEW_0803.md). See rearm/NIGHT_0801_FINDINGS.md.**


Read-only audit of the CP/THD/DSA forward path, cross-checked against every
empirical constraint from INVESTIGATION.md. Bottom line: **the CP
implementation is clean; the defect is in the cuDNN-frontend DSA indexer
top-k output path, which only GLM's cp>1 multi-document packed branch
reaches.** PR #843 (full-footprint warmup at boot + after rebuild) is a
correct *mitigation* — it closes the corruption window without touching the
defect below.

## The defect (candidate #1 — only survivor, complete mechanistic fit)

Call chain (verified): `_pack_thd_cp_microbatches` (megatron_controller.py:2481)
→ `pack_thd_cp_microbatch` (packing.py:880) → per-doc zigzag shard
(tex.thd_get_partitioned_indices, megatron_controller.py:2527) → DSAttention
with cp_comm_type="allgather" → `run_fused_dsa_attention` (dsa.py:2039) →
**`_indexer_topk_multi_packed_cp_thd`** (dsa_cudnn_kernels.py:732; b=1, local
q=16384 doc-major rows, gathered k=262144) → `_indexer_top_k_wrapper_chunked`
(dsa_cudnn_kernels.py:489) → cuDNN radix top-k → in-bounds filter/compaction
(dsa_cudnn_kernels.py:1114/1047) → FlashMLA sparse fwd.

- The wheel's top-k `output_indices` is allocated `torch.empty`
  (indexer_top_k_decode_varlen.py:684 in the vendored
  `nvidia_cudnn_frontend-1.26.0+dsatopk1` wheel) and never pre-filled, despite
  the wheel's own api.py L51-54 documenting an "initial (-1) state" for
  unwritten rows. REFINED 07-30 evening (verified in source): our entry path
  compiles with `enable_multi_cta=False` / `enable_dynamic_multi_cta=False`
  (constructor defaults :655-663; persistent scheduling off :612/:676) and
  launches a static one-CTA-per-row grid (:534-541) — the multi-CTA machinery
  (incl. ComputeDynamicCTAOffsets.MAX_NUM_ROWS=512, an upstream-hygiene issue
  but inactive here) is compiled out. The trivial branch (length <= top_k)
  full-writes every slot incl. explicit -1 padding
  (indexer_top_k_varlen_util.py:468-487), so the under-write can only live in
  the long-row (length > top_k) RADIX branch: histogram → threshold bin →
  collect + tie-refine through the smem candidate buffer with gmem spill.
  Exact failing edge unproven — the dump test / in-range poison are the
  provers.
- Test-coverage gap: `large_occupancy = num_rows > 148` is a compile-cache
  key (decode_varlen.py:611). It shrinks the smem candidate buffer (4096
  entries @ >=262144 cols) and enables the gmem spill path (:157-178) — the
  same fragile buffer complex as the #814 bug. Prod chunks (1024-8192 rows)
  always compile this variant; unit/parity tests (<148 rows) never do.
- Consumer glue trusts memory: `_compact_valid_topk_indices` (:1049) keeps
  any value ≥ 0; `_topk_in_bounds` (:1127-1130) only rejects values outside
  [starts_q, ends_q) — a window up to ~100k wide for long-context tail
  queries. Recycled plausible ints sail through; FlashMLA attends a garbage
  key set → 5-11 nats. (Even-K cuDNN path :467-469 has NO python-side
  masking, unlike the odd-K torch.topk fallback :475-486 which is fully
  initialized.)

Why every observation follows:
- **Tail anchoring, twice over**: local rows are doc-major (tail rows = tail
  docs identically on every CP rank), and the row-chunk loop's REMAINDER
  chunk (:493-498; 2 GiB / (8·max_segment_k), 512-aligned) covers exactly the
  tail rows — remainder shapes are the ones most likely to take the divergent
  multi-CTA dispatch.
- **Depth bound**: one chunk = chunk_rows × 16 global tokens ≈ 131k for
  max_segment_k=32768 — matches the observed ≤~150k corruption depth.
- **Healing**: identical /forward repeats reuse cached blocks but the
  free-list evolves; unwritten slots progressively land on blocks last
  holding -inf scores (negative int → filtered) or bf16 activations (huge int
  → filtered) → benign. Size-binning corollary: a fresh (rows × 2048) int32
  output allocation most plausibly reuses a block that last held a PREVIOUS
  top-k indices tensor — ~100% plausible in-range ints — which is why events
  are potent despite a random int32 having only ~0.006% chance of landing
  in-window, and why the dump test expects previous-layer echoes.
- **Boot/rebuild window**: `_clear_training_stack`/`_init_training_stack`
  (megatron_controller.py:1966-1976) refill the pool with checkpoint-load +
  quantization staging bytes — arbitrary positive ints, many in-range.
- **0xFF poison null result (the tell)**: 0xFFFFFFFF as int32 = -1 = the
  legitimate invalid sentinel → poisoned unwritten rows look *healthy*.
  Fresh driver-zeroed VA → index 0 fails `>= starts` for every non-first doc
  → devbox immunity. Only plausible-positive-int memory history (prod nodes)
  detonates.
- **GLM-only / CP exonerated**: Nemotron-3-Ultra (CP4 THD, no DSA) ran the
  same recipe through two fresh-init windows: 0 spikes in 30+ steps,
  bit-identical step-0 across a rebuild (GLM's step-0 was 2× base and wobbly).
  All Python-side CP buffers (packing pads, zigzag, allgather-KV, logprob
  stitch, chunked LM head) verified written-before-read. Allgather-KV
  geometry would corrupt doc MIDDLES, not tails — doesn't match; doc-major
  local-row geometry matches exactly.

## Eliminated by this audit

- Python packing/CP/loss path: every pad slot explicitly materialized
  (packing.py:943-997, :1042-1051; megatron_controller.py:734-754;
  chunked_lm_head.py:262-267).
- Backward-only kernel bugs (#396 TMEM, #439 empty-row, bwd index bounds):
  excluded by read-only /forward reproduction.
- Varlen indexer-forward scores buffer (was candidate #3) — killed FOUR ways
  (07-30 evening: read side verified in kernel source, not just the
  interface):
  1. scores output pre-filled -inf unconditionally before every launch
     (_interface.py:207-209 SM100; _interface_sm90.py:164);
  2. kernel n-block loop clamped to per-segment seqlen_k
     (indexer_fwd_sm100.py:1017-1028, `_causal_num_n_blocks`);
  3. boundary tiles per-element masked to -inf in registers before store
     (indexer_fwd_sm100.py:1168-1177; interior-tile in-bounds invariant
     :1163-1167);
  4. the K input has no uninit region at all — `segmented_k` is fully
     materialized by index_select (dsa_cudnn_kernels.py:796-797).
  So every score top-k can read is a real q·k product or -inf; "indexer reads
  past valid keys into uninit padding → garbage scores" has no entry point.
  Footnote: with TMA padding, the fill covers the view only; up to 3 uninit
  padded columns per row remain (_interface.py:127-138) but sit beyond the
  view top-k consumes and map past each segment's causal window → filtered;
  can only mildly pollute candidate ranking.
- Pre-#814 phantom-lane radix bug: real, but events reproduce on the
  +dsatopk1 wheel (candidate #2 applies only to unpinned images).

## The actual fix (vs #843's mitigation)

1. Glue hardening (small, real): `output_indices.fill_(-1)` (and topk_length
   sanity) before kernel launch in `_indexer_top_k_wrapper_chunked` /
   the wheel API — unwritten rows become "no candidates", filtered.
2. Kernel fix: ensure the long-row radix write-out stores exactly top_k
   slots per row (or the API layer pre-fills, honoring its own documented
   "-1 initial state") → vendored wheel rebuild (`+dsatopk3`; `+dsatopk2` is
   claimed by PR #821). Report upstream (NVIDIA cudnn-frontend): this path is
   NVIDIA's own announced GLM-5.2 long-context CP training recipe
   (github.com/NVIDIA-NeMo/Megatron-Bridge/discussions/4957), decode_varlen
   is the only top-k impl the wheel ships (api.py:18-19), and the
   large_occupancy compile variant prod runs is untested at small shapes.
3. Keep #843 (window closure + fixes rebuild-skips-warmup regardless).

## Verification set (cheap, decisive)

- Kill test: force the pure-PyTorch odd-K fallback path on a prod boot window
  (BT_SKIP_FULL_WARMUP=1) → symptom should vanish.
- Local repro: repeat the poison test with IN-RANGE POSITIVE INTS
  (e.g. fill int32 values in [0, 262144)) instead of 0xFF → devbox should
  finally fire.
- Dump test: log tk_result["indices"] for the last row chunk during an event
  → expect duplicates / previous-layer echoes / wrong-in-range values.
- Geometry test: halve _TOPK_WRAPPER_MAX_SCRATCH_BYTES (:132) → max
  corruption depth should halve.
