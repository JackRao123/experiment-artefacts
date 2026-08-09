# FILED 2026-08-08, then CLOSED same day at Jack's decision (workaround carried
# internally instead of upstreaming): issue #3331 and PR #3332 both closed.
# https://github.com/NVIDIA/TransformerEngine/issues/3331
# https://github.com/NVIDIA/TransformerEngine/pull/3332
#   validated H100×4 (torch 2.11.0+cu128, cuDNN 9.25.0.15): TE 2.17.1 FAILS
#   (auto arm: nondeterminism + mis-attention), TE main PASSES (both modes).
#   Side finding: TE 2.17.1 THD backward broken with cuDNN 9.19.0 (BAD_PARAM,
#   fused_attn_f16_arbitrary_seqlen.cu:934); cuDNN 9.25.0.15 resolves it.

# [DRAFT] TE issue: THD + CP(p2p): tail-padded sequences take the `pad_between_seqs=False` fast path → nondeterministic forward row, silently wrong attention (dropped keys, zeroed real rows), nondeterministic gradients

**Affects:** every released TE we checked (2.16.0, 2.17, 2.17.1); `main` is
incidentally unaffected for callers that pass distinct `cu_seqlens` /
`cu_seqlens_padded` tensors (see "main" note below).
**Stack reproduced on:** TE 2.16.0, torch 2.11.0+cu130, cuDNN 9.19.0, sm103;
cuDNN-independent — measured 18-cell matrix: TE {2.16.0, 2.17.1} FIRE 3/3
seeds × 60 iters on every cuDNN in {9.19.0.56, 9.21.1.3, 9.24.0.43,
9.25.0.15} with the wobble at the identical row in 108/108 firing
instances; TE main (2.19.0.dev0+8260f49) quiet 0/3 on all four (loaded
libcudnn verified per cell via /proc/self/maps + cudnnGetVersion). Fires
at CP4 within 20 iters, CP2 more rarely.

## Summary

With `qkv_format="thd"`, `attn_mask_type="padding_causal"`, CP `p2p`, and a
sequence whose REAL length is not divisible by `2*cp_size` (tail-padded, e.g.
real 698 padded to 704 at CP4 — the common case for packed THD training),
`DotProductAttention.forward`'s `pad_between_seqs` auto-detect compares
`cu_seqlens_padded[:-1]` with `cu_seqlens[:-1]` — deliberately ignoring tail
padding after the last sequence. The call therefore takes the
`pad_between_seqs=False` fast path in `attn_forward_func_with_cp`, which
approximates per-ring-step seqlens as `cu_seqlens // cp_size`
(`context_parallel.py`). That approximation mislabels each rank's
chunk-boundary rows whenever `real % (2*cp) != 0`, with three proven
consequences:

1. **Nondeterministic forward output** at exactly one row per affected rank
   (local row `T_LOCAL-2`): the per-step softmax-LSE aux tensor is allocated
   with `at::empty` (`allocateSpace(..., init_to_zeros=false)`,
   csrc/extensions/attention.cpp) and the cuDNN kernel writes only rows the
   (wrong) `cu_seqlens` calls real. The correction kernels
   (`thd_second_half_lse_correction`, `thd_out_correction`) iterate PADDED
   ranges and log-sum-exp-merge the uninitialized rows. The design contract —
   garbage LSE is harmless because `out_per_step == 0` at padding rows
   (explicit zero-guard in `thd_out_correction_kernel`) — breaks at the one
   row that is mislabeled-padding for the diagonal/lower steps but REAL for
   the upper-triangle steps: a real output row gets scaled by
   `exp(lse_real − merge(garbage, lse_real))`.
2. **Silently wrong attention (deterministic):** the mislabeled `cu` also
   drops each rank's last two chunk rows as padding KEYS, so `2*(cp-1)` real
   keys are invisible to every query (measured: 607/698 real rows differ from
   the CP1 reference by >1e-2, mean |Δ| 0.04 on random bf16 data), and real
   QUERY rows at the boundary get zero or partial outputs (`cp-1` real tokens
   receive EXACT-ZERO attention output; more receive upper-steps-only
   contributions).
3. **Nondeterministic gradients on ALL CP ranks:** the merged (garbage-bearing)
   LSE is saved for backward; the dKV ring spreads the contamination, so
   backward is nondeterministic across many rows on every rank — including
   ranks whose forward looks clean.

## Minimal repro

`standalone_te_cp4_repro.py` (attached): pure `te.DotProductAttention`, thd +
padding_causal + CP p2p + MQA (8q/1kv/d128, bf16), one 698-token sequence
padded to 704, identical inputs each iteration, 4 GPUs, ~2 min.
Observed: CP ranks 0-2 produce 3-7 distinct bitwise outputs in 20-30
iterations, always differing at exactly local row 174; rank `cp-1` clean
(it has no upper-triangle steps). `--backward` variant shows grads
nondeterministic on 4/4 ranks. CP1 is deterministic.

## Evidence chain (all element-level, fixed inputs)

- Per-ring-step fingerprints: every q/k/v/cu INPUT and every kernel OUT is
  bitwise stable across iterations; ONLY the per-step LSE varies, at exactly
  the mislabeled rows (diagonal step rows {174,175}; upper-triangle step row
  {87}), then total-LSE columns {174,175}, then out row {174} via the
  upper-step `thd_out_correction` calls only. cuDNN's written values are
  fully deterministic — this is not a cuDNN bug.
- Causality (poison test): overwriting the step-LSE padding rows with a
  constant makes the output DETERMINISTIC: `+1000` → deterministic and
  wrong (row 174 becomes exactly 0, as the scale factor underflows);
  `-1e30` → deterministic (but still missing the never-computed
  diagonal-step contribution at that row).
- Fix validation: passing `pad_between_seqs=True` (engaging the exact
  `get_cu_seqlens_on_cp_rank` path) makes forward AND backward bitwise
  deterministic (100/100 iters × 3 seeds × CP{2,4}) and restores
  correctness: reassembled CP4 output matches CP1 within bf16 rounding
  (max |Δ| 0.0039, no row >1e-2).
- Version matrix (18/18 cells, per-cell loaded-cuDNN proof): TE 2.16.0 and
  2.17.1 fire on ALL of cuDNN 9.19/9.21.1/9.24/9.25; TE main quiet on all
  four — cuDNN is not the variable; the main auto-detect rewrite is the
  difference. `NVTE_FUSED_ATTN_DIRECT_SEQLENS` arms inert (symbol absent
  in all three builds including main).
- Fix validated at trainer scale as well (Megatron-based stack, dense
  0.6B model, TP2×CP4): stock nondeterministic across 10 forwards
  (max |Δ| 0.083); with `pad_between_seqs=True` bitwise-deterministic
  10/10. On a 550B hybrid MoE production model the same defect amplified
  through router top-k flips to |Δ| up to 5 — this is not a cosmetic
  wobble at scale.

## Why `main` is only (accidentally, partially) unaffected — and the asks

`main` rewrote the auto-detect in PR #2898 (merged 2026-07-01; motivation:
`torch.equal` syncs during CUDA-graph capture — NOT a correctness fix for
this bug). The new detect is an identity check: distinct padded/real cu
tensor objects ⇒ `pad_between_seqs=True`. Our call pattern (distinct
tensors) therefore lands on the exact path on main, which is why the matrix
is quiet there.

But main's own comment codifies the blind spot as intended behavior:
"If padded cu_seqlens are the *same object* as the unpadded ones, no real
inter-sequence padding exists (only THD tail padding) -- treat as False."
Under CP p2p that inference is exactly wrong: tail-padding-only is
precisely the case that breaks `cu_seqlens // cp_size` whenever
`real % (2*cp_size) != 0`. Callers that pass the SAME tensor object for
real and padded cu — a pattern used in the wild (e.g. verl, see #2892) —
still reach the broken fast path on main today, with all three defects
below.

Asks:
1. Under CP, tail padding must count: make the detect compare FULL cu arrays
   (or document that THD+CP requires `real % (2*cp) == 0` and assert).
2. Add a regression test pinning these semantics (CP{2,4}, tail-padded THD:
   bitwise-determinism across iterations + equality with CP1 reference), so
   the accidental fix on main can't regress.
3. Consider zero/neg-inf-initializing the LSE aux tensor (or masking padded
   rows post-kernel) as defense in depth — the zero-guard contract in
   `thd_out_correction_kernel` is fragile.

## Prior reports of this symptom (unanswered) and related items

- **#1929 + Megatron-LM #1669** (same reporter, 2025): seq len 7 at CP2,
  padded to 8; TE "internally behaves as if cu_seqlens=[0,6]"; repro shows a
  REAL query row with exact-zero CP output = our defect (b). Filed as a
  usage question, never answered — the mechanism above is the answer.
- **#2892**: documents the same-object aliasing pattern (cu_seqlens passed
  as cu_seqlens_padded) in verl — i.e. main's identity check does not cover
  real callers.
- **PR #3269**: precedent that a wrong pad flag leaves CP padding
  uninitialized ("Honor requested FA padding in CP tests").
- **PR #2596**: the exact-path infrastructure (pad_between_seqs support for
  CP A2A+P2P) that the fix routes to.
- **#2186**: THD+CP second-chunk-tail NaN in backward, fixed in cuDNN 9.18 —
  different mechanism (cuDNN bwd workspace init), same structural site.
  Region bug history.
- **#2448**: adjacent evidence that uninitialized/reused aux memory in THD
  attention produces silent nondeterministic wrongness (CP=1, different
  mechanism).

## Workaround for users on released TE

Pass `pad_between_seqs=True` explicitly for THD+CP calls with tail padding
(public kwarg, available in all affected releases). Cost measured at ~0.3-1.5
ms/call host-side at microbench scale; noise at training scale.
