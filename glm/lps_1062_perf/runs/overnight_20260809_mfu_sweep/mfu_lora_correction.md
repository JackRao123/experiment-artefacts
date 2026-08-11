# MFU/HFU correction table — LoRA-corrected convention (2026-08-09, fibonacci)

**Mandate:** Jack, 2026-08-09 ("we're doing LoRA, not full fine-tuning … make it
correct"). **Method:** `runs/overnight_20260809_mfu_sweep/mfu.py` (rewritten this date; pure functions
`mfu3x(tps_per_gpu, seq_len, lora_rank)` / `hfu(..., full_recompute=True)`).
**Scope of this file:** derived corrections only — `runs/overnight_20260809_mfu_sweep/results/*.json`
are primary measurement data and are NOT edited; convert them with the
×-factors below or recompute from their own `tps_per_gpu`/`seq_len` via mfu.py.
NOTEBOOK.md cells are updated by pascal (single writer) from this table.

## What changed and why

The pre-2026-08-09 convention charged a flat 3× forward (useful) / 4× forward
(HFU, full recompute) — the full-fine-tuning pass model. Under LoRA the frozen
base weights run the full forward but a **dgrad-only backward** (the wgrad GEMM
is guarded on `weight.requires_grad` in every linear impl in the stack — TE
`_Linear`/`_LayerNormLinear`/`GroupedLinear`, mcore `LinearWithFrozenWeight`;
verified at the bench-commit dependency pins, trainers @ 0e0b65a6). Attention
and indexer ops keep a full 2× backward (their inputs are all activations).
The audit (00:35, `mfu_audit.md`) constants fixes are also folded in (21 full
indexer layers, not 22; indexer params only on those layers, wq_b from
q_lora_rank → active params 40.30 B, not 42.13 B).

  useful/token   = 2·F_matmul + 3·F_attn(L) + 3·F_lora(r)
  executed/token = 3·F_matmul + 4·F_attn(L) + 4·F_lora(r)   (full recompute)

  F_matmul = 80.60 GF (2 × 40.30 B active params; frozen)
  F_attn(L) = 10.47 GF (DSA top-2048 selected attn, length-indep)
             + 172,032 × (L/2) (indexer scoring, 21 full layers, causal avg)
  F_lora(r)  = 2·r·Σ(in+out) over the pinned GLM-5.2 target set
             = r × 12.098 MF  (r=32 → 387 MF; targets: q_down/q_up/kv_down/o
             ×78, dense-MLP fc1/fc2 ×3, shared-expert fc1/fc2 ×75, LM head —
             verified against lora_targets.py @ 0e0b65a6; the notebook's
             "attention-only LoRA" shorthand was wrong)

Peak unchanged: 2.5e15 FLOP/s/GPU (B300 dense bf16, LPS-1062 convention).
Ratios between rows are method-independent; raw tok/s/GPU is unchanged.

## ×-factors by seq_len (multiply a published old-convention value by)

| seq_len | FWD old→new (GF/tok) | mfu3x × | hfu × |
|---:|---:|---:|---:|
| 16,384 | 96.2 → 92.9 | 0.686 | 0.756 |
| 32,768 | 97.7 → 94.3 | 0.690 | 0.759 |
| 65,536 | 100.6 → 97.1 | 0.698 | 0.765 |
| 131,072 | 106.5 → 102.7 | 0.712 | 0.775 |
| 262,144 | 118.3 → 114.0 | 0.736 | 0.793 |

(Decomposition: audit constants ≈ ×0.96; LoRA pass structure ≈ ×0.72–0.83,
length-dependent because attention keeps the 2× backward while matmuls drop
to 1×. The notebook's "×0.96 for audited absolute" rule covered only the
first stage — published absolute MFU was ~30% high, not ~4%.)

## Per-label conversions (old published → corrected)

tps = mean-of-mains tok/s/GPU the published figure was computed from
(method-independent). Values in %; mfu3x = useful, hfu = executed (full
recompute). Recomputed via `mfu3x(tps, L, 32)` / `hfu(tps, L, 32)`.

| label | L | tps | old mfu3x | **mfu3x** | old hfu | **hfu** |
|---|---:|---:|---:|---:|---:|---:|
| A131-repro1 | 131,072 | 623 | 8.0 | **5.7** | 10.6* | **8.2** |
| A131-repro2-w2 (A/B anchor) | 131,072 | 630 | 8.1 | **5.7** | 10.7 | **8.3** |
| A131-patched-ABF (B+F) | 131,072 | 687 | 8.8 | **6.3** | 11.7 | **9.1** |
| A131-patched-ABF-v2 | 131,072 | 672 | 8.6 | **6.1** | 11.5* | **8.9** |
| A-anchor-262k (== exp06 repro) | 262,144 | 630 | 9.0 | **6.6** | 11.9 | **9.5** |
| C-D-262k-d2 | 262,144 | 300 | 4.3 | **3.1** | 5.7 | **4.5** |
| A-131k-d4 (F1 thrash) | 131,072 | 54 | 0.7 | **0.5** | 0.9 | **0.7** |
| C-D-131k-d4 (main1 steady) | 131,072 | 353 | 5.0*† | **3.2** | 6.7*† | **4.7** |
| A131-131k-d4 (golden@131K) | 131,072 | 620 | 7.9 | **5.6** | 10.6 | **8.2** |
| **B-131k-d4 (headline)** | 131,072 | 691 | 8.8 | **6.3** | 11.8 | **9.1** |
| A131-65k-d8 | 65,536 | 632 | 7.6 | **5.3** | 10.2 | **7.8** |
| A131-32k-d16 | 32,768 | 641 | 7.5 | **5.2** | 10.0 | **7.6** |
| A131-131k-d2 (half step) | 131,072 | 649 | 8.3 | **5.9** | 11.1 | **8.6** |
| C-E131-131k-d4 | 131,072 | 395 | 5.0 | **3.6** | 6.7 | **5.2** |
| C-E131-65k-d8 | 65,536 | 320 | 3.9 | **2.7** | 5.2 | **3.9** |
| C-G-131k-d4 (4-node winner) | 131,072 | 665 | 8.5 | **6.1** | 11.3 | **8.8** |
| B-65k-d8 (F5 thrash) | 65,536 | 116 | 1.4 | **1.0** | 1.9 | **1.4** |
| B-32k-d16 | 32,768 | 501 | 5.9 | **4.1** | 7.8 | **5.9** |
| C-G-65k-d8 (F5 thrash) | 65,536 | 108 | 1.3 | **0.9** | 1.7* | **1.3** |
| B-16k-d32 (customer shape) | 16,384 | 734 | 8.5 | **5.8** | 11.3 | **8.5** |
| C-G-16k-d32 (contaminated) | 16,384 | 339 | 3.9 | **2.7** | 5.2 | **3.9** |
| Aug-6 baseline (q480z53) | 262,144 | 446 | 6.3 | **4.7** | 8.4 | **6.7** |
| exp00 (Aug-7 anchor) | 262,144 | 416.5 | 5.9 | **4.4** | 7.9 | **6.3** |
| exp00b | 262,144 | 445 | 6.3 | **4.7** | 8.4 | **6.7** |
| exp04 | 262,144 | 417 | 5.9 | **4.4** | 7.9 | **6.3** |
| exp04b | 262,144 | 431 | 6.1 | **4.5** | 8.2 | **6.5** |
| exp05a | 262,144 | 464 | 6.6 | **4.9** | 8.8 | **7.0** |
| exp05b | 262,144 | 559 | 7.9 | **5.8** | 10.6 | **8.4** |
| exp05c | 262,144 | 583 | 8.3 | **6.1** | 11.1 | **8.8** |
| exp05d | 262,144 | 597 | 8.5 | **6.2** | 11.4 | **9.0** |
| exp05e | 262,144 | 603 | 8.6 | **6.3** | 11.5 | **9.1** |
| exp06 (Aug-7 ship) | 262,144 | 629 | 8.9 | **6.6** | 11.9 | **9.4** |

Steady-state references (synthesis/HANDOFF/PROTOCOL quote these):

| reference | L | tps | old mfu3x | **mfu3x** | old hfu | **hfu** |
|---|---:|---:|---:|---:|---:|---:|
| exp06 steady ~660 | 262,144 | 660 | 9.4 | **6.9** | 12.5 | **9.9** |
| B-131k-d4 steady ~745 | 131,072 | 745 | 9.5 | **6.8** | 12.7 | **9.8** |
| C-G-131k-d4 steady ~725 | 131,072 | 725 | 9.3 | **6.6** | 12.4 | **9.6** |

## Notes

- \* old hfu reconstructed where the notebook cell was "—" or differed;
  corrected values are authoritative regardless (same tps inputs).
- † C-D-131k-d4: the notebook's 5.0%/6.7% does not match its own stated 353
  t/s even under the old convention (4.5%/6.0%) — transcription slip, likely
  copied from the C-E131-131k-d4 (395 t/s) row. Corrected from tps=353.
- HANDOFF.md:17-18's "mfu3x ~9.5%" (745 steady @131K) corrects to **6.8%**
  (not ~7.3% as estimated in passing — 745 × 227.6 GF ÷ 2.5 PF = 6.78%).
  HANDOFF.md itself is left as a dated artifact per laplace.
- Real-data caveat (THD packing): per-doc indexer context is bounded by doc
  length, so on packed customer data the attention term — and only that term —
  drops; recompute per doc length with `useful_flops_per_token(L, r)` and
  token-weight (bench_driver2c's custmix path does this). Synthetic single-
  sequence benches above are exact.
- HFU here is the analytic pass-model estimate; `mfu.py`'s `hfu()` docstring
  lists the assumptions and the empirical (profiler/NCU) measurement route.

## Code change record (integrity; tree is unversioned)

Requested by pascal for verification. Files modified by fibonacci on
2026-08-09, complete list:

1. `runs/overnight_20260809_mfu_sweep/mfu.py` — full rewrite (old API: `mfu(tps, seq_len, hw_passes)`,
   `fwd_flops_per_token(seq_len)`). Pre-rewrite content archived verbatim at
   `runs/overnight_20260809_mfu_sweep/mfu_pre_lora_20260809.py`.
2. `runs/overnight_20260809_mfu_sweep/bench_driver2.py` — 3 exact-string edits:
   - `from mfu import mfu` → `from mfu import hfu, mfu3x`
   - argparse: `--hw-passes (type=float, default=4.0, "fwd-equivalent passes
     actually run (4=full recompute)")` → `--lora-rank (type=int, default=32,
     "LoRA rank of the run (mfu.py: adapter FLOPs scale with rank)")`
   - `"mfu3x": mfu(tps_per_gpu, args.seq_len, 3.0),` /
     `"hfu": mfu(tps_per_gpu, args.seq_len, args.hw_passes),` →
     `"mfu3x": mfu3x(tps_per_gpu, args.seq_len, args.lora_rank),` /
     `"hfu": hfu(tps_per_gpu, args.seq_len, args.lora_rank),`
3. `runs/overnight_20260809_mfu_sweep/bench_driver2c.py` — 4 exact-string edits:
   - `from mfu import mfu, fwd_flops_per_token` → `from mfu import (
     PEAK_FLOPS_GPU, executed_flops_per_token, fwd_flops_per_token, hfu,
     mfu3x, useful_flops_per_token,)`
   - argparse: same `--hw-passes` → `--lora-rank` replacement as above
   - custmix block: `eff_fwd = sum(L * fwd_flops_per_token(L) ...)` with
     `mfu3x = tps_per_gpu * 3.0 * eff_fwd / 2.5e15` /
     `hfu = tps_per_gpu * args.hw_passes * eff_fwd / 2.5e15` → per-doc
     token-weighted `eff_fwd`/`eff_useful`/`eff_executed` (lora_rank-aware),
     `"mfu3x": tps_per_gpu * eff_useful / PEAK_FLOPS_GPU`,
     `"hfu": tps_per_gpu * eff_executed / PEAK_FLOPS_GPU`, and the json gains
     `eff_useful_flops_per_token` / `eff_executed_flops_per_token` keys
   - non-custmix call sites: same replacement as runs/overnight_20260809_mfu_sweep/bench_driver2.py
4. Doc edits (this date): REPORT.md (3 MFU cells + dated note), README.md
   (headline MFU range + pointer), PROTOCOL.md (line-12 figure + MFU
   reporting section), NOTEBOOK.md by pascal from this table.
   `runs/overnight_20260809_mfu_sweep/__pycache__/*.pyc` regenerated as a py_compile side effect
   (derived bytecode only).
