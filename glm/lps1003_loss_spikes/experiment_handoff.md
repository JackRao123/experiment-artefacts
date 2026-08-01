# LPS-1003 Issue 2 — experiment handoff (DSA top-k OUTPUT-INDICES verification)

> **REWRITTEN 2026-07-30 evening.** The original version of this file (written
> 14:42 by a parallel analysis session) specced experiments for the
> "top-k *reads scores* past `seq_lens` → garbage scores win" mechanism
> (its Links 2/3/4). That mechanism was **refuted the same afternoon by direct
> source read of the vendored wheel** — see "Read-side walls" below and
> CODE_AUDIT_TOPK.md. The experiment program below targets the surviving
> mechanism: the top-k **output indices buffer** (`torch.empty`, radix path
> under-write). The reusable harness assets from the original (sitecustomize
> seam, baseline band, run discipline) are preserved.

Purpose: give a fresh agent the exact experiment program to prove the
under-write edge and validate the fix. **This doc is the spec; implementation
is yours to write.**

## Read first (in order)

1. `HANDOFF.md` — unified entry point (bug, fix plan, operational state).
2. `CODE_AUDIT_TOPK.md` — the defect + eliminated candidates.
3. `VERDICT.md` — evidence chain + addenda.
4. `EVIDENCE_INDEX.md` — claim → artifact map (check operational state first;
   it goes stale).

## The mechanism under test (60-second version)

Prod GLM trainers, in a window after boot or `init_trainer_server` rebuild,
intermittently destroy the last 1–5 docs of packed THD partitions (per-datum
NLL 5–11 nats vs ~0.5 healthy), healing over a few ops with no weight change.

Localized defect: the cuDNN-frontend DSA top-k allocates `output_indices` with
`torch.empty` (`indexer_top_k_decode_varlen.py:684`) and never pre-fills it,
despite its own API docstring promising an "initial (-1) state" for unwritten
rows (`indexer_top_k/api.py:51-54`). The long-row radix branch
(`length > top_k`) can leave slots/rows unwritten under some edge; recycled
allocator garbage shows through; the Megatron glue accepts any non-negative
in-window int; FlashMLA attends a wrong (but memory-safe) key set.

**One unproven link remains:** the exact radix-path edge that under-writes.
Everything else is verified in source or matched empirically. The experiments
below either catch the edge in the act (E1/E2) or prove causality around it
(E3) and validate the fix (E4).

## Read-side walls (why the old scores hypothesis is dead — verified in source)

All in the unpacked wheel `nvidia_cudnn_frontend-1.26.0+dsatopk1`:

1. Scores output pre-filled `-inf` unconditionally before every launch:
   `indexer_forward/_interface.py:207-209` (SM100), `_interface_sm90.py:164`.
2. Kernel n-block loop clamped to per-segment `seqlen_k`:
   `indexer_fwd_sm100.py:1017-1028` (`_causal_num_n_blocks`).
3. Boundary tiles per-element masked to `-inf` in registers before store:
   `indexer_fwd_sm100.py:1168-1177` (interior-tile invariant: `:1163-1167`).
4. K input fully materialized — `segmented_k = index_select(...)` writes every
   row (`dsa_cudnn_kernels.py:796-797`); no uninit padding exists in K.

Residue: ≤3 TMA-alignment output columns (`_interface.py:127-138`) sit beyond
the view top-k consumes and map past the causal window → filtered. The
original doc's claim "cuDNN path has no post-hoc -inf masking" was true only
of the *Python* layer — the masking lives inside the wheel.

## Key code facts (output side, verified this session)

- Our entry path (`indexer_top_k_wrapper` → `cute_dsl_topk_wrapper`) compiles
  with `enable_multi_cta=False`, `enable_dynamic_multi_cta=False` (constructor
  defaults, `indexer_top_k_decode_varlen.py:655-663`), persistent scheduling
  off (`:612/:676`), launch = static one-CTA-per-row grid (`:534-541`). The
  multi-CTA machinery (incl. `MAX_NUM_ROWS=512`) is **compiled out** — do not
  chase it.
- The trivial branch (`length <= top_k`) full-writes every slot incl. explicit
  -1 padding (`indexer_top_k_varlen_util.py:468-487`). The under-write can
  only live in the long-row radix branch.
- `large_occupancy = num_rows > 148` is a compile-cache key (`:611`); it
  shrinks the smem candidate buffer (4096 entries @ ≥262144 cols) and enables
  the gmem spill path (`:157-178`). Prod chunks (1024–8192 rows) always
  compile this variant; unit/parity tests (<148 rows) never do. **Instrument
  prod-shaped launches or you are testing a different kernel.**
- Allocator size-binning: a freshly allocated `(rows × 2048) int32` output
  most plausibly reuses a block that last held a *previous top-k indices
  tensor* → expected garbage signature = previous-layer echoes / duplicates /
  wrong-in-range values, not random noise.
- Instrumentation seam (proven pattern, reuse it): env-gated sitecustomize
  wrapping `_indexer_top_k_one_chunk` — see
  `../dsa_topk_ima/scripts/dsa_capture_sitecustomize.py` and
  `dsa_margin_sitecustomize.py` here. Never modify vendored `3rdparty/` files.
- Odd-K calls take a fully-initialized torch fallback
  (`dsa_cudnn_kernels.py:475-486`); GLM-5.2 runs K=2048 (even) → cuDNN path.

## The experiments

### E1 — dump test (passive audit; prod window or devbox repro)

Wrap `_indexer_top_k_one_chunk`; per call log stats of
`tk_result["indices"]`: count of slots ≥ that row's window, duplicates within
a row, and (during a destruction event) full dump of the last row chunk.
Prediction if the mechanism is right: during an event, tail-row chunks show
wrong-in-range values / duplicates / echoes of a previous layer's indices.
Passive, prod-safe (audit adds GPU→CPU syncs — verify baseline first, below).

### E2 — in-range-int poison (THE decisive devbox experiment)

Rerun the whole-GPU poison (`devbox_artifacts/poison_gpus.py`) writing int32
values in **[0, 262144)** instead of 0xFF (0xFF failed because it decodes to
-1, the legitimate sentinel; driver-zeroed pages decode to 0 and are filtered
by `>= starts`). Then fresh boot + standard batch-0 probe (`probe_nll.py` +
`train_bundle_0_31.jsonl.gz`, ×3–4 /forward reps).
Prediction: tail-doc destruction on the devbox — the prod fingerprint on
demand, giving a local repro for fix validation.

### E3 — kill test (causal A/B on a fresh window)

Force the pure-PyTorch odd-K fallback path (monkeypatch
`_indexer_top_k_one_chunk` to route even K through the `torch.topk` branch,
or shim `topk_k` to an odd value one less and pad). Fresh boot/window with
warmup skipped (BT_SKIP_FULL_WARMUP-style env): symptom should vanish vs the
even-K cuDNN path under identical conditions.

### E4 — fix A/B (validation of the actual fix)

Sitecustomize wrapper that `fill_(-1)`s the output before the wheel call
(equivalently: apply the glue-hardening patch on a branch). Two checks:
(a) symptom gone across N fresh boot + rebuild windows (zero datums with
per-datum NLL > 2.0; batch-0 partition map: docs [0-6][7-14][15-22][23-30][31],
destroyed = tails);
(b) with the fill in place, count rows where `topk_length < min(K, window)` —
each is a caught under-write, turning the silent bug into a counter.

## Run order

1. **Boot + E1 audit only.** Baseline cleanliness gate: batch-0 probe must
   stay in the 0.762–0.765 band (five prior clean boots) with the harness ON.
2. **E2 poison boot** → read verdict off per-datum NLLs.
3. If E2 fires: **E4** on the same setup (poison + fill) → destruction gone,
   under-write counter > 0. That pair is the complete proof.
4. **E3** as independent causal confirmation (or fallback if E2 won't fire).
5. Optional prod loop-closure: E1 harness on a fresh prod session window.

## Decision tree

```
E2 in-range poison:
├─ devbox tail-doc destruction → mechanism proven, local repro in hand
│   → E4 fill_(-1): destruction gone + nonzero under-write counter = fix validated
│   → ship glue hardening; wheel fix + upstream report follow
└─ clean → the radix edge needs more than hostile memory
    → E1 dump on a prod fresh window (events are known to fire 4/4 there)
    → if prod dump shows garbage indices: mechanism confirmed, iterate E2
      geometry (chunk rows/cols matching the event's shapes)
    → if prod dump is clean during an event: mechanism wrong — reopen
      CODE_AUDIT_TOPK.md eliminated list (NOT the scores path; that is
      code-refuted above)
```

## Operational notes & cautions (carried over, still valid)

- **Check `HANDOFF.md` → "Live operational state" first** (single source;
  EVIDENCE_INDEX no longer duplicates it). Devbox was `tj-3y0n54q` (2×8 B300,
  ali); artifacts mirrored in `devbox_artifacts/`.
- **Do not touch** Mudith's session `4q9z7xw` / trainer `4w5y6r3`.
- Audits add GPU→CPU syncs — diagnostics only; the baseline gate in run step 1
  is mandatory before trusting any harnessed result.
- Fresh boot per poison mode (allocator state carries over in-process).
- Secondary bugs to not trip over: the 3rd in-process `init_trainer_server`
  rebuild can deadlock the trainer (prefer fresh boots over rebuild cycles);
  the rebuild path skips warmup.

## Definition of done

A deterministic repro of the under-write (E2 destruction → E4 clean, or a
prod E1 dump showing garbage indices), plus fix validation numbers. Then:
update `INVESTIGATION.md` (results), `VERDICT.md` (mechanism → proven),
`EVIDENCE_INDEX.md` (new artifacts), and fold results into the fix PR.
