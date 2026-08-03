# LPS-1003 Issue 2 — verdict: GLM-5.2-FP8 loss bumps

> **RESOLVED 2026-08-01: prefill/launch stream race, fixed (PR basetenlabs/trainers#875; wheel now +dsatopk5 after the 08-02 review — PR875_REVIEW_0803.md). See rearm/NIGHT_0801_FINDINGS.md.**


> **2026-07-31 evening: this file is historical.** Current status — trigger
> (`CUDA_DEVICE_MAX_CONNECTIONS` unset) + mask (=1) proven, all
> consumption-path stream-race and buffer-underwrite mechanisms falsified,
> defect narrowed to the DSA score/attention kernels (indexer-fwd CLC
> scheduler / FlashMLA) — see HANDOFF.md top block and parity/NOTEBOOK.md
> "DAY SESSION 2026-07-31".

**Status: 2026-07-30 NIGHT — the buffer-level localization is now in doubt.**
The symptom characterization below (positional, window-anchored, healing,
prod 4/4) is unchanged and still solid. What changed: the experiments the
evening session specced to catch the `output_indices` under-write have been
run, and they came back **negative under a verified detector** — 0 unwritten
slots over 190.5M rows in-situ (91 shapes, staging control 1.0000 on every
one) plus 48 local prod-geometry cases, with the two candidate-drop code paths
shown safe by construction. See INVESTIGATION.md "2026-07-30 night" and
`probe2/`.

So: mechanism = *some* uninitialized/stale read consumed by the GLM DSA path
remains the best explanation of the symptom, but "cuDNN DSA top-k
`output_indices` torch.empty under-write" is no longer supported evidence and
should not be written up as the identified root cause. The earlier scores-side
candidate is dead twice over (source read + runtime measurement: invalid region
100% -inf over 1.6e12 elements).

Next discriminator is the spec's own fallback, not more devbox work: the E1
audit harness on a **fresh prod session window**, where events fire 4/4 (devbox
is now 0/8). `probe2/` is prod-ready for it. Fix posture meanwhile:
`fill_(-1)` glue hardening is still correct defensive hygiene (the wheel
violates its own api.py:51-54 contract regardless of whether the radix branch
ever under-writes), but it can no longer be claimed as *the* fix. PR #843
(warmup mitigation) stays closed.

## One-paragraph verdict

The loss bumps are a **trainer bug, not a data or model property**: during a
window after every trainer boot or adapter (re)initialization, the forward
pass intermittently produces catastrophically wrong outputs (5–11 nats vs
~0.5 normal) for the **last 1–5 documents of packed THD partitions** — the
docs adjoining the packed row's padding. The corruption is positional (proven
by batch-rotation A/B), decays over the first ops/steps as memory pages get
overwritten, requires no weight change to heal, and fires on both B300 and
B200. A batch containing 1–3 destroyed docs shows up as a 1.5–3× step-NLL
"bump"; sessions look spiky early and stabilize later. Nemotron/Qwen don't
show it because (working hypothesis, consistent with all data) the GLM DSA
sparse-attention path is the consumer of the uninitialized tail region; their
dense paths don't read it.

## Evidence chain (all reproducible; details in INVESTIGATION.md)

1. Not data: bump batches score identically to quiet ones on frozen base
   weights; prod bump values exceed untrained-base NLL.
2. Not recipe/stack replay: byte-exact 32-step training replays (validated
   bit-identical batches vs prod ClickHouse aggregates) are flat, twice.
3. Real and server-side: prod per-step metrics reproduce W&B bumps to 4
   decimals; no foreign ops/retries/checkpoints near bumps.
4. Live capture: fresh prod session bumped at steps 0/2/3; in-pod probe
   caught 30/32 datums normal + adjacent datums destroyed, healing minutes
   later on identical payloads.
5. Position-anchored: rotating the batch moved destruction exactly to the new
   partition tails (predicted victims 15/22 destroyed; 3×-destroyed 4/5/6
   clean when mid-partition).
6. Window-anchored: 4/4 fresh boot/rebuild windows fired on prod pods
   (e02-sg pool nodes); decays within ~2-5 ops. Historical runs: dense early
   bumps decaying over ~50-150 steps on both B300 (ali) and B200 (hyd);
   the "17.7% vs 4.8%" hardware gap is mostly window-position sampling
   (spiky B300 runs never got past the early window).
7. Devbox control (b300-1 pool, hours of same-workload memory history): 0/5
   rebuild windows fired, incl. with prod's expandable_segments allocator —
   memory-history dependence; poison-repro test running to confirm.
8. Substrate: DSA indexer top-k boundary is tie-degenerate everywhere
   (exact score ties every call; ~30% of rows within 1e-3) — explains the
   separate, benign per-token wobble noise floor (~±0.02-0.19 per datum on
   frozen weights) and makes tail docs maximally sensitive to garbage scores.

## Secondary bugs found

- **init_trainer_server rebuild deadlock**: third in-process rebuild wedged
  the prod trainer (per-rank LoRA-init logs then silence, GPUs idle, op never
  completes). Needs its own ticket.
- **Rebuild path skips warmup**: boot runs a warmup fb before accepting ops;
  init_trainer_server rebuild does not — first customer op after a rebuild
  hits the corruption window directly.
- (Prior session, still open) silent LoRA re-init on crash restart + client
  retry masking = LPS-1003 Issue 3.

## ADDENDUM 2026-07-30 afternoon (branch session): defect localized

- **CP exonerated empirically**: Nemotron-3-Ultra (CP4 THD, no DSA), same
  recipe/data, 0 spikes through two fresh-init windows (boot + rebuild),
  step-0 bit-identical across the rebuild. All three GLM cp16 runs: 3-8
  spikes in the same window. Flat non-CP controls unchanged. W&B:
  jackrao-lps1003-compare (full) + jackrao-lps1003 (4-run overlay).
- **Defect localized by code audit** (see CODE_AUDIT_TOPK.md): cuDNN-frontend
  DSA indexer top-k output buffer is torch.empty and never pre-filled; the
  long-row radix write-out can leave slots/rows unwritten; glue accepts any
  non-negative in-range int as a key index. Complete mechanistic fit incl.
  tail-run geometry, ~131k depth bound, healing, boot/rebuild window,
  prod-vs-devbox memory history, and both poison-test nulls (0xFF = -1
  sentinel; zeroed pages → filtered).
  Status: confirmed-by-elimination + mechanism-complete; kernel-level catch
  (kill test / indices dump / in-range-int poison) still pending.
- **PR #843** (main session) = window-closing MITIGATION (full-footprint THD
  warmup at boot + after every rebuild) — correct and ship-worthy, but the
  defect remains reachable. Real fix on top: output_indices.fill_(-1) glue
  hardening + a wheel fix ensuring the radix write-out stores all slots
  (+dsatopk3; +dsatopk2 claimed by #821) + upstream report.

## ADDENDUM 2026-07-30 evening (code-verification session): mechanism narrowed, rival hypothesis dead

- **"Indexer reads past valid keys → garbage scores" (the read-past
  hypothesis, ex-candidate #3; also the premise of the ORIGINAL
  experiment_handoff.md) is refuted by direct source read** — four
  independent walls in the wheel: unconditional -inf prefill of the scores
  output (_interface.py:207-209; sm90 :164), n-block loop clamped to
  per-segment seqlen_k (indexer_fwd_sm100.py:1017-1028), boundary tiles
  masked -inf in registers pre-store (:1168-1177), and a fully-materialized K
  input with no uninit region (index_select, dsa_cudnn_kernels.py:796-797).
  Every score top-k can read is a real q·k product or -inf.
- **Dispatch narrowed**: multi-CTA/dynamic-multi-CTA variants are compiled
  OUT on our entry path (defaults False; static one-CTA-per-row grid); the
  short-row branch full-writes incl. explicit -1 padding. The under-write
  lives in the long-row RADIX write-out of the `large_occupancy` (num_rows >
  148) compiled variant — a variant prod always runs and small-shape tests
  never compile. Exact edge still unproven (dump test / in-range poison).
- experiment_handoff.md was REWRITTEN accordingly (same file, new program);
  its original 14:42 version should not be executed.

## Recommended fixes (in order)

1. **`output_indices.fill_(-1)` before the top-k kernel launch** (glue or
   wheel API layer) + a `topk_length == min(K, window)` sanity signal — the
   buffer is pinned: cuDNN DSA top-k output indices
   (decode_varlen.py:684). Unwritten slots become the documented "-1 / no
   candidate" sentinel and get filtered; under-writes become countable
   instead of silent.
2. **Run full-shape warmup after every adapter (re)build** (not just boot) —
   cheap, burns the corruption window on synthetic data; also mitigates
   whatever residual uninit-consumers remain.
3. **Fix the rebuild deadlock** (or make rebuild = process restart).
4. **Wheel-level fix + upstream report**: make the long-row radix write-out
   store exactly top_k slots per row (vendored `+dsatopk3`; `+dsatopk2`
   claimed by #821) and report to NVIDIA/cudnn-frontend as an api.py contract
   violation (their own announced GLM-5.2 CP-training recipe reaches it —
   Megatron-Bridge discussion 4957). (Scores-side hardening is NOT needed:
   the scores path was verified clean, see evening addendum.)

## Customer guidance (Mudith)

- Bumps do not indicate bad data or a bad model; early-window bumps corrupt
  those steps' gradients but the damage is bounded (LoRA recovered within a
  few steps in every observed run).
- Until the fix ships: expect elevated NLL noise in the first ~100 steps of a
  fresh session; avoid drawing conclusions from early-step loss.
- The B200 path with long runs (like 4q9ex6w, 0.59% spikes over 1529 steps)
  is fine for full training runs today.
