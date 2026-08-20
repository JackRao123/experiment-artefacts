# PRE-REGISTRATION — blockK warm-pool revival probe (CONDITIONAL)

Author: pauli, 2026-08-14. Status: PRE-REGISTERED, CONDITIONAL — **fires ONLY
if box hours remain after W3/W4 + the P4 soak** (bayes's scheduling rule).
This is a NEW experiment, not a retro-appeal: W1b's DOES-NOT-FIT verdict is
CLOSED and stays closed regardless of outcome. One boot, no retries.

## The question

Does block+K=25 fit at 131k at STEADY STATE once the cold-pool first-window
excess is absorbed? The only existing K=25 measurement is a warmup0-only peak
(265.0 smi / 256.8 torch-reserved / 253.2 allocated, canary PASS) — the run
was terminated before any main window, so steady state is unmeasured. The
memory model (MEMORY_LEG_FINAL_STATUS.md §4) predicts the 265.0 was the
first-window excess, not steady demand.

## Protocol

- Config: `pp2cp8ep8/configs/trainer_pp2cp8ep8_131k_blockK25.json` (mission
  headline config + `recompute: {granularity: full, method: block,
  num_layers: 25}`), mission tree, fixed wheel (cudnn-frontend 1.27.0 — the
  wheel-bump-first rule applies to any fresh box).
- Env: ship NCCL env + BT_TF32_LM_HEAD=1 + BT_SKIP_WARMUP=1 (the trainer
  warmup carries a ~+60 GiB transient base at these shapes — boot#1's OOM
  class; dodging it is why we skip) + BT_PROFILE_RANKS unset (no trace tax on
  a memory read).
- Warm-pool protocol: driver warmup at d1 (`--warmup-datums 1` — the existing
  warmup0 shape, which absorbs the first-window excess), then **d2 mains ×≥4**
  (past the creep plateau; per the W1a pattern the plateau declares by ~step
  9, and at d2 window cadence 4–6 mains suffice).
- Read rule (campaign rule): the fit read is the PLATEAU — poller-max over
  the last two mains, which must agree within ~2 GiB; a window-1 read is not
  a steady read.

## Pre-registered predictions (the model is the test)

- warmup0 reproduction: ~256 torch / ~264–266 smi (the prior run's exact
  256.8/265.0 — a materially different warmup0 number indicts the model
  before the mains read).
- Plateau: **236.7 torch-reserved ± 5 → ~245–250 smi** (steady 236.7 +
  small d2-class creep).

## Bars

- **PASS (fits):** plateau ≤ 250 smi AND ≥ 12 GiB below the warmup0 peak →
  the first-window excess was the binding term; K=25 fits steady with margin;
  per the model, K=24 (246.3 steady smi) and K=23 (251.9) become reachable
  follow-ups.
- **FAIL (structural):** plateau ≈ 265 (within ~3 GiB of warmup0) → the
  stage-1 peak is structural at the ceiling; blockK stays dead at 131k; the
  dial's K floor stands at 25+ with the valve stack as the coverage path.
- **Ambiguous (250–260):** report the plateau number; K=25-only at best; no
  K<25 claim.
- **Fail-closed:** if warmup0 OOMs this boot (run-to-run burst variance), the
  probe ends with no fit claim. One boot total.
- **Canary (every window):** loss in the 12.2–12.4 band + gn comparable to
  the prior run's 12.3242/0.4185. Drift >5e-3 = STOP.
- **Abort-guard deviation (explicit):** the 255 poller line applies to the
  PLATEAU read, NOT the warmup0 transient (expected ~265, over the line by
  design). The run is killed only on an actual OOM or a canary stop.

## Secondary (only if PASS)

d4 perf vs the W1a same-box anchor (847.9 tok/s/GPU): within-boot controls at
±2%, win bar > anchor + 2% (model estimate +4–7% at K=25 from the recompute
refund). This is the blockK revival's standalone prize; the dial-floor
relaxation is the program-relevant one.

## Provenance

Prediction basis: the reconciliation chain in MEMORY_LEG_DECISION.md
(correction notes 1–4) + MEMORY_LEG_FINAL_STATUS.md §4; evidence set
`w1b_evidence/` (CSVs, runlogs, trainer peak lines, W1a anchors).

## VERDICT (hertz, 2026-08-14; curie driving) — FAIL, STRUCTURAL (sharper than the FAIL band)

warmup0 gate: EXACT reproduction (256.68/253.25 torch, 264.3/217.3 poller
vs ref 256.8/253.2, 265.0/218.3; canary loss +0.0015 < 5e-3, gn 0.4247 in
the 0.36-0.49 class) — model not indicted at the gate. Outcome: actual
CUDA OOM on the FIRST d2 main (M=2): leader stage-0 217.3 -> 267.4 GiB in
~20s, OOMs at 266.1-266.8 vs the 267.69 cap, still climbing; worker flat
~264.4; contamination check clean (618 MiB children intrinsic to every
boot incl. the W1b reference). Ruling: the warm-pool hypothesis is
REFUTED — the 265-class warmup0 peak is the M=1 STRUCTURAL demand, not
cold-pool excess; M=2 adds ~50 GiB on stage 0 (pipeline-depth activation
fill) over the ceiling. No plateau exists; the minimal main window is
infeasible; the mission shape runs M=16. W1b's DOES-NOT-FIT stands,
mechanism-explained: the binding term is M-scaling of stage-0 activation
holdings, not a cold-pool artifact. Consequences per the FAIL clause:
blockK stays dead at 131k; dial K floor 25+ with the valve stack as
coverage path; no K<25 follow-ups; no d4 secondary. Memory-model
disposition: vindicated at M=1, corrected at M>=2 (M-scaling, not
first-window excess) — correction routed to MEMORY_LEG_FINAL_STATUS.md
section 4. Canary-bar allocation: the >5e-3 drift bar binds the loss
canary; gn holds the 0.36-0.49 comparability class; both passed (probe
ended by actual OOM, not canary stop). Box idle-armed for Jack's
stop/keep.
