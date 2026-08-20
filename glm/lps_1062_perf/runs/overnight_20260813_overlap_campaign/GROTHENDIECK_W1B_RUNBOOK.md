# W1B DRIVER RUNBOOK — grothendieck (block+K21 config-only A/B @131k)

Written 2026-08-14 ~00:2x CDT, post-rehydration. Lane assigned by bayes
(session-restart wave successor to doppler's box-mechanics lane; lovelace =
ex-doppler, owns lifecycle supervision + W0/W1a). If grothendieck dies
mid-window, this file + MEMORY_LEG_DECISION.md + BOX_SCHEDULE.md rehydrate
the lane.

## HARD RULE (Jack via bayes, zero-tolerance, 2026-08-14)

NEVER block the session on trainer/bench state: no foreground log-tailing,
no ssh-and-stare, no self-written sleep-loops. Sanctioned pattern: launch
via the .devbox_up scripts, background waits (nohup + completion marker /
exit reported), short NON-BLOCKING checks only, stay message-responsive.
wait_trainer_health.sh exists exactly for this. If no wait tool covers the
case: papercut + message bayes — do NOT improvise a blocking wait. A
session found blocking hands its lane off immediately. (Written after my
own 1h stall cost me this lane at the succession boundary.)

## Identity / reporting

- I DRIVE: boots, configs, logs. lovelace SUPERVISES lifecycle
  (start/stop/wait_trainer_health.sh — I do not freelance lifecycle).
- Report to bayes at WINDOW OPEN and WINDOW CLOSE (+ any STOP condition).
- pauli (ex-jacobi) owns the memory-leg analysis my measurement feeds;
  K-slope arithmetic confirmation requested from pauli ~00:2x; their answer
  goes into the window report to bayes (cc requirement).

## Preconditions (do not open the window without)

- W0 GREEN on wprm693: venv nvidia-cudnn-frontend==1.27.0 (PyPI --no-deps,
  pip from /tmp, cu13 lane, import name 'cudnn'), both DSA test files
  5-passed. (lovelace's gate.)
- W1a anchors IN-BAND on wprm693: d2 canary 12.2-12.4 / gn 0.36-0.49;
  d4 878-886; d16 984 ±3% (954.5-1013.5). W1b A/B is vs THESE same-box
  anchors, not the carried-in 143/162 peaks alone.

## The change (single variable)

- Config: `trainer_pp2cp8ep8_131k_blockK21.json` = mission config +
  top-level `recompute: {granularity: full, method: block, num_layers: 21}`
  (stock mcore block recompute: first 21 layers/stage checkpointed,
  remaining 19 eager; LoRA-safe on stock path; no shim/flag/patch).
- Staged box-side: `/root/.cache/user_artifacts/lps1062_pp2/trainer_pp2cp8ep8_131k_blockK21.json`
  sha256 `c94a9a42af4bde9103f818b1446d0b96c6f2b4921a076f97dbf95df6f25622b6`
  (verified leader AND rank-1 sibling read, 00:2x — CPFS quirk insurance).
- ENV: byte-identical to W1a + the config swap. NO BT_DIAL_MEM_PROBE
  (bayes adjudication: dial-branch-only var, inert on mission tree
  mcore 57efae08b — verified by grep; a no-op env var invites
  "was it load-bearing?" questions). BT_SAVE_STATE_SYNC=1 ALWAYS (F1
  landmine: async save hangs under CP>1).

## Sequence (per MEMORY_LEG_DECISION §5 pre-registrations)

1. **d1 boot → fit read:** peak reserved ≤ 255 GiB with I=2 in-flight.
   Predicted 220-240. (d2 adds nothing memory-wise — I=2 already at d2.)
2. **d2 canary:** loss 12.2-12.4 + gn comparable to fixed-wheel 0.36-0.49.
   Drift > 5e-3 = STOP (would indict checkpoint-RNG or LoRA-patch
   interaction). Recompute-path changes should be numerics-neutral.
3. **d4 A/B pair vs W1a anchor (878-886):** within-boot control pairs must
   agree ±2% — controls trip ⇒ DISCARD + repeat (tonight's d4 noise ran
   3.0-3.8% twice). Win bar > anchor + 2%. Prediction +4-8% (≈915-955).
4. **d16 pair vs 984** (only if d4 passes): prediction +6-10% (≈1045-1080).
   BT_PROFILE_RANKS=0,8 on this pair; pull traces Mac-side IMMEDIATELY
   (box can die); jacobi (ex-kepler) reads them.
5. **K=18 and/or K=24 slope boots:** ONLY after the primary d4+d16 pairs
   complete and still in-window (bayes: perf verdict primary, S_eager
   slope secondary).

## Bars / failure actions

- Fit miss (>255 GiB RESERVED — poller metric, not torch-allocated):
  K′ = 40 − 19×(255 − peak_full_same_rung)/Δ with Δ = peak(21) −
  peak_full_same_rung (pauli-confirmed), ONE retry. Second miss ⇒ memory
  model wrong ⇒ back to bayes/pauli, no further dial-family boots.
- Canary drift > 5e-3: STOP + verbatim report.
- Control gate trips at d4: discard run, repeat.
- Memory abort line: >255 GiB reserved.
- d4-vs-d16 slope disagreement (material): investigate before trusting
  any K (pauli: growth would be in the stored set, not the base).

## Measurement deliverable (feeds pauli's dial K)

Peak reserved at K=21 per rung → per-layer eager stored delta → S_eager
measured (retires the 2.25 GiB E1 upper bound; dial risk §6.3).

**pauli's confirmed arithmetic (00:3x, verbatim answers to my 4 questions):**

1. Base = SAME-RUNG W1a anchor peak, NOT the 143/162 carry-ins (different
   box/boot/wheel-state; tonight's cross-boot drift ran to 5%). The whole
   slope value lives in same-boot deltas.
2. BOTH rungs, check agreement. S_eager is rung-invariant in theory
   (per-mb tokens = 16384 at every dN; I = min(M,PP) = 2 both rungs); M-
   growth should live in BASE (allocator/optimizer class), which same-rung
   anchors absorb. d4-vs-d16 slope DISAGREEMENT = the growth is in the
   stored set instead ⇒ investigate before trusting any K. d4 primary if
   forced (cheaper; the dial's first A/B is d4).
3. ONE K=21 point per rung SUFFICES — do NOT fight for K=18/24.
   dPeak/dK ≈ −4.1 GiB/layer, so a 10% slope error mis-sets K by <1
   layer; the dial's d1/d2 ramp guard absorbs that. Rung agreement is the
   free linearity check. Extra boots ONLY if the two rung slopes disagree
   (the linear-model-failure signature that earns the second point).
4. Retry formula CONFIRMED: K′ = 40 − 19×(budget − peak_full_same_rung)/Δ,
   Δ = peak(21) − peak_full_same_rung. (a) budget = **255 GiB** (abort
   line), not 275 — keep the 20 GiB band; (b) 'peak' = poller's max
   **RESERVED** MiB throughout (the 255 line is reserved; E1's 258.8 was
   torch-allocated — DO NOT mix metrics).

**TRANSFER CAVEAT (pauli, must ride the window report):** W1b measures
S_eager on the plain M=N path. The dial arm runs the combined-1f1b
executor whose per-node detach/free machinery (should_free_input et al.)
may shift eager-layer stored cost slightly — E1's 2.25 bound was also
plain-path. ⇒ dial K* = measured K* MINUS 1-2 layers transfer margin for
the first dial boot; kolmogorov's d1 ramp then re-measures under the
executor and walks K up.

## Bench mechanics (box)

- Kit: `/root/.cache/user_artifacts/lps1062/` (bench_driver2c.py,
  run_bench2c.sh, poll_gpu_mem.sh, fold_mem.py, mfu.py). Driver hits
  trainer HTTP :8001 on the leader; run_bench2c wraps with per-node GPU
  mem pollers + folds max-mem into the result JSON.
- dN = N datums × 131072 tok per window: `--seq-len 131072 --num-gpus 16
  --datums N --repeats R`; canary drift via `--canary-json <baseline>`.
- Evidence: result JSONs (lps1062_bench/<label>.json) + logs pulled to
  Mac-side campaign folder; NOTEBOOK.md append at window close.

## EVENT LOG (window live)

- ~02:3x CDT — arm boot #1 (slurm job 19): **OOM in trainer-internal
  warmup** (first forward after weight load), rank15/stage-1 MoE bias_act:
  260.33 GiB torch-allocated / 266.76 in-use / 267.69 capacity. No bench
  windows ran; no poller peak exists. Box cleaned (no strays, GPUs 0).
  Reported to bayes as fit-breach STOP with Path A (BT_SKIP_WARMUP=1
  reboot → d2 fit read per protocol) vs Path B (blind K cut). Awaiting
  call. LESSON: wait_trainer_health.sh exits 1 for both death and
  3-min timeout — discriminate on TEXT ("TRAINER PROCESS DIED" vs
  "TIMEOUT"), not exit code (papercut filed).
- ~02:5x CDT — boot #2 (BT_SKIP_WARMUP=1): HEALTHY in ~6 min; d2 bench
  fired (fit probe = bench warmup0). **OOM AGAIN at bench warmup0** —
  ranks 14/15-class at 266.3-266.7 GiB reserved (rank15 died first, same
  as boot #1). lovelace cleaned; box idle. VERDICT (lovelace, definitive):
  blockK21 @131k OOMs on the first full-shape fwd+bwd EVEN POST-SETTLE —
  the eager-layer store does not fit at K=21 regardless of settle state.
- ~03:0x CDT — **CONSTANT CORRECTION (pauli, ratified bayes): S_eager =
  2.94 GiB/layer/mb, not 2.25** (2.25 was the E1-selective constant with
  core_attn still checkpointed; block+K eager layers retain DSA attention
  internals, +0.7). Exact fit: 147 + 2×(21×0.19 + 19×2.94) = 266.7 =
  measured. K ≥ 23.1 → **K=25 ratified for the single authorized retry**
  (K=24 = ~250, 5 GiB margin; K=25 = ~244.6, ~10 GiB vs optimizer-state
  materialization uncertainty; K=22-on-wrong-base = 261 certain OOM).
  Prize rebases: block+K standalone ~+5-8% (from +6-10%); win bar stays
  anchor+2% same-class. If K=25 misses fit: STOP, verdict = "blockK does
  not fit at 131k on this stack" — a real answer. Dial ladder memory gate
  re-registered TWO-STAGE (d1 slope under the executor → recompute K → d2
  canary at that K). blockK25 config exists Mac-side (sha256 ddc75eed…,
  = mission + num_layers 25 only, verified by diff).
- ~03:1x CDT — my session stalled ~1h (restart-wave pattern); bayes
  handed W1B EXECUTION to lovelace at the clean boundary per the
  succession rule. grothendieck = support/resume-at-next-boundary role.
- ~04:4x CDT — **W1B FINAL VERDICT: blockK DOES-NOT-FIT at 131k on this
  stack (lovelace driving, verdict accepted).** K=25 d2 warmup0 COMPLETED
  (canary in-band: loss 12.3242, gn 0.4185; fb 193.4s compile-heavy) but
  peak hit poller-max 265.0 GiB stage-1 (trainer-reported 256.8 reserved /
  253.2 allocated — ~8 GiB poller-vs-trainer non-PyTorch gap; the 255 bar
  is poller-reserved per pauli). Over the bar at d2; d16 projection ~282 =
  unreachable. No third boot per bayes's pre-registration. Support task
  done: full evidence set (both Ks' poller CSVs + runlogs + K=25 peak
  lines + OOM evidence note preserving the truncated boots' numbers)
  sha256-verified in w1b_evidence/, delivered to pauli, confirmed to
  bayes. Campaign moves to kolmogorov's ruling (dial lane).

## Box facts (wprm693, verified 00:1x)

- 2×8 GPUs, 275040 MiB each (B300; driver string says "L20D" — quirk, HBM
  size is the tell). Idle at handoff. ssh aliases tj-wprm693 / tj-wprm693-1.
- Lifecycle scripts: `/root/.cache/user_artifacts/.devbox_up/`
  (start_trainer.sh / stop_trainer.sh / wait_trainer_health.sh — ONLY
  these; no ad-hoc launchers/pkill loops).
- Shared FS persists kit + prior results (mn-d2..d16 JSONs present).
