# P4 SOAK SPEC — ship-stack validation (mission + B/F) — kolmogorov, 2026-08-14

Status: **FINAL (composition settled by bayes, 2026-08-14): mission + B/F,
nothing else.** W1/option-6 are closed-negative (W4 verdict) and are NOT in
the stack. This is the campaign's LAST box window; after S3 the box goes
idle-armed for Jack's stop/keep call.

Ship stack: **PP2/CP8/EP8 @131k M=N headline + B/F host caches armed + ship
NCCL env (inert-but-kept, W1c Arm-1) + TF32 head (PR 995 content) +
BT_SAVE_STATE_SYNC=1 (F1 workaround)** on the canonical exact bits
(73c24b00+TF32; mcore = gate-stack tree, B/F armed by env).

## Boot checklist (the swap-drop lesson, papercut pc_6dca3b0f40f0)

The box is currently on the option6-staged tree. Restore to the ship stack
per the W1d protocol, with the TF32 line EXPLICIT this time:

1. trainers → 73c24b00; **re-apply the TF32 patch and VERIFY by the applied
   diff's sha256 == the Mac-side reference patch
   (`tf32_head_port_21d0c578.patch`, sha256
   f503c9bf1baccbd260bfa531525f6f3f4de2c7bd6b114b73fda9a4a06697e801) — NOT a
   grep count** (the canonical patch has exactly ONE occurrence of the env
   var; my earlier `grep -c ≥ 2` line was miscalibrated — lovelace caught it;
   the sha is the authoritative check). The W3/W4 swap dropped the patch and
   ran TF32-less with an inert env var — do not repeat.
2. bridge → 20fcf2ea; mcore → 57efae08b + re-apply the gate-stack snapshot;
   verify dirty-diff sha256 == `e1e46818dfc684566815fdf482574c08bc275451d4471706dc89d0154eee3653`
   EXACT (mismatch = STOP, partial apply silently changes content).
3. Wheel: cudnn-frontend 1.27.0 in the trainer venv (verify, don't assume).
4. Env on every boot: `BT_TF32_LM_HEAD=1` + ship NCCL 3 knobs +
   `BT_SAVE_STATE_SYNC=1`; B/F gates per leg (S2 needs one OFF boot).
5. Trainer lifecycle: `.devbox_up` scripts + `wait_trainer_health.sh`
   **backgrounded** (zero-tolerance wait rule); pre-dispatch `squeue` empty
   check (lovelace's standing rule).

## Goals (what the soak must prove before the final report/PR package)

1. **Stability over duration** — the campaign's benches are 2–4-window sprints;
   the soak catches slow drift, memory creep, cache-counter anomalies, and
   throughput settle over many optimizer steps.
2. **Numerics parity of the ship stack** — B/F-on vs B/F-off must be
   numerically identical within the parity bars (the caches are bitwise-exact
   by construction; this is the on-hardware proof at the mission scale).
3. **Export/save sanity on the ship stack** — the LoRA export path and the
   sync-save workaround exercised end-to-end after real steps.

## Leg S1 — the soak (main event)

- One boot of the ship stack (B/F armed; hard gate: both ACTIVE lines ×16).
- Run: d16 windows (2M tok/step) for **≥ 60 optimizer steps** (~2.5–3 h at
  the 1103-class rate), mem-poller on (2 s cadence, per-GPU max).
- Pre-registered bars, evaluated per window:
  - loss trajectory within the established band for the step count (reference:
    the off-arm anchors' trajectory class; drift vs the B/F-off reference
    boot's matched-step losses ≤ 2e-3 per the standing canary rule);
  - gn smooth, no step discontinuities outside the observed 0.36–0.49-class
    evolution;
  - **memory flat over time**: per-GPU peak must not grow beyond the d16
    reference (163 GiB class) by >2 GiB across the soak (creep = stop);
  - B/F telemetry: the per-window counters show zero fallbacks/misses
    (stale-cache or fallback = stop);
  - throughput: steady-state band holds (no monotonic decline beyond the
    tonight-observed settle drift).
- Any STOP condition → halt, snapshot logs (the clobbering rule), report.

## Leg S2 — B/F numerics parity (the on-hardware bitwise-class proof)

- Fresh boot A: ship stack with B/F OFF. Fresh boot B: B/F ON. (Fresh-boot
  weight-validity rule per PARITY_RUNBOOK: LoRA B zero-init ⇒ fresh-boot
  forward == base model exactly.)
- Driver: `tools/parity_driver.py`, the fixed 9-datum mixed set (262,032 real
  tokens), NO optim_step (weights never move).
- Bars (PARITY_RUNBOOK): loss rel diff ≤ 1e-6; per-token logprobs max abs
  diff ≤ 1e-3; datum count/lengths EXACT. The caches are bitwise-exact by
  construction, so any exceedance = a real bug (stale-cache/identity-key
  failure class) → STOP + escalate.

## Leg S3 — export + save sanity

- After S1's soak (weights moved): run the LoRA adapter export
  (`stream_adapter_weights_megatron_to_hf`) on the ship stack — verify the
  key/shape set == reference, values non-degenerate (dedekind's L1
  protocol), adapter_config correct.
  **⚠️ IN-PLACE CORRECTION (kolmogorov, 2026-08-14, per turing/hertz):**
  the "392-key" figure this line originally carried was a transcription from
  dedekind's Qwen3-0.6B small-model test and is INAPPLICABLE to the mission
  model. The GLM-5.2 reference of record is the **1094-key complete set**
  (internally consistent, L1 non-degeneracy 547/547, adapter_config
  correct — lovelace's S3 result). The bar's intent (complete + correct +
  non-degenerate set for the target model) is unchanged; the constant is
  corrected from the small-model reference to the mission model. Ratified
  in my S3 adjudication; CAMPAIGN_REPORT §6 reads "scope note ratified" on
  this basis.
- One `/save_state` cycle with `BT_SAVE_STATE_SYNC=1` (the F1 workaround
  exercised end-to-end): completes < 5 min, checkpoint on disk, trainer
  resumes clean after.
- Note on scope: the CP>1 async-save wedge (F1) is a parked escalation; the
  soak exercises the SYNC path only (the ship workaround), not a re-validation
  of async.

## Ordering + cost

S2 (parity, ~30 min: two fresh boots + driver) → S1 (soak, ~2.5–3 h) → S3
(export + save, ~30 min, rides the S1 boot). Total ≈ 4 h of box time.
**jacobi's R2 seam-probe rides an S1 idle gap** (their runner is prepped:
isolated py-spy in /tmp scratch venv — trainer venv untouched; 420s @250Hz
--threads; artifacts to shared-FS `r2_seam_probe/`; fully nohup'd with a
PROBE_COMPLETE completion line — zero blocking). Coordination protocol
(jacobi's three asks, to be executed by lovelace): (1) ping jacobi with the
job id when the S1 boot is healthy + in steady state; (2) the gap is
schedulable ~30–60 min into the soak (confirm at the S1 start report);
(3) if jacobi's env-scan pid identification fails on this box image, lovelace
drops a pid-file at `r2_seam_probe/rank0.pid` / `rank8.pid` on the shared FS.
The probe is observation-only; probed steps are excluded from perf reads per
jacobi's spec, and the soak's steady-state read is undisturbed (mem-poller
cadence stays on).
Standing rules: fixed wheel, `BT_SAVE_STATE_SYNC=1`,
`wait_trainer_health.sh` backgrounded, traces/logs snapshotted per the
clobbering rule, one-variable discipline (the soak stack changes nothing vs
the W1c/W1d on-arm bits except duration).

## Post-window disposition

After S3: snapshot all soak logs + the parity JSONs + export artifacts to the
shared FS and pull Mac-side (the clobbering rule); then the box goes
**idle-armed** — stop/keep is Jack's call (turing relays; succession
correction 2026-08-14, was "bayes relays").

## S1-end pre-registered reads (added 2026-08-14 ~18:5x CDT, BEFORE main60 — turing's ask + hertz's shadow annotations)

1. **Throughput record language (turing's framing question):** the soak's
   deep-steady warm class (1121–1128 at windows 1–21) runs above the W1c
   A/B's 1103 record mean (1088/1118 fresh-pair). PRE-REGISTERED READ: this
   is **the same config measured at deeper settle, not a new lever** — if
   windows 22–60 hold the class (± the tonight-observed settle drift), the
   number of record's steady-state estimate is REFINED to the soak's
   deep-settle mean, reported WITH the note that the A/B's 1103 was the
   fresh-pair read of the same quantity. No "new record" claim (nothing
   changed in the config); no lever credit.
2. **gn evolution (hertz's watch item):** the 0.36–0.49 band is the
   step-0/d2 CANARY reference, not a 60-step band. The soak's gn 0.37→0.22
   smooth monotonic decline is the training-progression regime (Aug-9
   precedent: gn 0.90→0.39 over 10 optim steps adjudicated as weight
   evolution). The soak's gn bar is SMOOTHNESS (no step jumps), not the
   canary level. A step JUMP = the stop signal, not the level.
3. **Absolute memory annotation (hertz's record-keeping note):** the soak's
   179.8 GiB absolute sits above the A/B's 163-class d16 reference because
   the soak's continuous 2 s polling catches sub-window transients the
   per-window driver peak misses, and the allocator's reserved pool
   stabilizes over the early windows (the +0.9 GiB early rise then flat).
   The BINDING bar is creep ≤ 2 GiB over the soak (holds); the absolute
   level is annotated here so the record does not read 179.8 as anomalous
   against 163.

## Falsifier / escalation summary

| signal | reading | action |
|---|---|---|
| parity leg loss rel > 1e-6 | cache numerics bug | STOP, escalate (the ship stack's correctness claim dies) |
| memory creep > 2 GiB over the soak | leak class (cache carrier retention?) | STOP, profile before continuing |
| fallback/miss counters nonzero | cache keying failure in the wild | STOP, escalate |
| throughput monotonic decline | settle/thermal/fragmentation class | annotate; >5% decline = stop |
| export key/shape mismatch | export-path regression on the ship stack | STOP, escalate to the export lane |
