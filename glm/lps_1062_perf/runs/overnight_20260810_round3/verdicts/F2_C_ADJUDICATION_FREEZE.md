# FROZEN ADJUDICATION — F2 item (c) re-design (numbers pinned BEFORE main-window reads)
Author: curie (verification lane) · Date: 2026-08-10 · Status: FROZEN.
Trigger (helmholtz, URGENT): grothendieck proved phantoms NEVER-FIRED at (c)'s
warmup0 (equal partition counts at 2-datum/2-replica warmup), yet warmup0 vs
DP1-golden = -34.6e-3 ⇒ the DP1-golden 5e-3 bar measures a reduction-topology
CONFIG effect (DP1 vs DP2), not phantom behavior ⇒ verdicts against DP1-golden
are INVALID-BY-DESIGN. Structure ruled by helmholtz; numbers frozen by curie here.

## (c1) — Split-consistency bar on the EXISTING run (screen; not the verdict)
Definitions:
- For any window w: Δ(w) = loss_run(w) - loss_DP1golden(w)  (existing (c) run).
- Config-effect baseline: Δ(warmup0) = -34.6e-3  (n=1, PROVEN not-fired).
- Consistency statistic per phantom-fired window w:  s(w) = Δ(w) - (-34.6e-3)
  i.e. s(w) = Δ(w) + 34.6e-3.  Equivalently Δ(w) read against
  [-36.6e-3, -32.6e-3] (in-class) and [-39.6e-3, -29.6e-3] (stop edge).

Bar (per fired window):
- IN-CLASS:   |s(w)| <= 2e-3   (Δ within [-36.6e-3, -32.6e-3])
- MARGINAL:   2e-3 < |s(w)| <= 5e-3
- OUT-CLASS:  |s(w)| >  5e-3   (Δ outside [-39.6e-3, -29.6e-3])

Aggregate verdict for (c1):
- PASS:     EVERY phantom-fired window IN-CLASS.
- STOP:     ANY fired window OUT-CLASS.
- MARGINAL: otherwise (>=1 marginal window, none out-class) => escalate to
            helmholtz with the full s(w) distribution; no re-thresholding.

REPORTED context (no bar): trend form — regress s(w) vs fired-window index;
a monotone trend indicates config-effect DRIFT (undermines the static window),
a step at firing boundaries indicates phantom effect. Reported to help the
escalation, not gated.

n=1 WEAKNESS (stated per ruling): warmup0 is the ONLY proven-not-fired window.
Config-effect drift between warmup0 and the fired windows is indistinguishable
from phantom effect under this static form. Therefore (c1) is a SCREEN on the
existing run: STOP/MARGINAL triggers scrutiny, but the VERDICT rides on (c2).

## (c2) — Verdict-carrying gate: equal-count DP2 flag-ON vs flag-OFF A/B
Design (pinned): single variable = the phantom flag; topology fixed at DP2 both
arms; equal datum counts per replica across arms; two boots (cross-boot class).
- Band: house band, cross-boot class — |Δ| <= 2e-3 PASS / > 5e-3 STOP,
  middle zone per standing house ruling (escalate with distribution).
- Matched --warmup-datums MANDATORY across arms: mismatch => INVALID, re-run.
- 20-step drift cover per the standing numerics rule (standing-rule carryover).
- Primary statistic: per-phantom-fired-window Δ(w) = loss_ON(w) - loss_OFF(w),
  windows matched 1:1 across arms; every fired window must be in band.
- Aggregate (reported, no separate bar): mean/median/max|Δ| over fired windows
  and over main windows.
- Verdict: PASS = all fired windows <= 2e-3 with drift cover clean;
  STOP = any fired window > 5e-3; MARGINAL = otherwise => escalate.
- INVALID != FAIL throughout (warmup-datums mismatch, boot-env anomaly,
  telemetry gap => repair + re-run, not a verdict).

## Sequencing note
(c1) is computable on the existing run the moment window losses + fired-window
telemetry are joined. (c2) requires the new A/B boots. (c1)'s screen outcome is
reported WITH (c2)'s verdict, never as a substitute for it.

## AMENDMENT #1 (2026-08-10, pre-(c2)-boot; helmholtz adjudication input)
- (c1) is VACUOUS-AS-SCREEN on the existing run: the run has ZERO phantom-fired
  windows BY CONSTRUCTION (131k-d4 @DP2 = 2 equal partitions/replica — phantoms
  never fire at equal counts). There are no fired windows to screen. The
  observed monotone drift -34.6e-3 (warmup0) -> -67.8e-3 (main) is recorded as
  PURE CONFIG-TRAJECTORY DIVERGENCE (DP1-vs-DP2 reduction-topology effect
  evolving over training), NOT as a screen result. The (c1) machinery above
  stands frozen for any future run that DOES contain fired windows.
- SCOPE OF THE (c) SUITE (pinned for verdict text + helmholtz's scorecard):
  the (c) suite tests FIX-INERTNESS — equal-count, single-variable (flag),
  same topology. It does NOT and cannot cheaply test phantom-FIRED correctness
  against a numerical reference (fired requires unequal counts; the flag-OFF
  arm cannot run those). Fired-correctness rests on: (i) the zero-mask design
  argument, (ii) (b)'s sane completion, (iii) the 1a/1b NaNx0 checksum canary
  as the SHIP gate. Any (c) verdict text must scope itself as inertness-only.
- CONDITIONAL (data-dependent, folded in only if the data exists): IF the (c)
  driver logged per-replica PRE-REDUCTION losses (replica-1 step-0 locals,
  matched data), a free fired-vs-unfired comparison may be added as REPORTED
  context. If the data does not exist, this is struck without prejudice.
- (c2) stands exactly as frozen (equal-count DP2 flag-ON vs flag-OFF A/B,
  house band, matched warmup-datums, cross-boot class, drift cover).

## AMENDMENT #2 (fleet label A3) (2026-08-10, pre-(c2)-rerun; helmholtz
## request, curie wording; helmholtz ACK requested)
- BOOT-HISTORY SYMMETRY — new INVALID precondition across the frame set.
  Every arm of an A/B (and any arm compared against a golden/baseline) runs a
  FRESH boot with IDENTICAL op-history before window 0: same optim_step count
  (normally zero beyond the declared startup warmup), and warmup handling
  itself symmetric (matched --warmup-datums AND warmup executed-or-skipped
  identically across arms). Evidence: boot-region startup lines (step counter
  at window 0 / optim_step count) pasted into the arm's evidence bundle
  pre-drive, alongside the A2.1 arm-check markers. Any arm on a REUSED boot
  carrying prior optim_steps, or any op-history asymmetry vs its comparison
  arm/golden => the arm is INVALID (not FAIL): halt, re-boot fresh, re-drive.
  Lesson of 2026-08-10: (c2) pass 1's arm A rode the aged (b)/(c) boot (8
  prior optim_steps) and produced a textbook decaying trajectory offset
  (-0.10..-0.13, dgn zero-crossing) that masquerades as an early-window
  signal; at equal partition counts phantoms structurally cannot fire, so any
  such offset is staging, never mechanism. grothendieck's earlier 'per-run
  weight reset' claim is REFUTED (helmholtz's aged-boot caveat was the live
  one; the warmup0~init reading was 8-steps-drift being small).

## STATUS LOG (running)
- (c2) pass 1: INVALID (staging error, self-caught by grothendieck): arm A on
  aged boot, 8 prior optim_steps. Arm B json STANDS (fresh boot, skipped
  startup warmup).
- (c2) RE-RUN in flight: fresh flag-ON boot, default startup warmup (no
  optim_step — history-symmetric with arm B's skipped warmup), same 20-main
  drive; verdict ~11:25 UTC. The (c2) warmup0-excess prediction (see
  AV3_CANARY_ADJUDICATION.md, WARMUP0-TRANSIENT-PATTERN pre-registered
  application) now reads on the RE-RUN, not the invalidated pass.
