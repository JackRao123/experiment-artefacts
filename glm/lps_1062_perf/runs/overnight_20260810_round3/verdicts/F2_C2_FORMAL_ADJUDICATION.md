# FORMAL ADJUDICATION — F2 (c2): equal-count DP2 flag-ON vs flag-OFF A/B
Author: curie (verification lane) · Date: 2026-08-10
Gate: F2_C_ADJUDICATION_FREEZE.md (c2) as frozen + Amendments #1 (vacuous (c1)
screen; inertness scope) and #2/A3 (boot-history symmetry).
Trigger: helmholtz's re-open (the box-lane verdict reached PR #1001 without
routing through this gate). All numbers below are MY OWN computation from the
md5-matched drops (jsons 684a218a/7d851ad8, arm log 94d604bf, flag-ON boot log
f5bbe4d9, driver logs f1899ca7/c9f4b487).

## VERDICT: (c2) PASS — the fix is numerically INERT at structurally equal
## partition counts. My numbers AGREE with the box-lane verdict; no amendment
## to the F2 verdict set or PR #1001 is needed on the numbers.

### 1. Per-window deltas vs the band (primary statistic)
21 windows matched 1:1 (20 mains + 1 warmup; no A-only/B-only windows).
Both arms: 131k-d4, 16 GPUs, DP2, warmup_datums=2, final step 21.
- max |dloss| = 1.6499e-3 at **main1** (box lane said "main0" — trivial
  attribution slip; main0 is +1.609e-3, the max is main1's −1.650e-3; both
  in-band, immaterial).
- ALL windows in-band (≤2e-3); none near the 5e-3 stop.
- Reading: at equal counts phantoms never fire; the flag is inert (one 4-byte
  all-reduce per step); the deltas are cross-boot noise class.

### 2. Matched warmup-datums — VERIFIED by direct evidence
Both jsons warmup_datums=2; warmup0 losses cross-match the trainer/driver
logs exactly (A: 12.316429645001564 = trainer step-1 loss; B:
12.317383326593983 = driver-log warmup0).

### 3. A3 boot-history symmetry — SATISFIED IN INTENT, one documented
### forced asymmetry + one evidence-class caveat
- Both arms fresh boots, ZERO optim_steps before window 0 (flag-ON boot log:
  first optim_step is step=1.0 = window-0 warmup's own step; nothing prior).
- Boot-warmup execution asymmetry: arm A ran the default 1-datum boot warmup
  (no optim_step); arm B skipped it (BT_SKIP_WARMUP=1). This is STRUCTURALLY
  FORCED — the 1-datum boot warmup is the (a3)-proven unequal-count deadlock
  trigger that flag-OFF cannot survive — and weight-state-neutral (no
  optim_step either way; both arms enter window 0 at init weights). The
  aged-boot lesson (prior optim_steps) does not apply; A3's intent (identical
  weight-trajectory history) holds.
- Evidence-class caveat: the flag-OFF boot's trainer log was overwritten by
  the flag-ON boot (single-log launcher class — the filed papercut). Arm B's
  symmetry evidence rests on the arm-log record + driver log + json (final
  step 21, warmup_datums 2). Consistent and sufficient here; recorded as a
  survivable-evidence gap, not an INVALID condition.

### 4. Pre-registered warmup0-excess prediction — CONFIRMED
warmup0 delta = −0.954e-3: in-band, NO middle-zone excess. The frozen
prediction (AV3_CANARY_ADJUDICATION.md, pattern record): an inert flag at
equal counts = behaviorally same-config ⇒ ARM-0 contrast ⇒ no warmup0
excess. Confirmed on first data. This is positive evidence FOR the
inertness premise (a middle-zone excess would have flagged executed-code-path
differences). Also the pattern's third no-excess point (config-shifted in
form, same-config in behavior).

### 5. Drift cover (standing rule) — CLEAN
20 mains: deltas oscillate and TIGHTEN (±1.6e-3 early → ±0.3–0.7e-3 late);
no monotone trend. dgn context: max |Δgn| 0.021 (main1), same-sign, no
zero-crossing (the aged-boot signature class absent).

### 6. Scope wording (per Amendment #1, for the scorecard + PR #1001)
The (c2) verdict covers fix-INERTNESS at equal count. It does NOT numerically
test phantom-FIRED correctness (impossible cheaply: fired requires unequal
counts, flag-OFF can't run those). Fired-correctness rests on the zero-mask
design argument + (b)'s sane completion + the fibonacci 1a/1b NaNx0 checksum
canary as the SHIP gate.

### Context for the (d) perf item (NOT part of this verdict)
Driver summaries: flag-OFF 740 vs flag-ON 700 tok/s/GPU (~5.7%) — surprising
at equal counts where the fix is nearly-free (one 4-byte all-reduce/step);
cross-boot, single-boot-per-arm. Flagged for the (d) perf investigation
(fibonacci), not adjudicated here. DEFLATORS (helmholtz, 2026-08-10 — check
before the 5.7% is believed): (i) box 2 had NO anchors — its wall numbers
are anchor-less cross-boot reads; (ii) the ON boot may have carried
F2_SHADOW-mode instrumentation overhead. Both are (d)-investigation items;
the delta goes to the scorecard as a ship-gate QUESTION, not a finding.

### (c1) note
The arm log's (c) drift rows (w0 −34.5e-3 → m2 −67.8e-3, monotone) are
consistent with the frozen Amendment #1 disposition: VACUOUS screen (zero
fired windows by construction at 2=2 partitions/replica), recorded as pure
DP1/CP16-vs-DP2/CP8 config-trajectory divergence — not a screen result.
