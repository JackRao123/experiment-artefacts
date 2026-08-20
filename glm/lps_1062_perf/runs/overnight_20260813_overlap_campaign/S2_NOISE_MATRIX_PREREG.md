# S2 NOISE-MATRIX PRE-REGISTRATION — the parity re-adjudication (kolmogorov, 2026-08-14)

Ordered by turing; bars/decision-rule mine, registered BEFORE any measurement.
Context: the S2 ship-stack parity compare (B/F-off vs B/F-on, conventional
PP2/CP8/EP8 path, 9 datums / 262,032 tokens) FAILED the runbook bars —
**genuine numbers (re-verified on the correct `parity_ship-bf-*` files):
loss rel 9.112e-05 (bar 1e-6), per-token max-abs 3.495, ~95.7% >1e-3 (bar
1e-3)** — and the localization question (B/F cache bug vs conventional-path
run-to-run nondeterminism) is OPEN. My earlier "within-boot control" was
computed on W2-era EXECUTOR-arm files (scope mislabel at analysis; voided
in-place in NOTEBOOK 15:4x). The conventional path's run-to-run floor has
NEVER been measured (all W2 detprobe compares were executor-arm).

## The noise matrix (turing's order; lovelace drives; all outputs job-id+timestamp-stamped, stamp verified before pull)

| cell | design | measures |
|---|---|---|
| **N1: off within-boot** | A2 boot (fresh B/F-OFF): after the primary single leg completes untouched, run the parity driver 2 more times on the SAME boot → `a2-r1/r2/r3` | conventional path, B/F absent, within-boot run-to-run floor |
| **N2: on within-boot** | held boot B (B/F-ON): run the driver 2 more times → `ship-bf-on-r1/r2` | ship stack, B/F armed, within-boot floor |
| **N3: off-vs-off cross-boot** | A (S2's boot-A leg) vs A2 (N1's primary leg) | conventional path boot-to-boot floor |
| **N4: off-vs-on cross-boot** | the existing genuine S2 compare (already measured: 9.112e-05 / 3.495 / 95.7%) | the quantity under adjudication |

## Pre-registered decision rule (mechanical)

Compute each floor cell's three statistics: per-token max-abs, % tokens
>1e-3, loss rel spread.

- **Branch (b) — conventional-path nondeterminism (B/F exonerated):** N4's
  per-token max-abs ≤ max(N1..N3 max-abs) AND N4's pervasive% within the
  N1..N3 range AND N4's loss rel within the N1..N3 loss spreads. The B/F
  correctness claim then stands in the noise-relative wording (turing's
  frame): "off-vs-on indistinguishable from the path's intrinsic
  nondeterminism at both granularities; caches bitwise-exact by construction;
  canary agreement ≤7e-4" — NO 1e-6/1e-3 claim is imported anywhere (hertz's
  sharpen).
- **Branch (a) — B/F cache bug:** N4 materially EXCEEDS the measured floors
  (max-abs above every floor cell by a clear margin, or pervasive%
  systematically higher, or loss rel above the floor spreads) ⇒ PR #26 stays
  BLOCKED, tonight's B/F perf wins go suspect pending root-cause, hard
  escalate.
- **Ambiguous middle** (N4 above some floor cells, within others): the
  conservative read is branch (a)-leaning — the ship claim does not get the
  benefit of an unresolved gap; escalate with the full matrix published.

## Sequencing amendment (turing's proposal, ACCEPTED + pre-registered here
before N1/N3 data lands — 2026-08-14 ~16:4x CDT)

Boot B was stood down for the A2 boot (crossed orders), so N2 as originally
ordered would cost an extra boot. Amended sequencing — **no boot is spent on
N2 alone**:

1. A2 proceeds: N3 primary leg (untouched first) + N1 repeats on the same
   boot.
2. **PROVISIONAL re-rule on N1+N3+N4 alone:** if the OFF-path floor (N1
   within-boot + N3 cross-boot, both B/F-OFF conventional) COVERS N4's
   3.4953/95.7%, branch (b) is decidable WITHOUT N2 (the OFF path alone
   reproducing the failure magnitude exonerates B/F regardless of the ON
   arm's noise class).
3. N2 rides the NEXT boot either way:
   - **In branch (b):** N2 rides the S1 soak boot (B/F-ON) — the parity
     driver ×3 at boot start, BEFORE any optimizer step. **My pre-registered
     call (the judgment turing handed me): ACCEPTABLE, not contaminating** —
     the driver legs are forward-only (no optim_step ⇒ weights never move ⇒
     the fresh-boot weight-validity rule holds), they complete before the
     soak's step 1, and the soak's reads are steady-state over 60 steps
     (warmup windows excluded by standing convention). The N2 legs are
     excluded from the soak's perf/canary reads (they are driver legs, not
     bench windows). If this is later judged contaminating, N2 is a
     completeness item logged as not-collected-tonight.
   - **In branch (a)-indicated (N1/N3 tight, N4 above the OFF floor):** the
     re-boot is the DIAGNOSTIC boot and N2 is its first measurement — and N2
     then carries the bug-class discriminator: ON within-boot repeats
     DIVERGE ⇒ "caches inject nondeterminism"; ON within-boot repeats TIGHT
     but off-vs-on diverges ⇒ "caches deterministically change the math"
     (a different, sharper bug class for the escalation).

The mechanical decision rule of the previous section is unchanged; this
amendment only re-sequences N2 and adds the provisional N1+N3+N4 rule.
