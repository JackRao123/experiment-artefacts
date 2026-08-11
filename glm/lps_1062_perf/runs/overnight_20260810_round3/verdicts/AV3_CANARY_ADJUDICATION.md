# ADJUDICATION — A-v3 VERIFY=0 canary (box 1, 318g61w) + WARMUP0-TRANSIENT pattern record
Author: curie (verification lane) · Date: 2026-08-10

## SCOPE CORRECTION (2026-08-10, helmholtz; supersedes all scope readings below)
Both A-v3 boots ran UNDER-ARMED (C′+W1 gates OFF; grothendieck's
boot-region-scoped catch; arms INVALIDATED — regime mismatch, not defect).
REGIME TRUTH refined 2026-08-10 PM (fermi's code read): the later RE-ARM
boots (soak + VERIFY=0) ran the status-quo FREE-ROUTING replay regime — FORCE
was set but INERT (its mechanism depends on CACHE's frames), CACHE unset.
So all A-v3 measurements tonight are pre-C′-regime; the post-C′ composition
and wall await the both-flags boots (VERIFY=1 soak + VERIFY=0 timed).
- The SOAK verdict stands ONLY as a B/F-REGIME result: V3 mechanism
  self-consistency (6,863/6,863 bitwise verifies, zero miss/drop/fallback)
  proven in the B/F regime. It is NOT evidence of composition-with-C′; the
  post-C′-regime soak is DEFERRED with the A-v3 re-arm. "A-v3 soak PASS" must
  not read as post-C′-regime-proven.
- The CANARY comparison as-run was (B+F + fixa_v3) vs the r3anchor golden —
  a B/F stack per fourier's handoff ("Anchors (box 1, B+F)") — i.e. a
  regime-MATCHED single-variable B/F-regime canary. The mains in-band verdict
  stands SCOPED to the B/F regime; the intended post-C′ canary is DEFERRED
  with the re-arm. (grothendieck to correct me if the golden's stack differs.)
- The 16k noise characterization describes the as-run (under-armed) boot and
  does NOT bear on A-v3's post-C′ 16k wall; its discriminator is retained for
  any future 16k question.
- Pattern point #2 below (warmup0 +2.8e-3) remains a valid cross-boot
  config-shifted data point (the as-run shift was larger than intended);
  regime annotated.
- The A-v3 TIMED wall numbers (131k +2.0% steady, 16k −2..−5%) are
  INVALIDATED with the arms for the post-C′ question; they stand only as
  B/F-regime measurements.

## VERDICT: ACCEPT-WITH-CAVEAT (numerics PASS on mains) — SCOPED per the
## correction above (B/F regime; post-C′ deferred)
Relayed numbers (helmholtz, from grothendieck's incoming telemetry):
warmup0 +2.8e-3 (middle zone, 2e-3 < x <= 5e-3); mains 0.9–1.8e-3 (in-band).
CONTINGENCY (house discipline): verdict issues on the relayed numbers and is
CONFIRMED against my own extraction when the canary log/json land; any
disagreement => re-adjudicate, not amend.

Basis:
1. Mechanism is bitwise-inert by direct evidence: today's VERIFY=1 soak PASS —
   6,863/6,863 emitted verifies matched the first pass bitwise, zero misses,
   zero drops, zero fallbacks, 22 windows x 16 ranks unanimous.
2. Mains in-band => no persistent or growing numerics effect; a real mechanism
   effect would persist or drift, not vanish.
3. The warmup0 marginal is the SAME signature seen on tonight's box-3 anchor
   (+2.4e-3, mains in-band) where NO treatment was in play — the negative
   control. The excess is mechanism-independent.

## PATTERN RECORD — WARMUP0 CROSS-BOOT TRANSIENT (for the scorecard's
## numerics disclosure; cite as WARMUP0-TRANSIENT-PATTERN)
Observed tonight (2026-08-10), cross-boot comparisons vs house band
(<=2e-3 pass / >5e-3 stop):
| # | comparison | warmup0 | mains | treatment in play |
|---|---|---|---|---|
| 1 | box-3 anchor qualification (cross-box) | +2.4e-3 marginal | in-band | NONE (control) |
| 2 | A-v3 VERIFY=0 canary (same-box, box 1) | +2.8e-3 marginal | 0.9–1.8e-3 in-band | fixa_v3 |
| 3 | 08-09 318g61w anchor re-baseline vs qr4ggv3 cross-box refs | recorded "<=2.5e-3 PASS", WINDOW-UNATTRIBUTED | — | none (cross-box) |

(Point 3 cited as "consistent, window-unattributed" per helmholtz's retraction
of the 'three' phrasing; its window attribution is unrecorded, so it is
context, not a counted pattern point.)

CONTRAST (solid, helmholtz 2026-08-10): ARM-0 same-config CROSS-BOOT warmup0
deltas measured 0.7–1.7e-3 — IN-BAND. => The middle-zone excess appears in
CROSS-BOX / CONFIG-SHIFTED comparisons, NOT in same-config cross-boot ones.

SHARPENING POINT (2026-08-10, W3-v3 canary, curie): config-SHIFTED same-box
cross-boot comparison (C′-ON + W3-v3 vs C′-ON refs) with warmup0 +0.1e-3 —
NO excess, near-bitwise class. => config shift alone does not force the
warmup0 excess; the asymmetry must actually change executed numerics paths
(the W3-v3 change is pure scheduling; at warmup the lookahead kicks have not
yet engaged). Consistent with the leading asymmetry-driven candidate and with
the (c2) pre-registered application (a truly inert flag predicts no excess).

SHARPENING POINT 2 (2026-08-10, A-v3 leg-(b) canary, curie): config-shifted
same-box comparison (C′-engaged + A-v3 vs cprime-timed) with warmup0
−1.1e-3 — NO excess, and NEGATIVE (the two excess points were positive;
sign is not one-directional across three no/mild-excess config-shifted
points: +0.1, −1.1, and the W3-v3 point). Combined with the two positive
middle-zone points (box-3 cross-box +2.4, under-armed A-v3 +2.8 — the latter
now known to be a REGIME-shifted comparison), the pattern's sharpest form:
warmup0 middle-zone excess appeared only in CROSS-BOX or REGIME-SHIFTED
comparisons tonight; same-box config-shifted comparisons with the regime
matched stayed in-band at warmup0 (+0.1, −1.1). n=2 vs n=2 — characterized,
not proven; the break condition stands.

Characterization (candidate, n=2 confirmed + 1 contrast):
- Signature: warmup0 delta lands in the 2–5e-3 middle zone, POSITIVE both
  times; main windows settle in-band. One-directional-ish (n=2 — weak sign
  evidence, stated as such).
- Mechanism-independent: point 1 is a no-treatment control with the same
  signature.
- SHARPENED by the ARM-0 contrast: the excess correlates with COMPARISON
  ASYMMETRY (different physical box, or shifted config/code-path), not with
  mere cross-boot repetition. Leading candidate: asymmetry-driven early-
  trajectory divergence (hardware- or config-path-dependent kernel/algo
  selection visible from the first datum). The pure loss-magnitude
  amplification candidate is DEMOTED: ARM-0's same-config warmup0 carries the
  same ~12.3 loss magnitude yet stays in-band, so magnitude alone does not
  produce the excess (it may still modulate the size of an asymmetry-driven
  delta).
- Standing disposition: the house band is NOT re-thresholded post-hoc.
  warmup0 marginals matching this pattern are adjudicated with the pattern as
  recorded context; mains remain the carrying windows.
- BREAK CONDITION: the pattern is falsified (and re-adjudication triggers) if
  any future warmup0 exceeds 5e-3, or mains leave band, on a matched-datums
  comparison.
- PRE-REGISTERED APPLICATION (context, not a bar): the upcoming (c2) DP2
  flag-ON vs flag-OFF A/B is a config-shifted comparison in form, but the
  inertness premise says the flag changes NO executed kernel at equal counts.
  If the premise holds, (c2) is behaviorally same-config and the ARM-0
  contrast predicts NO warmup0 excess; a middle-zone warmup0 on (c2) would
  itself be evidence the 'inert' flag changes executed code paths and would
  warrant a look. The house band governs (c2) regardless; this is a
  diagnostic reading, not a threshold.
