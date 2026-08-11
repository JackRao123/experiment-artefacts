# FINDING — FIX C′ replay-elimination NOT ENGAGING in tonight's C′-ON boots
Author: curie (verification lane) · Date: 2026-08-10 · Status: CONFIRMED by
same-checker arbitration; root cause OPEN pending the FIX C counter lines.

## Arbitration (grothendieck's suggestion, executed)
Same checker (Mac authority d4418e58636857548d28c3c4fb5e898f, the exact file
md5-proofed on-box), same profile (post-patch-BFC-4mb131k), same class of
steady capture:

| row | C′-era trace (round3/arm-cprime rank0) | tonight's re-armed A-v3 trace |
|---|---|---|
| eventsync_calls | 300 PASS | 912 FAIL |
| eventsync_cpu_s | 10.02 PASS | 33.9 FAIL |
| eventsync_gt1ms | 300 PASS | 600 FAIL |
| dispatcher replay_calls | **0** PASS | **300** FAIL |
| dispatcher_allgather_calls | **300** PASS | **600** FAIL |
| dtoh_pinned_count | 1914 PASS | 3114 FAIL |
| (A-v3 rows on the re-armed trace) | n/a | ALL PASS (a_calls==312, a_cpu 1.8ms, a_gt1ms 0, nonzero_gt5ms 8, streamsync 461) |

C′-era: ALL PASS. => The pins are NOT stale. Tonight's C′-ON boots genuinely
differ from the C′-era boot: the dispatcher replay eventSyncs are present at
the full 300/step (75 MoE x 4 mb) and the replay preprocess all-gathers at
600 — i.e. FIX C′'s replay cache eliminated NOTHING, despite the C′ gates
being armed (7-gate arm-check pasted, fresh history-symmetric boot).

## RATIONALE CORRECTION (2026-08-10, fermi's code read via helmholtz;
## recompute.py:130-131)
With CACHE off, `_wrap_checkpoint_chunk_pass` no-ops => no frames =>
`_routing_force_frame()` returns None => **FORCE WAS ALSO INERT**. The three
affected boots (job-41 W3-v3 canary, A-v3 re-arm soak, A-v3 re-arm VERIFY=0)
did NOT run a "FORCE-without-CACHE" regime — they ran the **status-quo
FREE-ROUTING replay regime**: C′ fully inert, both halves. The env-grep table
(FORCE=1 present, CACHE absent) is still the correct staging description; the
regime semantics are corrected: FORCE's mechanism depends on CACHE's frames,
so CACHE-off implies FORCE-inert.
Corrected survival grounds (superseding "cache is value-neutral, FORCE
preserves routing"):
- The canaries' MEASURED in-band deltas are direct evidence (W3-v3 near-
  bitwise: w0 +0.1e-3, mains <=0.7e-3; A-v3 re-arm: w0 -0.3e-3, mains
  <=1.5e-3) — whatever regime the boots ran, the comparisons landed in-band.
- Free-routing IS the pre-C′ status-quo class every era baseline ran — the
  regime the boots actually ran is the known, band-calibrated class.
- Supporting: the C′-era's own verify soak (6300/6300 green) proved the
  engaged cache value-identical to free-routing recompute.
Sharpened W3-v3 regime caveat: the canary host ran the full pre-C′ replay
regime (consistent with the 33.9s vs 10.0s eventsync-cpu figure) — the
A-v3-prerequisite thesis (W3 capture needs run-ahead) strengthens again.
The v2/v3 confound sharpens correspondingly: v2 (boot~23) was C′-ENGAGED vs
v3 (boot~33) C′-INERT — still verdict-insensitive (both ~50x below the bar).

## Classification
The house's own silent-inert class (process lesson: "activation telemetry
mandatory for env-gated patches"). The A2.1 arm-check proves gates ARMED
(env/markers); it does NOT prove FIX C ENGAGED — the engagement evidence is
the log-based activation lines: one-time 'armed' + 'first replay cache hit',
and the per-window 'BT_MOE_DISPATCH_REPLAY_CACHE window' counters
(hits==300/step, misses==0, shape_mismatches==0 expected).

## Open root-cause candidates (need the re-armed boot's FIX C counter lines)
1. Cache armed but never hitting (misses==300/step) — e.g. routing-shape key
   mismatch on this stack (the 0626 verify FAIL was exactly a
   num_tokens_per_local_expert_dev metadata mismatch; a silent variant would
   miss without the verify instrumentation).
2. The dedicated clone's shared-tree byte-state predates C′ (code provenance,
   not env) — the arm-check checks markers, not code content.
3. Interaction: A-v3's async probes altering dispatcher metadata timing
   (weakest candidate — the soak proved probe self-consistency, and the
   under-armed boot showed the same rows with A-v3 present and C′ off).

## Consequences (verification-lane rulings)
- The A-v3 re-arm SOAK's "post-C′ regime" composition claim is HELD: the soak
  proves V3 self-consistency in whatever regime it ran; if FIX C′ was not
  engaging, the regime was C′-armed-but-not-engaging — NOT the post-C′ regime
  the experiment was for. Canary (in-band: w0 -0.3e-3, mains <=1.5e-3) and
  timed (131k summary +5.3% with C′'s slow-w0 caveat; 16k still-ramping) are
  likewise regime-qualified. All three re-arm verdicts stay PENDING until the
  engagement question resolves.
- W3-v3 canary regime caveat (for the scorecard; verdict UNAFFECTED): the
  W3-v3 trace also shows replay-class eventSyncs at 300 — under W3 the
  W3-aware split counts the kicked recompute's dispatcher syncs as replay, so
  this is EXPECTED under W3 and is NOT independent evidence about FIX C′
  there. T1 FAIL stands on its own measurement. The T6 causal chain's
  "host drain-throttled" mechanism is regime-robust (measured, not assumed);
  if anything a non-engaging FIX C′ made the host MORE throttled (33.9s
  eventsync cpu vs 10.0 in the C′-era) — the capture headroom under a
  C′-engaging boot is untested and bounded by the T6 structural couplings.
- A2.1 arm-check AMENDMENT CANDIDATE (helmholtz's call): for gates with an
  armed-vs-engaged distinction, the declared marker list must include the
  ENGAGEMENT markers (for FIX C: 'first replay cache hit' + window counters
  hits==300/step), not only the armed line. Tonight's re-arm passed the
  arm-check yet the gate was inert.
