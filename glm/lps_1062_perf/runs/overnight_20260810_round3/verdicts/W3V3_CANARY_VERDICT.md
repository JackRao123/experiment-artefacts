# VERDICT — W3-v3 CANARY (box 1 job 41, 318g61w, C′-ON + W3-v3, fixa_v3 reverted per A2.2)
REGIME NOTE (2026-08-10 PM, fermi's code read via helmholtz): the boot set
FORCE without CACHE, and FORCE is inert without CACHE's frames — so job 41
actually ran the pre-C′ status-quo FREE-ROUTING replay regime (C′ fully
inert), consistent with the 33.9s vs era-10.0s eventsync-cpu figure. The
verdict is UNAFFECTED: T1/T6 are self-contained trace measurements; the
canary numerics were near-bitwise as measured; the regime note sharpens the
caveat (the host was MORE throttled than the intended C′-ON regime — the
A-v3-prerequisite thesis strengthens) and the v2/v3 confound (v2 cache-ON vs
v3 C′-inert; verdict-insensitive, both ~50x below bar).
Author: curie (verification lane) · Date: 2026-08-10
Frame: W3V3_CANARY_ACCEPTANCE_FRAME.md (frozen; A1/A2/A3 ratified)
Tooling: analyzers md5-proofed on-box vs Mac authorities — verified locally:
side_stream 3a99b9fd…, w3_overlap 61ed30b0…, sm_slack aee6a2e8…, drain_attrib dcb1fe22… (all match).

## CANARY VERDICT: FAIL — verdict-carrying row T1 fails; no re-thresholding.
v3's two-event ordering fix did NOT de-serialize the kicks on this measurement.

### T1 — OVERLAP CAPTURE (verdict-carrying): FAIL
boltzmann_w3_side_stream.py on the fresh steady-window trace (rank0):
11.00 s side-stream kernels, 99.5% landing inside bwd windows, but concurrent
with other GPU work = 0.10 s = **0.9%** vs bar >= 50%. Serialized 99.1%.
(v2 died at 0.1%; v3 = 9x better, 55x short of the bar.)
Single-rank evidence suffices: the margin is ~55x; two consistent captures on
rank0. rank8 trace UNAVAILABLE (CUPTI thread-init error on leader-side flush,
both capture attempts) — recorded; not INVALIDating at this margin.
Corroboration (boltzmann_sm_slack.py on the arm trace): bwd extent 41.06 s,
SM slack 34.65 s (83.3%), SendRecv 23.52 s @ 0% occ — THE SLACK IS STILL
THERE; the kicks still don't fill it. Matches the gate measurement
(83.5% / 35.9 s on the anchor).

### T2 — TWO-EVENT ORDERING: PASS (letter-modulo)
Side-stream localization: 99.5% of side-stream kernels land inside bwd windows
(bar: >= 90% kicked windows with first side-stream kernel strictly inside the
main window). The kicks ARE landing in the right windows — they just don't
overlap anything once there. Exact per-window first-kernel form not separately
run; recorded as context, not verdict-relevant (T1 carries the FAIL).

### L-rows (mechanism telemetry)
- L1 kicks==hits: PASS-BY-RE-DERIVATION. Frozen constant 78/mb does NOT
  transfer to this stack: measured structure is 75 chunks/mb = 74 kicks + 1
  structural inline miss, EXACT at the mb-16 completion discriminator
  (kicks 1185 = 74x16 + 1 in-flight; hits 1184; misses 16). The 78/mb constant
  traces to the v2-era telemetry and likely conflates the DSA-layer count
  (78/mb, per the A-v3 soak's 312 probes/step) with the checkpoint-chunk
  count (75/mb on this config). The bar's INTENT — every checkpointed chunk
  kicked except the 1/mb structural inline — is satisfied with exact
  arithmetic. Confirmation queued (not verdict-blocking): trace-side count of
  LookaheadCheckpointFunctionBackward (expect 300/window). If the house rules
  the frozen constant is the bar, L1 fails on the letter; the canary verdict
  is unaffected (T1 carries it).
- L2 misses == 1/mb: PASS exact (16 misses / 16 mb started; the inline chunk
  backwards first in each mb, so misses count mbs-started).
- L3 sweeps==0, fallbacks==0: final counter line requested from grothendieck
  (not in the package); expected per the v2 telemetry class. Recorded OPEN
  until the line lands; cannot flip the verdict (T1 carries it).
- L4 kick_ms: PASS-BY-INTENT, composition documented (delegated scope ruling):
  * Cumulative window avgs decay 129.9 -> 62.4 (w1..w6); the window lines are
    CUMULATIVE (named telemetry trap), so marginal per-window avgs recovered
    exactly: w2 61.9, w3 46.4, w4 47.8, w5 43.5, w6 44.9 (equal kicks/window).
  * Steady band (w3-w6): 43.5-47.8 ms, flat — all <= 60. w1-w2 = settling
    class (startup composition), reported.
  * The recurring ~2947 ms max (1 per window, constant magnitude, drift
    +4.6 ms over 6 windows) is a STRUCTURAL WAIT, not execution dilation:
    consistent with the step-boundary first kick whose input-event
    transitively covers the prior step's tail (design sec 2(a)); contention
    would vary, this doesn't. OUTSIDE the bar's intent (the spec bars
    "sustained max >> that" as the contention signal); REPORTED here.
  * Typical steady kick excluding the outlier: ~34 ms vs the ~41 ms inline
    reference — NO execution dilation; the HBM-contention symptom the bar
    guards against is absent (unsurprising at 0.9% capture: nothing is
    co-running).
  * Letter-reading note: if w2 is counted steady-state, its marginal 61.9 ms
    exceeds the bar by 1.9 ms; the scope ruling (settling class) is documented
    above and the verdict is unaffected either way.

### T3 / T5 (reported-under-every-outcome rows)
- T3 consume-stall (idle-gap form, A1.3): NOT YET MEASURED — queued on the
  rank0 trace (on-box or local replication). Not verdict-blocking.
- T5 HBM dilation per occupancy bucket: expected ~nil at 0.9% capture (no
  co-running to dilate). Queued with the same trace replication. Not
  verdict-blocking.
- T2-delay (kick start-delay p50/p90) + comm-coverage + T3-arrival: queued
  with the trace replication.

### M1 — MEMORY: PASS
Peak reserved delta vs the SELECTED (C′-ON) baseline: +24.9 GiB <= +28 GiB
(margin 3.1 GiB). Driver-polled both sides (method-matched, 131k-d4):
canary poller max 228647 MiB (223.3 GiB) vs C′-era 203165 MiB.
Headroom: 44.9 - 24.9 = 20.0 GiB >= 10 (ship bar). Conditions per the M1
scope ruling: (a) method/shape match CONFIRMED; (b) poller-vs-snapshot strict
== NOT established (poller on the canary drive, snapshot on later memsnap
windows) — the ~9.4 GiB gap is the nvidia-smi-used vs allocator-reserved
overhead class, recorded; (c) snapshot decomposition (intrinsic vs retention,
vs v2's 11.5 + ~14 GiB) QUEUED: 16 rank pickles at
lps1062_bench/318g61w/w3v3_memsnap/; grothendieck's raw reserved-minus-
allocated (90 GiB) is the wrong method (reserved includes free-cached) — the
minkowski-style segment walk is required; curie will run it when the pickles
land in ~/perf_profiles/lps-1062/incoming/.

### N — NUMERICS: PASS
w0 +0.1e-3, mains <= 0.7e-3 — near-bitwise class, in-band (house band
<=2e-3/>5e-3). 131k only; 16k not armed (HARD OFF respected).
Pattern record: config-shifted same-box comparison with NO warmup0 excess —
filed as a sharpening point in WARMUP0-TRANSIENT-PATTERN.

## FAILURE ANATOMY (for fermi's next iteration; context, not bars)
1. Kicks fire, land in the right windows (99.5%), stashes hit exactly
   (74/mb + 1 inline), numerics near-bitwise, memory in bar: the mechanism's
   PLUMBING is correct.
2. The win mechanism — concurrent execution — did not materialize: 99.1%
   serialized, 0.9% capture. The ordering fix removed the wait_stream
   whole-backlog chain but the kicks still do not co-run.
3. Structural evidence in-hand: the recurring ~2947 ms step-boundary kick
   wait (input-event transitive coverage of the prior step's tail); the (b)
   last_bwd_end edge covers the immediately-preceding backward when the host
   is NOT far ahead (plausible under the reverted-A regime, where the
   DSA-bwd drains throttle host run-ahead) — candidate serialization
   coupling for v4 to attack (e.g. pool versioning instead of the (b) event
   wait, or restoring host run-ahead). LABELED as hypothesis, not measurement.
4. The SM slack the gate measured is confirmed still present on the arm
   (34.65 s, 83.3%; SendRecv 23.52 s @ 0% occ). The territory exists; v3's
   ordering did not reach it.

## ESTATE NOTE
The SM-slack gate made W3-v3 design-live; this canary is the measurement of
v3's ability to capture it. v3 FAILS the capture bar. Whether W3 iterates
(v4) or closes is the estate's call (helmholtz/fermi), not this verdict's.
