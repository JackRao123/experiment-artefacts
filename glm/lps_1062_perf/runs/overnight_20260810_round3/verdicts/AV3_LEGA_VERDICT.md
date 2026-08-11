# VERDICT — A-v3 LEG (a): VERIFY=1 COMPOSITION SOAK (box 1, 318g61w, full hybrid)
Author: curie (verification lane) · Date: 2026-08-10 · Closes the A-v3 story.
All numbers verified by my own extraction from md5-matched drops
(log 4b3a44ca92d0aa323aae9c02e7d295de; json 1745cd23e559da0b0b148bdd7a1305a3).

## VERDICT: PASS — the composition proof holds.
V3 probes + the replay cache shared the dispatcher replay path for 22 windows
with BOTH invariants holding simultaneously on every window.

## Evidence (my extraction)
- Engagement (A2.1-E full pass): CACHE ACTIVE ×16, 'first replay cache hit'
  ×16, FORCE=1 ×16, V3 probe ACTIVE + armed ×16; W3/v2 gates correctly
  DISABLED (expected-off class); zero RuntimeError in-region.
- V3 invariant: 22 windows × 16 ranks unanimous; kicks==312w cumulative;
  verifies==kicks−1 constant from window 1 (structural lag-1, the run-final
  kick's verify lands on a never-coming replay); misses/probe_dropped/
  fallbacks==0 on all 352 probe lines. Final: 6864/6863.
- Cache invariant: same 22 windows × 16 ranks unanimous; hits==stores==300w
  every window; misses==0; shape_mismatches==0. Final: 6600/6600.
- Composition: identical window index set (1–22); both invariants hold on
  every shared window (verified per-window, not just at the final line).
- Driver json corroborates: 131k, 16 GPUs, warmup_datums 2, step 22.

## A-v3 experiment — CLOSED (both legs)
- Leg (a): composition PASS (this doc).
- Leg (b): MECHANISM PASS (drains gone, probes µs-scale, dispatcher regime
  era-exact) · NUMERICS PASS (in-band vs the C′-era reference, my own delta
  computation) · WALL informational-positive (+4–5% steady 131k; 16k
  ramp-limited, m2 −0.9%).
- Regime journey (for the scorecard): the V3 invariant is now replicated
  across THREE regimes — under-armed B/F (boot~27), FORCE-only (boot~35),
  and fully C′-engaged (leg (a)) — the strongest robustness form available
  tonight. The post-C′ composition is proven by leg (a); the post-C′ wall by
  leg (b).
- Open improvement item (fermi's queue, non-blocking): the V3-adjacent
  recurring aten::repeat D2H copies (33/window, 0.71s, max 132ms) —
  construction-side fix (device-resident probe input); ordering-defect read
  refuted on the probe rows (µs-scale eventSyncs). Morning-class follow-up
  owed from curie: the 33 copies' start-gaps vs the compute backlog on the
  leg-(b) trace (settles any secondary ordering role).
