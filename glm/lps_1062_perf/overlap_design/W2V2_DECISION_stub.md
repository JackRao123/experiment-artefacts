# W2 v2 (backward seq-bump pipelining) — decision stub (helmholtz, 2026-08-09)

Status: **UN-PARKED 2026-08-10** (minkowski's call per rule 3, ratified at
succession; recorded by fermi) — **sequenced behind W3-v3**; see the
disposition at the bottom. This stub records the conditions under which v2
would still be built, so the decision is pre-registered and not re-litigated
later.

## What v2 is (recap)

W2 v1 ships fwd/replay pipelining with a status-quo (issue+wait) backward.
v2 would pipeline the backward itself: bump the per-group combine/dispatch
A2A Functions' autograd sequence numbers (the
`set_tensor_grad_fn_sequence_sr` mechanism, shared_experts.py:585-593) so
both reverse A2As issue before the MLP backwards, with wait-aware
Unsort/Slice consumers and events carried on grad tensors. Standalone value:
−1.2…−1.5 s/step (≈6–8 ms of the ~23 ms bwd A2A exposure per layer-bwd ×
300). Design: DESIGN_helmholtz.md §3.4.

## Why W3 mostly subsumes it

W3 (lookahead recompute) covers the backward phase by overlapping
recompute(L−1)'s compute with bwd(L)'s A2A chain — the bwd-phase A2As are
hidden behind the *neighbor layer's* compute regardless of their intra-layer
scheduling. With W3 active at capture ratio c, v2's residual value ≈
(1 − c) × its standalone win, minus nothing (v2 reorders work inside a
window W3 already fills — strictly non-additive where W3 captures).

## Conditions that would un-park v2 (any one suffices)

1. **W3 blocked at V2**: the CDMC-unset probe (ARM 4) shows a
   correctness/perf regression → W3's capture collapses (head-of-line
   blocking at =1) → the bwd phase needs intra-layer cover and v2 is the
   fallback mechanism.
2. **W3 memory-gated off at 16k×d32** (+8–12 GiB lookahead activations
   exceed headroom there) while W2 stays on → v2 for that shape only.
3. **W3 measured capture < ~50%** of bwd-phase comm (dependency tails, host
   stalls without FIX C, DSA-holder surprises) → v2 recovers part of the
   residual exposure. Measurement: the W3 canary's overlap-% counters +
   trace (SendRecv time inside bwd-phase windows).
4. **W3 production instability** (thread/stream/allocator fragility beyond
   the CPU-proven model) → park W3, take v2 as the simpler bwd mechanism.
5. **Boundary residue** (only if measurements show it matters): W3 never
   kicks the last chunk's recompute (inline) and the first chunk's backward
   has no kick — v2 would cover those windows too, but they are O(1/78) of
   the phase; do not justify v2 on this alone.

## Decision rule

After the W3 canary: if W3 on target shapes shows bwd-phase comm overlap
≥ 50% and no per-shape disable → **park v2 permanently**. Else implement v2
(scope: the two seq-bumps + wait-aware wrappers + event-on-grad-tensor
plumbing, ~1–2 days incl. tests; the v1 bitwise argument carries over
unchanged — v2 is pure scheduling).

## DISPOSITION (2026-08-10, fermi — recording minkowski's call + the v3 re-entry)

**Decision rule OUTCOME: else-branch fired.** The W3 canary measured bwd-phase
comm overlap ≈ 0 % (kicks 99.9 % serialized) — decisively < the 50 % bar
(rule 3). W2-v2 and option 6 are **UN-PARKED** on that reading.

**Resource distinction (minkowski, ratified here):** W3-v2 failed for want of
ordering, not comm-window cover; regardless, option 6 and the v2 seq-bump are
**comm-resequencing** plays — they add no compute and hide existing comm
behind existing bwd windows. The SM-slack gate (boltzmann's measurement,
PASSES: 83.5 % of bwd kernel time <10 % occupancy, ~35.9 s slack/step,
SendRecv windows ~0 % occ) gates W3-v3 only; it does **not** apply here.
Neutral-to-supportive either way: comm-hiding wants busy windows, and the
0 %-occupancy SendRecv windows are exactly where a resequenced reverse lands.

**Sequencing (helmholtz, 2026-08-10):** W3-v3 (input-dependency-only kick
ordering) re-entered as priority 1 on the positive gate. Per this stub's own
subsumption note, W3 covers the bwd window regardless of intra-window order,
so where v3 captures, option 6 and v2 are strictly non-additive. Therefore:
**option 6 and W2-v2 remain un-parked but are built ONLY if W3-v3 fails its
canary** (or captures < 50 % again). If built, option 6 first (small,
mechanism understood), v2 behind it (stub scope below; the v1 bitwise
argument carries over).

**Option-6 disposition card (pre-registered, so a future build is not
re-litigated):**

- **Design (two parts, both required):** (i) split the probs grad path out of
  the fused `sort_chunks_by_index_with_probs` backward so d(global_probs) is
  produced right after fc2-dgrad instead of being held hostage until the
  token path (fc1-dgrad pack) completes — the sort is a row permutation, so
  the probs edge is an exact inverse gather; (ii) issue the probs reverse A2A
  on comm 2 at that early point with a **deferred wait** (event/work carried
  on the grad tensor to its consumer in the router-path backward) — an inline
  wait would just relocate the exposure. Locus: TEGroupedMLP backward chain +
  the dispatcher's probs-reverse issue point; NOT the a2a Functions.
  Verification reuses `tests/test_w1v2_paired_dispatch_STOPPED.py` +
  `w1v2_probe*.py`.
- **Expected win:** ~1–1.5 s/step model (one latency-bound probs reverse per
  layer-bwd, partially absorbed); measured W1 residual ~0.5 s/step on box 1
  (wall, informational per two-tier). Mechanism target: bwd-window probs
  late-start fraction → ~0 (trace-checkable).
- **Numerics class:** pure scheduling — same kernels, same bytes, same
  reduction orders (the probs inverse gather is row-exact; no accumulation
  reorder at any junction). Expected bitwise at the in-process gate; ship bar
  is the variance-class house band regardless.
- **Memory cost:** ≈ 0 — the probs payload is KB-scale (latency-bound); the
  only lifetime change is the reverse's recv buffer arriving earlier within
  the same layer-bwd window. No new graphs, no new pools.

## Option 6 (folded 2026-08-09): backward-chain reorder for the probs reverse

From the W1-v2 post-mortem (hilbert's launch-vs-kernel-start cut): the probs
reverse's launch is ~24 ms early on an idle stream; the kernel waits ~41 ms
for the **probs grad itself**, produced post-fc2-dgrad deep in the layer's
backward chain (behind the expert dgrad pack). Neither W1-v2 candidate
addressed this (the seq-bump was inert; the paired node would have chained
the tokens reverse to the late grad — its ship-gate resolved negative and it
is archived-unshipped, `patches/W1V2_PATCH_NOTES.md`).

The actual lever: **emit d(probs) right after fc2-dgrad and issue its
reverse ahead of the remaining expert dgrads** — i.e., reorder the backward
chain so the probs-grad producer runs early (the scaled-activation backward
path) instead of late. Estimated recovery: only ~1–1.5 s/step (the exposure
is one latency-bound probs reverse per layer-bwd, partially absorbed). It is
**subsumed by W3's lookahead** (the bwd window overlaps the next layer's
recompute regardless of intra-window order), so it shares this stub's
un-park conditions: build only if W3 is absent/disabled on the target shape
AND the measured W1 residual (~0.5 s/step on this box) justifies the
backward-graph surgery. Scope note if un-parked: the reorder lives in the
TEGroupedMLP backward chain (activation-bwd earlier) + the dispatcher's
probs-reverse issue point — NOT the dispatcher's a2a Functions; the preserved
W1-v2 probes/tests (`tests/test_w1v2_paired_dispatch_STOPPED.py`,
`w1v2_probe*.py`) are reusable for its verification.
