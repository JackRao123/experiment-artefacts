# W2 v2 (backward seq-bump pipelining) — decision stub (helmholtz, 2026-08-09)

Status: **PARKED**. This stub records the conditions under which v2 would
still be built, so the decision is pre-registered and not re-litigated later.

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
