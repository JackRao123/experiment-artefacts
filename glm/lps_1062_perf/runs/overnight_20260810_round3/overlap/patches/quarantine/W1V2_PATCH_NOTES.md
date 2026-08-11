# W1-v2 — BT_MOE_PROBS_A2A_COMM_V2 paired dispatch node (helmholtz, 2026-08-09)

## DISPOSITION: ARCHIVED-UNSHIPPED (2026-08-09, fibonacci)

**The pre-registered ship-gate below resolved NEGATIVE with data in hand.**
hilbert's launch-vs-kernel-start cut answered the grad-arrival question
directly: the probs reverse's LAUNCH is already ~24 ms early on an idle
stream; the kernel waits ~41 ms for the **probs grad itself**, produced
post-fc2-dgrad deep in the layer's backward chain (behind the expert dgrad
pack). The late grad is the probs' — exactly the readiness-drift case under
which the paired node would chain the tokens reverse to that late grad
(correct but potentially slower than v1). Per the gate, the paired node is
the wrong tool; both v2 candidates are dead. The real lever (backward-chain
reorder: emit d(probs) right after fc2-dgrad and issue its reverse ahead of
the remaining expert dgrads) recovers only ~1–1.5 s/step and is subsumed by
W3's lookahead — parked as option 6 in `../../W2V2_DECISION_stub.md`
(W3-absence fallback only). W1 ships as v1; the residual (~0.5 s/step on
this box) is understood, bounded, and documented in W1_PATCH_NOTES.md.

The patch (`runs/overnight_20260810_round3/overlap/patches/quarantine/w1v2-paired-dispatch.patch`) is kept for reference only — do not
apply. The stopped variant's tests + the seq-order probes are preserved in
`../../tests/` (`test_w1v2_paired_dispatch_STOPPED.py` — skips cleanly on the
ship tree; `w1v2_probe.py`, `w1v2_probe2.py`).

## What

`runs/overnight_20260810_round3/overlap/patches/quarantine/w1v2-paired-dispatch.patch` — replaces W1's two separate deferred-A2A nodes
(tokens on the EP comm, probs on the second comm) with ONE paired autograd
node (`_AllToAllDeferredWaitPair`): forward issues both collectives
asynchronously (waits still deferred to the caller, shared-expert fc1 still
between issues and waits); **backward issues BOTH reverse A2As back-to-back
inside the single node's backward** — the autograd engine's execution order
can no longer separate the two reverses.

## Why this form (the seq-bump was retracted)

fibonacci's prescribed W1-v2 was a seq-bump of the probs-reverse to sit
adjacent to the tokens-reverse. **Probe evidence (accepted; prescription
retracted):** construction order already assigns the two nodes adjacent
sequence numbers (tokens=N, probs=N+1), and the measured engine execution
order on the real deferred Function is already `[consumer.bwd, probs-a2a.bwd,
tokens-a2a.bwd]` — the probs reverse runs FIRST and adjacent. The seq-bump
was therefore inert (a no-op patch that passes parity silently — the v1
trap). The paired node removes the engine-ordering degree of freedom
entirely and is correct under every ordering hypothesis. Probe:
`../../tests/w1v2_probe*.py` (kept with the suite).

## Pre-registered PERF risk (fibonacci, binding)

A multi-output Function's backward fires only when ALL its output grads have
arrived. Under readiness-drift hypothesis (a) — the PROBS grad arriving late
via the sort-with-probs backward chain — the paired node would DELAY the
tokens reverse to the probs grad's arrival: correct, but potentially slower
than W1-v1. Hence the ship-gate below.

## Ship-gate (BOTH measurements must agree with the design before box time)

1. **hilbert's launch-vs-kernel-start cut** confirms the mechanism is
   engine-issue-order (issue late), not kernel-start-late (channel
   assignment). If kernel-start-late: W1-v2 is moot; the fix is
   stream/channel placement, not autograd.
2. **Grad-arrival measurement** (fibonacci's addition — hilbert to add
   grad-producer timing to the cut): WHICH incoming grad (tokens vs probs)
   is ready later in the backward window. If probs-late dominates, the right
   fix moves UPSTREAM (make the sort-bwd produce the probs grad earlier, or
   leave v1 alone and let the exposure ride) and the paired node is the
   wrong tool.

## Trace-checkable predictions (pre-registered)

- bwd-window probs late-start fraction → ~0 (the defect signature).
- **tokens-reverse start NOT regressed vs W1-v1** (the readiness-coupling
  guard — if this moves, the paired node is slower and must not ship).
- Values/bitwise: unchanged (same messages, same communicators, same waits).

## Gate + base

- `BT_MOE_PROBS_A2A_COMM_V2=1`, requires `BT_MOE_PROBS_A2A_COMM=1`; default
  OFF; one-time WARNING lines (ACTIVE / ignored-because-v1-off). W1's
  per-window counters cover the paired dispatches.
- Base: the post-W1 tree (`../w1-probs-a2a.patch` applied over 57efae08b +
  FIX A/B/F + FIX C). Independent of W2/W3 (the token_dispatcher.py hunk was
  isolated to the W1 branch; verified the v2-only construction compiles).
  Apply: `patch -p1 < runs/overnight_20260810_round3/overlap/patches/quarantine/w1v2-paired-dispatch.patch`. Verified: apply-clean +
  byte-for-byte vs the author's construction.

## Tests

`../../tests/test_w1_probs_a2a.py` extended (ALL PASS on Mac gloo): sec6 — pair
fwd values == reference; pair bwd grads == reverse-A2A references (both
tensors); world_size==1 bypass; sec7 — v2 gate requires v1. The existing
T1/W2/W3 suites are unaffected (re-run green).
