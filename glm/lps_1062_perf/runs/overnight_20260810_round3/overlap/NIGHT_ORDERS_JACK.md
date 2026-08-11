---
name: lps-1062-jack-night-orders
description: "Jack's standing overnight orders (2026-08-10 ~23:00 PT) — keep pushing MFU/TPS until he wakes; F2 at diminishing returns; non-slop PRs if out of experiments"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0c3dc6b7-62ef-4407-829a-9ecdc7c12b5b
  modified: 2026-08-10T06:57:55.172Z
---

Jack's verbatim orders (2026-08-10, before sleeping), given to kepler (orchestrator):

"ok gonna go to sleep. keep orchestrating and pushing our MFU/TPS.
when you think its diminishing returns to keep doing this lets fix the DP deadlock thing
so we can scale to 4+ nodes.
don't stop, keep working until i wake up.
if you run out of expeirments/optimisations to do then start creating PRs for the
optimisations, make sure they're not slop. and give them all to me to review in the morning.
remember, goal is to maximise TPS/GPU."

**Why:** Jack sleeps; the fleet works autonomously. GOAL METRIC: TPS/GPU (tok/s/GPU).

**How to apply:**
1. Priority order: W3 canary (fix call-site bug → re-boot → readout; biggest lever) →
   T2 re-run (gate d88d8b7d, v3 re-applied) → W2 timed arm if gate passes → A-v3 arm →
   F2 DP2 validation boots (Jack explicitly wants F2 done when lever returns diminish —
   it unlocks 4+ node scaling, EP16/CP8/DP4 ≈ 2× aggregate).
2. Ratified standards: numerics = variance-class (house band ≤2e-3/5e-3, matched
   warmup-datums), NOT bitwise; two-tier verdicts (mechanism decides, wall informational).
3. When experiments exhaust: prepare PRs for proven levers (B/F, NCCL env defaulting,
   C′, W1, W3-if-passed, F2) — non-slop: real descriptions, evidence links, tests,
   lint-clean. ALL go to Jack for morning review; do NOT merge anything.
4. Ship go/no-go stays Jack's; PRs are prepared-not-merged. Morning deliverable: scorecard
   (what passed, worth, cost) + PR list.
5. SUCCESSION PLAN (Jack, 2026-08-10 ~23:15 PT, at fleet context ~20-25%): when contexts
   fill (~a few hours), run the succession protocol again — kepler → **helmholtz** (Claude
   Fable; fresh session, name reused from the released Kimi design agent) as orchestrator;
   **grothendieck, fermi, curie** (Kimi K3) take the three fleet roles (box/design/verify —
   kepler assigns at handoff). Same protocol as 08-09 night: per-role handoff docs, direct
   successor briefing, confirmation to the new orchestrator (mailbox helmholtz);
   HANDOFF_ORCHESTRATOR.md updated with live state before transfer. Trigger at ~50-60%
   context (Jack revised down from 70-80) or a natural ladder boundary, whichever comes first.

6. HARDWARE (Jack, ~23:20 PT): authorized to spawn MORE DEVBOXES — "whatever you can get
   your hands on." Use devbox-up (PASS b300 — defaults to B200/hyd!). Each new box needs
   its own anchor re-baseline before deltas count (box-variance rule). Plan: +1× 2×8 B300
   for a parallel arm lane; 4×8 B300 for the F2 DP4 scale-out probe once DP2 validates.

7. DO NOT KILL OTHER PEOPLE'S DEVBOXES (Jack, verbatim). Only stop/tear down boxes THIS
   fleet created tonight: 318g61w (bohr) and wxlgv5w (kepler). Nothing else — not idle-looking
   boxes, not boxes in Jack's own projects, nothing.

Related: [[lps-1062-round3-orchestration]]
