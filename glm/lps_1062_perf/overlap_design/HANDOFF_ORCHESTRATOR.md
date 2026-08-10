# Orchestrator handoff: fibonacci → kepler (2026-08-09/10 night)

You (kepler) inherit the LPS-1062 round-3 GLM-5.2 perf push mid-ladder. This file + the
memory entry `lps-1062-round3-orchestration` + `TRACE_ACCEPTANCE.md` (same dir) are the
authoritative state. Jack's mandate: reduce step time; **constraints: config must support
131k seq with ≥10 GiB/GPU headroom; loss numerically the same** (see the numerics
adjudication below — surface it to Jack in any final report; he has not yet ratified it).

## Fleet succession (message via ~/.agents/scripts/send-message.sh <name> "...")
- **kepler** (you) ← fibonacci: orchestrator. Your session name for replies is kepler.
- **fourier** ← bohr: box ops, benches, F2. Bohr is handing off the LIVE box mid-run — see below.
- **minkowski** ← helmholtz: design + W2/W3 ownership.
- **boltzmann** ← hilbert: adversarial verification, FIX C′/A-v3, trace analysis, check_acceptance.
Each predecessor writes its own handoff doc and briefs its successor directly; they were
instructed to confirm completion to you. Comms arrive as 📨-injected turns.

## Where the ladder stands (exact order; two-tier verdicts per TRACE_ACCEPTANCE.md)
Box: **318g61w** (2×8 B300, ali; ssh tj-318g61w / tj-318g61w-1; HTTP server on node1;
runs trainers 0e0b65a6; **CDMC UNSET** — verified, and so was qr4ggv3, don't re-litigate).
Anchors (this box, binding baseline): 131k×d4 ~705 tok/s/GPU / 46.5 s; 16k×d32 ~703.5.
Patch stack applied on box + Mac (all env-gated, default OFF): B/F/A → fixc → C′
(`fixc_prime_routing_force`) → w1-probs-a2a → w2 v3. W3 + A-v3 patches staged, not applied.

1. ✅ ARM 0 inertness (passed, canary band; cross-boot bitwise floor established)
2. ✅ ARM 1 FIX C → verify FAIL → diagnosed → **C′ pivot GREEN** (soak 20/20, zero raises,
   6300/6300 hits, +3.7 GiB cache, canary in band)
3. ✅ ARM 2 W1 (mechanism PASS; wall −0.5 s on this box; residual fully diagnosed:
   bwd probs-reverse waits ~41 ms for the probs grad (input-readiness) — investigation
   CLOSED, W1 ships as v1, both v2 variants archived-unshipped)
4. ▶ **IN FLIGHT: timed C′ arm** (B+F+W1 vs B+F+W1+C′; bohr was mid-run at handoff —
   fourier picks it up; ship rows: eventsync replay ~0, allgather replay ~0, BFC profile
   PASS, canary band; wall informational)
5. ⏭ W2 t2 gate (`run_t2_gate.sh`, BT_T2_SKIP_FIXC as needed — decide whether to now
   include the fixc variants since C′ is green) → W2 arm (mechanism-primary bars)
6. ⏭ W3 canary (BT_MOE_LOOKAHEAD_RECOMPUTE=1; CDMC precondition satisfied as-is;
   +2–4 GiB at 131k, measure 16k×d32 before enabling there; the biggest lever, −8…−12 s model)
7. ⏭ F2 validation boots (DP2; patch has the kill-switch WARNING; time-box the deadlock
   re-confirmation; main-rebase patch drafted for the ship PR)
8. ⏭ conditional: A-v3 timed arm (`dispatcher_opt/fixa_v3/`, only after C′ timed passes)

## Standing rulings (do not silently re-decide; Jack may veto)
1. **Numerics**: cross-boot bitwise is unattainable (0.7–1.7e-3 same-config boot floor,
   kernel-algo selection). Bitwise bars = IN-PROCESS only (C′ verify mode, W2 t2 gate);
   cross-boot = ≤2e-3 pass / >5e-3 stop, matched --warmup-datums, warmup0 recorded.
2. **Two-tier verdicts**: mechanism bars decide levers; wall magnitudes on 318g61w are
   informational (box has tail-heavy fabric variance ~+1.3 s vs qr4ggv3; ship-magnitude
   claims need prod-grade fabric).
3. W1 stays armed in all subsequent arms; its +1.3 GiB (2nd NCCL comm) and C′'s +3.7 GiB
   are recorded memory costs (68 GiB headroom remains at 131k).
4. Combined-ship precondition: DP2+phantoms+W2 canary before any combined ship (F2×W2
   zero-count interaction).
5. Ship-PR deliverables parked: F2 main-rebase; upstream candidates B/F/C′/W1/W2/W3;
   prod NCCL env defaulting ticket (round-1); DeepEP fabric-qual ticket (round-1).

## Key documents (all under experiment_artefacts/glm/lps_1062_perf/)
- `overlap_design/LEVERBOARD.md` — ranked levers + step anatomy (start here)
- `overlap_design/TRACE_ACCEPTANCE.md` — per-lever bars + every adjudication
- `overlap_design/DESIGN_helmholtz.md` — W1/W2/W3 design + field calibration (§6.1)
- `dispatcher_opt/fixc/DESIGN_FIXC_PRIME.md` — C′ evidence chain (the stack-level fact:
  recompute replay is NOT bitwise vs fwd here; peer-flips propagate via allgather)
- `f2_fix/` — F2 design/review/patches; `dispatcher_opt/fixa_v3/` — A-v3 package
- `overlap_design/W2V2_DECISION_stub.md` — parked fallbacks incl. bwd-reorder (option 6)
- Traces: `~/perf_profiles/lps-1062/round3/` (+ straggler_summary.md with §4 verdict)
- Trace tooling: scratchpad trace_processor_shell v56.1 pattern + `q.py` (TP_ADDR env);
  port 9001 on this Mac is owned by a stale foreign server — use 9101+.

## Watch items
- Box waiter false-positives during boots (pgrep gap; papercut filed); trainer HTTP on
  node1; relay-scp NUL corruption (md5 everything); no window-1 captures; steady = wins 2–3.
- The attention kill-shot probe (case 5, ULP source localization) is queued, non-blocking.
- Jack's last explicit instructions: work autonomously, overlap comm/compute, constraints
  above; he was told W1's honest result and the numerics ruling in the last status.
