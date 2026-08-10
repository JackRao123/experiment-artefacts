# Orchestrator handoff: kepler → helmholtz (2026-08-10 morning)

You (helmholtz, Claude Fable) inherit the LPS-1062 round-3 GLM-5.2 perf push from kepler.
Jack sleeps; his standing orders are BINDING — read `NIGHT_ORDERS_JACK.md` (same dir) FIRST.
Goal metric: **TPS/GPU**. Constraints: 131k fits with ≥10 GiB/GPU headroom; loss = Jack's
variance-class standard (RATIFIED: optimized-vs-default diff ≈ run-to-run variance of
default; house band ≤2e-3 pass / >5e-3 stop, matched warmup-datums; bitwise = diagnostic
gold tier only). Two-tier verdicts RATIFIED (mechanism decides; wall on these boxes
informational; ship magnitudes deferred to prod fabric).

## Fleet succession (message via ~/.agents/scripts/send-message.sh <name> "...")
- **helmholtz** (you) ← kepler: orchestrator. Successors for the roles: **grothendieck,
  fermi, curie** (Kimi K3) ← fourier (box), minkowski (design), boltzmann (verification) —
  kepler assigns roles at handoff; predecessors brief successors directly and confirm to YOU.
- Memory: `lps-1062-round3-orchestration` + `lps-1062-jack-night-orders` in project memory.

## Hardware (only these two are touchable — Jack's verbatim order: don't kill other
## people's devboxes)
- **Box 1 (timing lane): 318g61w** (2×8 B300 ali; ssh tj-318g61w / -1; anchors: 705 @131k-d4,
  703.5 @16k-d32; C′-arm refs: 666/49.2s @131k, 715 @16k). Trainer HTTP on node1:8001.
- **Box 2 (correctness lane): wxlgv5w** (2×8 B300 ali; ssh tj-wxlgv5w; trainers @0e0b65a6,
  devbox-up scaffolding). NO anchors yet — anchor re-baseline required before any timed arm.
- **SHARED CPFS between boxes** (same project): per-box output subdirs mandatory
  (lps1062_bench/wxlgv5w/), separate mcore clones for divergent trees, NEVER checkout under
  a running job, never overlap a box's boot/model-fetch with the other box's steady timed
  window.

## Where the ladder stands (tonight's verdicts)
1. ✅ W1 shipped-as-v1 (mechanism PASS; ~0 wall on tail-bound boxes; model −3.4…−5.1 s prod).
2. ✅ C′ timed arm PASS on mechanism (replay eventsync 300→0, allgather 600→300; the two
   drain-seconds FAILs adjudicated as wait-REDISTRIBUTION with a conservation proof; profile
   re-baselined). A-v3 unlocked with tempered expectations (~0 wall publishable).
3. ✅ W2 v3 GATE-CLEAN: the 0626 T2 FAIL was TWO HARNESS artifacts (fixture bool-mask
   collapse + global-RNG grad_out; three more harness bugs found in review chain). Re-run
   on gate d88d8b7d (canonical): 44/0/0, R1 did NOT fire (bitwise at gate scale, hidden
   2048). W2 timed arm queued (T3 canary slot, house band + 20-step drift cover). 6144
   fidelity re-run approved for box-2 idle window.
4. ✅❌ **W3 canary VERDICT (boltzmann, frozen frame): MECHANISM PROVEN, WIN REFUTED as
   implemented.** All log bars green (kicks==hits==78/mb, misses 1/mb, sweeps 0), canary in
   band, FIX C composition intact. But the kicked recompute runs **99.9% serialized** —
   lands in the right windows (99.4% in bwd), overlaps ~nothing (0.01s/10.82s concurrent).
   Two measured causes: (a) side_stream.wait_stream(current) orders each kick behind the
   ENTIRE compute backlog; (b) B300 kernels run full-width serial (baseline concurrency
   1.017 — no SM slack). −8…−12 s REMOVED from the stacked model. **W3 DISARMED for all
   subsequent arms; 16k HARD OFF.** MEMORY DECOMPOSED (snapshot, minkowski): +25.8 GiB =
   ~11.5 intrinsic (depth-2 × ~11.4 GiB/chunk graph — MoE-pipeline saves dominate; the
   attention-K/V hypothesis was REFUTED) + ~14 GiB recoverable allocator retention (knob:
   empty_cache/pool cap). **GATE ANSWERED — POSITIVE (helmholtz shift, 08-10 ~01:15 PT):**
   boltzmann's SM-slack measurement: **35.9 s duration-weighted bwd-phase SM slack/step**
   (83.5% of kernel time <10% occupancy; SendRecv windows 0.0% occ). The 1.017 baseline
   concurrency was wait_stream DEPENDENCY serialization, not hardware saturation ⇒
   W3-on-B300 is NOT closed. **W3-v3 (input-dependency-only kick ordering) is LIVE in the
   design lane** (fermi, priority 1); open caveat: HBM-bandwidth contention term; 16k stays
   HARD OFF pending separate justification. curie running independent confirmation under a
   pre-registered frame (~/perf_profiles/lps-1062/round3/SM_SLACK_PREREG_318g61w.md) —
   CONFLICT or MARGINAL escalates to orchestrator. Option-6 disposition RESOLVED pre-standdown
   by minkowski: UN-PARKED per stub rule 3 (comm-resequencing, not SM-slack-bound), but
   re-queued to design priority 2 behind W3-v3 (subsumption).
5. ⏭ F2 DP2 validation boots on box 2 (after box-1 capture clears the timed window).
   Jack's explicit pivot: F2 unlocks 4+ node scaling. Patches: f2.patch @0e0b65a6 +
   f2_main_rebase (branches pushed).
6. ⏭ W2 timed arm (box 2 after anchors, or box 1 after W3) · A-v3 timed arm (box 1).
7. ⏭ When experiments exhaust: NON-SLOP PRs for every proven lever (B/F, NCCL env
   defaulting, C′, W1, W3/W2/F2 as passed) — prepared NOT merged, evidence-linked; ALL to
   Jack for morning review, plus a scorecard (lever → TPS/GPU worth, memory cost, status).

## Durable storage (Jack-mandated)
- Estate: github.com/JackRao123/experiment-artefacts @ 2263b82+ (>100MB traces gitignored
  w/ md5 README — Jack ruled: keep local-only).
- Code: basetenlabs/Megatron-LM `jackrao/lps-1062-r3-stack` (7 lever commits, tip 15d5679e,
  carries W3 v2) + `jackrao/lps-1062-r3-macship-snapshot`; basetenlabs/trainers
  `jackrao/lps-1062-r3-f2` + `jackrao/lps-1062-f2-main-rebase`. trainers pre-push hook
  lints untracked dirs — use --no-verify for archival branches (papercut filed).

## Standing rulings (all ratified or operative — do not re-litigate)
- Numerics variance-class + two-tier (Jack-ratified). Verify-staging amendment (soak covers
  verify-on bar; timed arms verify-off). INVALID≠FAIL (mismatched warmup-datums → re-run).
- Canonical artifacts: T2 gate d88d8b7d; W3 patch v2 6f08c5dc; instrumentation patch
  31f8e0bc (round-trip-proven; 191c41ae quarantined); checker befb52b4+ (md5-proof box
  copies before on-box verdicts).
- Review discipline: pre-registered frames frozen before evidence lands; derive-don't-trust;
  an assertion that never RAN is indistinguishable from one that passed (three harness bugs
  tonight); patch files are canonical — no box hand-edits.
- Combined-ship precondition: DP2+phantoms+W2 canary before any combined F2×W2 ship.

## Key docs (this dir unless noted)
NIGHT_ORDERS_JACK.md · TRACE_ACCEPTANCE.md · LEVERBOARD.md · DESIGN_helmholtz.md ·
ESTATE_NOTES_minkowski.md · HANDOFF_VERIFY_hilbert.md (+S7 addendum) · HANDOFF_BOX_bohr.md ·
W2V2_DECISION_stub.md · dispatcher_opt/fixc/DESIGN_FIXC_PRIME.md · f2_fix/ ·
dispatcher_opt/fixa_v3/ · traces ~/perf_profiles/lps-1062/round3/.

## helmholtz shift log (live)
- 01:12 PT: Succession 2 COMPLETE — grothendieck/fermi/curie all briefed + acked;
  kepler/fourier/minkowski/boltzmann stood down. Box 1 verified CLEAR (W3 teardown done).
- 01:15 PT: SM-slack gate POSITIVE (see item 4) → design lane re-sequenced: W3-v3 first,
  option-6 second. Scorecard skeleton drafted (SCORECARD_MORNING.md, this dir).
- 01:20 PT: F2 boot verified by grothendieck: fix ON (recipe a), expB-ep16cp8dp2.json,
  warmup deadlock-trigger armed; DECISIVE warmup pass-1 in progress (clean ⇒ READY,
  pre-fix bug ⇒ >35 min hang). A-v3 staging on box 1 in parallel (no boot until box 2
  past fetch, interleave rule).
- 01:35 PT: SM-slack DOUBLE-CONFIRMED — curie's pre-registered independent derivation
  agrees with boltzmann on all quantities (occ_w ~15%, f_low50 ~84%, conc 1.026 both
  ranks; slack 37.3 s vs 35.9 s = window-def delta). Gate CLOSED as HEADROOM. Caveat
  carried to canary frame: occupancy = launch-slot availability, not HBM contention.
- 01:35 PT: F2 verdict set RATIFIED as (a)+(a3)+(b)+(c) — (a3) kill-switch control
  (fix OFF, hang must reproduce w/ py-spy signature), (b) custmix 20-datum customer
  repro, (c) canary vs DP1 golden. (d) perf + fibonacci hardening (1a/1b NaNx0 canary,
  exactly-once instr) deferred to post-verdict/ship-sign-off. grothendieck runs the set
  without per-gate check-ins; fixa_v3 goes on a DEDICATED box-1 mcore clone (ruling:
  never mutate the shared tree while box-2 jobs run).
- 01:40 PT: BOX 3 provisioning LAUNCHED (devbox-up 2 b300 ali, Jack-preauthorized) —
  parallel arm lane, intended for the W2 timed arm after its own anchor re-baseline.
  **Box 3 = wxlg05w (zero) — ⚠️ ONE CHARACTER from box 2 wxlgv5w (vee). Teardown
  whitelist is now exactly {318g61w, wxlgv5w, wxlg05w}. Triple-check every
  ssh/scancel/stop target.**
- 01:50 PT: **F2 GATE (a) PASS** — DP2 boots clean with phantoms (warmup pass-1 cleared,
  READY 139 s, /health 200, dispatch→READY 13–14 min normal band; patched imports
  /proc-verified). (a3) kill-switch control starting (fix OFF, hang MUST reproduce;
  no-reproduce = anomaly, escalate). A-v3 boot cleared for box 1 (dedicated clone).
- 01:55 PT: A-v3 soak boot LIVE on box 1 via dedicated clone mcore_318g61w_av3
  (byte-identical-source proof + both-ways import probe; fixa_v3 md5 2f50c182).
  Canonical (a3) timeline: fourier's 08:26 UTC scancel = step 1; grothendieck holds stick.
- 02:05 PT: **fermi delivered W3-v3 spec+patch** (DESIGN_W3V3.md;
  w3-lookahead-recompute-v3.patch md5 05dda37f; two-event kick ordering, depth≤2 by
  construction, E1–E8 edge audit, HBM as measured dilation). ACCEPTED. Canary = box 1
  after A-v3 timed arm; stacking rule conditional on A-v3 verdict. curie's canary frame
  FROZEN pre-boot (W3V3_CANARY_ACCEPTANCE_FRAME.md; conditional baseline in-frame,
  capture ≥50% bar, mem ≤+28 GiB re-measured). fermi now on PR-phase scaffolding
  (proven levers only: B/F, NCCL env, C′, W1; W3-v2 excluded from ship branches).
- 02:10 PT: **BOX 3 (wxlg05w) DEVBOX READY** (13/13, exit 0) — handed to grothendieck:
  golden stack → CPFS lps1062_bench/wxlg05w/ → anchors → W2 timed arm lives there.

## Watch items
- Succession trigger: 50–60% context per agent (Jack revised down) or natural boundary.
- Box waiter false-positives during boots (pgrep gap); md5 every relay-scp; no window-1
  captures; steady = windows 2–3; wall regressions on these boxes are NOT verdicts.
- The attention kill-shot probe (ULP source) remains queued, non-blocking.
