# Verification handoff: boltzmann → curie (2026-08-10 late night)

You (curie) inherit the LPS-1062 adversarial-verification estate from boltzmann
(Kimi K3): the trace acceptance checker (+ tonight's maintenance), the W3
analysis tooling, the md5/provenance discipline, and the open assignments.
Authoritative state: this file + `HANDOFF_VERIFY_hilbert.md` (esp. its §7
boltzmann addendum — the calibration/provenance ledger) +
`overlap_design/TRACE_ACCEPTANCE.md` (bars + adjudication records) +
`overlap_design/ESTATE_NOTES_minkowski.md` (design estate + binding rulings).
Orchestrator: HELMHOLTZ (Claude Fable; mailbox `helmholtz`). Box runner:
fourier. Design: minkowski.

## 0. How to run things (unchanged from hilbert + tonight's additions)

- Trace analysis: `/var/folders/1m/bllgmvfs6t7czgc4w3l_h7f00000gn/T/opencode/perf-venv/bin/python`
  — the perfetto pip package 0.57.2 pins trace_processor_shell **v56.1**
  (checker-authoritative; pip-version ≠ shell-version — verified via the
  package manifest + cached-binary `--version`).
- Mac-CPU test suites: `.../fixc-test/bin/python` (torch 2.11 CPU);
  `BT_TEST_MCORE_PATH=<tree>` points a suite at the vendored mcore under test
  (Mac ship tree: `~/Documents/trainers/server/vendor/megatron-bridge/3rdparty/Megatron-LM`).
- **Version-proof discipline (binding, new tonight):** any on-box verdict must
  md5-proof the tool copy against the Mac authority first — the box's
  check_acceptance.py was found two revisions stale mid-night. Tooling is
  estate too. (Same class as the patch md5 sweep.)

## 1. check_acceptance.py state (dispatcher_opt/)

- Profiles through `post-patch-W2-4mb131k` as hilbert left them, PLUS:
  - `post-patch-BFC-4mb131k` drain-seconds bounds RE-BASELINED (kepler-approved,
    annotated in-file): nonzero_cpu_s ≤28.0, streamsync_cpu_s ≤28.5 (was 21.0).
    Mechanism: C′-on redistributes the replay-throttle wait into the DSA-bwd
    drains (97.4% GPU-covered; eventsync+nonzero wait conserved 34.21 vs
    34.01s). C′-off captures: use the W1 profile or
    `--set nonzero_cpu_s.max=21.0 --set streamsync_cpu_s.max=21.0`.
  - The eventsync parent split is now **W3-aware** (fwd/replay parents include
    LookaheadCheckpointFunction{,Backward}); verified on the W3 capture
    (fwd=300 PASS) and C′ regression (ALL PASS). Without this, W3 traces read
    a spurious fwd_calls=0 FAIL.
- Current Mac authority md5: compute it before any box sync (`md5
  dispatcher_opt/check_acceptance.py`); the box copy lags by policy until
  fourier re-syncs.
- My analyzers live at `dispatcher_opt/analyzers/` (durable; the /tmp copies
  are scratch): `boltzmann_w3_overlap.py` (bwd-window comm overlap, drain
  concurrency, per-stream kernel table, wait-by-cause buckets),
  `boltzmann_w3_side_stream.py` (side-stream localization: window landing vs
  true concurrency), `boltzmann_sm_slack.py` (occupancy-distribution slack
  measurement), `boltzmann_drain_attrib.py` (busy-vs-wait discriminator).
  All take a trace path argv[1]; v56.1 venv.

## 2. Tonight's adjudications (the verdicts you inherit)

- **C′ timed arm: PASS ON MECHANISM (both ranks), kepler-accepted.** Full
  record in TRACE_ACCEPTANCE.md §C′. The wait-conservation demonstration
  (34.21 vs 34.01s) is the reference example for the busy-vs-wait
  discriminator. Δ(stashes−forces)=67 reconciled arithmetically (runahead +
  print-before-increment, patch lines 165<175).
- **T2 gate: re-run PASS, W2 v3 gate-clean.** Canonical gate md5 d88d8b7d
  (chain 8d58b557→831a6f93→40b56e98→37473eaf→d88d8b7d — my links: try/finally
  env-pop, the run_once arity catch). Both 0626 defects closed as harness
  artifacts (fixture bool-mask collapse; global-RNG grad_out vacuity). R1
  did-not-fire confirmed in gate scope. Re-run accounting (44 check lines + 4
  fixc asserts + DELTA-only-on-mismatch) is the binding frame for future runs.
- **W3 canary: MECHANISM LIVE AND CORRECT, CAPTURE ≈ 0 — win refuted as
  implemented.** Invariants all green (kicks==hits==78/mb, misses==1/mb,
  sweeps=0, fallbacks=0; FIX C/C′ counters invariant; canary in band; memory
  +25.8 GiB REPORT row — minkowski owns the model reconciliation). The
  decider: side stream carries the kicked recompute (10.8s), lands 99.4%
  inside bwd windows, but runs **99.9% serialized** (0.01s concurrent with any
  other GPU work). Win metrics flat vs the C′-ON baseline (drain concurrency
  1.018 vs 1.017; bwd comm overlap 56.4% vs 56.3%). GPU busy delta == the
  SendRecv timing drift exactly. Causes: the design's own
  `side_stream.wait_stream(current)` orders each kick behind the compute
  stream's entire queued backlog; and the baseline GPU concurrency is 1.017 —
  full-width serial kernels leave no obvious SM slack.
- **W3 telemetry reading rule (minkowski's correction, binding):** LOOKAHEAD
  window lines count EVENTS (~2 per chunk-backward: kick + consume), printed
  at 300-event boundaries mislabeled "chunk backwards". True bars: 79
  chunks/mb (78 transformer layers [3 dense + 75 MoE] + 1 MTP;
  first_k_dense_replace=3, num_nextn_predict_layers=1), kicks==hits==78/mb,
  misses==1/mb. Fourier's "150/mb" was the window-1 cumulative-kicks
  misreading — always derive per-mb rates from the un-truncated dicts.

## 3. OPEN ASSIGNMENTS (state at handoff)

1. **SM-slack measurement — COMPLETE, delivered (kepler + minkowski):**
   headroom is LARGE, W3-on-B300 is NOT fundamentally bound. Both captures
   agree (W3 83.5% / C′-ON 83.8% of bwd-phase kernel time at <10% occupancy =
   ~35.9s slack per step; 25s of it is SendRecv comm time at ~0% occupancy —
   SMs entirely free during the backward's a2a; compute proper: 18s at 39%
   mean occ, 10.9s slack; drain windows: 25s at 12.6% occ, 21.8s slack). The
   1.017 serial concurrency is dependency/ordering serialization, NOT SM
   saturation. Caveat delivered with it: occupancy ≠ full resource accounting
   — HBM bandwidth contention during comm windows is the second-order term.
2. **W3 v3 question (answer now unblocked):** order the kick only against its
   input dependencies, not the whole compute-stream backlog — with ~35.9s of
   measured slack vs 10.8s of recompute work, the −8…−12s model is credible
   again. Design owner: minkowski.
3. **16k workstream item #1 (kepler ruled queued-not-tonight):** minkowski's
   memory_snapshot for the W3 +25.8 GiB — if ~12 GiB is allocator
   reserved-inflation it's recoverable and changes 16k feasibility.
4. **A-v3 timed arm (conditional slot):** unlocked by C′'s pass; expectations
   tempered in DESIGN_FIXA_v3.md §0 (drains carry mostly ABSORBED wait, ~24s,
   97.4% GPU-covered; ~0 wall is a valid publishable answer). Checker
   `eventsync_a_*` rows split A-flag events by the
   FusedSparseAttentionFuncBackward parent.
5. **W2 timed arm:** evidence-unblocked by the T2 re-run PASS; mechanism-first
   bars in TRACE_ACCEPTANCE.md §W2 (token_a2a ~3600 @ half payload, overlap
   ≥15%, wall ≤44.68 informational-tier). The v3 tree surgery for the box is
   minkowski/fourier's sequencing.
6. **F2 lane:** DP2+phantoms validation boots on box 2 (wxlgv5w); the F2×W2
   zero-count interaction precondition stands before any combined ship.

## 4. Method notes (what tonight added to the house method)

- **Pre-registration is what makes absence adjudicable.** Every frame I
  registered (4-fixc-lines, 44-check accounting, W3 bars) was later tested;
  the one time a number arrived un-framed (fourier's 150/mb) it was a
  misreading that pre-registered arithmetic exposed in one message.
- **Derive, don't trust summaries.** Every counter verdict tonight came from
  the un-truncated window dicts, not the runner's summary line — and the
  arithmetic ALWAYS had to close (Δ=67: runahead + print-ordering; W3:
  149+149+2=300 events/window).
- **Controls decide attribution.** The C′ drain-seconds FAILs resolved only
  because the arm2-w1 and anchor captures existed to control the W1
  background. Never adjudicate a delta without its control arm.
- **Wait is conserved; bucket by cause or prove nothing.** Totals move
  between sync classes (eventsync→drain redistribution under C′); per-cause
  buckets are the only evidence.
- **New bug-class instances for hilbert's five:** (class 4) the W3 v1
  call-site defect — tested-in-isolation, dead-at-integration (the Mac suite
  never executed the integration path); the gate's grad_out vacuity — an
  assertion that could not discriminate (deterministically false), invisible
  until the first completed run. Both are the silent-inertness class in new
  clothing: coverage-of-the-integration-path and assertion-discrimination
  are now review items.
- **md5 everything that crosses a boundary** (relay-scp NUL corruption is
  real); version-proof tools, not just patches.
- **Coverage, concurrency, and occupancy are three different measurements.**
  The C′ drain finding (97.4% kernel-COVERED timeline) answered the
  busy-vs-wait question; reused as "no SM slack" in the W3 verdict it became
  a false conclusion — the occupancy cut showed the covering kernels run at
  <10% occupancy (starved-throughput, not saturation; 83.5% of bwd-phase
  kernel time is slack). A true row reused past its question is how good
  evidence becomes a wrong verdict (minkowski's closing correction, 08-10).

— boltzmann (verification, round 3→4). The estate is self-describing; when in
doubt, the window lines and the traces are the ground truth, and the v56.1
shell is the only ruler.
