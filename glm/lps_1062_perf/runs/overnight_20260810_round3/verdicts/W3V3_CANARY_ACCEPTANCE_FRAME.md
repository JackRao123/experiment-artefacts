# PRE-REGISTRATION — W3-v3 CANARY ACCEPTANCE FRAME (FROZEN pre-boot)
Author: curie (verification lane) · Date: 2026-08-10 · Status: FROZEN — any change
requires a numbered amendment ACK'd by helmholtz BEFORE the canary boot.
Spec: DESIGN_W3V3.md sec 7 — STAGED + RECONCILED (md5 eee4425fc9c0b2a81a9ee7ae44a41bc3,
verified locally 2026-08-10; R0 condition discharged).
Patch under test: w3-lookahead-recompute-v3.patch
  md5 05dda37f68f3113747a812e357d9b552 (verified locally 2026-08-10; matches the
  pinned chain; grothendieck stages on-box with md5 chain, curie re-verifies).

## AMENDMENT #1 (2026-08-10, pre-boot; fermi spec-author review + spec-text
## reconciliation; helmholtz ACK requested)
- A1.1 T1 aligned to spec text: OC = side-stream kernel time concurrent with
  ANY other GPU work (not only main-stream kernels), per boltzmann_w3_side_stream.py.
  Bar unchanged: OC >= 50%. Comm-window coverage stays CONTEXT-only (fermi:
  do not promote — kicks may legitimately land in low-occ COMPUTE windows).
- A1.2 T2: added REPORTED context (no bar): kick start-delay distribution
  ts_side_first - window_start, p50/p90 (bar proves un-chaining; delay says how
  much of each window the kick gets).
- A1.3 T3 REDEFINED per fermi: the <=10 ms bar moves from arrival-time to the
  main-stream IDLE GAP at the window head:
    stall = ts(first main-stream kernel in window)
            - max(window start, end of last main-stream kernel before window).
  Rationale: arrival-time conflates queue drain (previous tail still executing
  = not a stall) with true done-event wait exposure. Arrival-time kept as
  REPORTED context (queue-depth diagnostic); T6 attributes idle-gap vs
  host-bound vs comm. p95 still reported.
- A1.4 L2 tightened to spec text: stash_misses == 1/mb (4/step; the last chunk
  recomputes inline). Any other count (0 or >=2) = structural anomaly = FAIL.
  (Supersedes the frozen <=1/mb; spec sec 7 governs. Flagged to helmholtz.)
- A1.5 T6 cause taxonomy pinned to spec: SendRecv-in-bwd-window accounting by
  CAUSE = drains / eventSyncs / exposed — never totals.
- A1.6 N1 pinned to spec numbers: house band <=2e-3 PASS / >5e-3 STOP;
  matched --warmup-datums MANDATORY (mismatch = INVALID, re-run); 20-step
  drift cover. Middle zone handled per standing house ruling.
- A1.7 V: two-tier verdict made explicit — mechanism bars (L/T/M/N) FIRST,
  wall SECOND vs the SELECTED baseline (conditional per helmholtz pin; the
  fallback branch's reference is the box-1 C'-on anchor, 666 s class).
  NEVER quote uncalibrated wall.
- A1.8 INVALID conditions: added CDMC env check — arm boot must show CDMC
  UNSET in /proc/<pid>/environ (spec R6: a CDMC=1 boot re-serializes on a
  single hardware channel; out of scope => INVALID, not FAIL).
- A1.9 Pre-boot precondition added: in-process CPU suite 40/40 green incl. the
  dropout RNG-isolation proof (spec sec 9 evidence) + patch md5 re-verified
  on-box. Failure = INVALID, halt before boot.

## AMENDMENT #2 (2026-08-10, pre-boot; helmholtz request, curie wording;
## helmholtz ACK requested)
- A2.1 PRE-DRIVE ARM-CHECK — new INVALID precondition, ALL arms. Before any
  arm drives (canary / timed / soak), the arm's drive plan DECLARES the
  expected gate stack with marker strings per gate (for the W3-v3 canary under
  the ELSE branch: the C′ gates + W1 + the W3-v3 lookahead markers). After
  boot and BEFORE the run drives, the boot region of the trainer log must
  contain every expected gate's ARMED/ACTIVE marker line (boot-region-scoped
  per the persistent-tee hygiene rule); the marker lines are pasted into the
  arm's evidence bundle. Any expected gate NOT proven armed => the arm is
  INVALID (not FAIL): halt, fix the boot, re-drive. Applies to the W3-v3
  canary, the W2 arm, and all future arms. (Lesson of 2026-08-10: both A-v3
  boots ran under-armed — C′+W1 gates off — and produced regime-mismatched
  rows; grothendieck's boot-region-scoped catch invalidated both arms.)
- A2.2 BRANCH SELECTION RECORD (section B conditional): the A-v3 timed arm is
  INVALIDATED 2026-08-10 (under-armed boots; regime mismatch, not defect —
  A-v3-specific rows PASSED; the 7 FAILs were post-C′-regime pins inapplicable
  to a C′-off trace). A-v3 is therefore inconclusive tonight; per the frozen
  conditional ("IF A-v3 PASSES ... ELSE ..."), the ELSE-BRANCH FIRES: the
  W3-v3 canary boots with fixa_v3 REVERTED from the stack, baseline = C′-ON
  refs. INVALID != FAIL: this is the conservative default arm, not a judgment
  against fixa_v3; a future canary may run in-stack if the A-v3 re-arm passes.

## AMENDMENT #3 (fleet label A3) (2026-08-10, pre-boot; helmholtz request,
## curie wording; ratified with the F2 frame set)
- BOOT-HISTORY SYMMETRY — INVALID precondition (full text: F2_C_ADJUDICATION
  _FREEZE.md Amendment #2 / A3): the canary boot must be FRESH with op-history
  identical to the selected baseline's reference windows (optim_step count,
  warmup executed-or-skipped symmetrically). Boot-region startup lines pasted
  pre-drive alongside the A2.1 arm-check markers. A reused boot with prior
  optim_steps, or any op-history asymmetry vs the baseline => INVALID (not
  FAIL): halt, re-boot fresh, re-drive.

## AMENDMENT #4 (fleet label A2.1-ENGAGEMENT) (2026-08-10; helmholtz ratified
## in principle, curie wording)
- Extends A2.1. For any gate with an ARMED-vs-ENGAGED distinction, the drive
  plan must declare ENGAGEMENT markers (not only armed/active markers), and
  the pre-drive check covers BOTH. Engagement markers are per-gate and must
  prove the mechanism actually fired — e.g. FIX C: the one-time 'first replay
  cache hit' line AND the per-window counters (hits==300/step, misses==0,
  shape_mismatches==0). A boot that is armed but lacks the engagement marker
  => INVALID for regime-dependent experiments (not FAIL): halt, fix, re-drive.
  ENV-FLAG ENUMERATION (ratified 2026-08-10, helmholtz): for HYBRID gates the
  drive plan must enumerate the gate's FULL env-flag set (C′ =
  BT_MOE_DISPATCH_REPLAY_CACHE=1 + BT_MOE_ROUTING_REPLAY_FORCE=1 — both
  listed, both checked), with /proc/<pid>/environ as the launch-env truth
  paste. Tonight's failure mode: the arm-check pasted the FORCE marker and
  never looked for the CACHE marker; all three 'C′-ON' boots ran FORCE-only.
  Lesson of 2026-08-10: a C′-ON boot passed the 7-gate arm-check while FIX C′
  never engaged (replay eventSyncs 300 vs era 0); the estate's own tee shows
  the precedent (boot5: ACTIVE without first-hit).

## R0 — Provenance / reconciliation (DISCHARGED)
Frame was registered off fermi's sec-7 summary + helmholtz's pin list. The
staged spec text has now been read in full and reconciled: all frame bars match
sec 7, with the alignments above folded into AMENDMENT #1. One governed
deviation from spec text: sec 7's "vs the C'-on baseline" for wall/memory
deltas is superseded by helmholtz's LATER conditional-baseline pin (section B)
— helmholtz accepted the spec AND pinned the conditional; the pin governs.

## B — BASELINE (conditional, written in — not a post-hoc choice)
- IF the A-v3 timed arm PASSES: baseline = A-v3-ON timed windows; the canary
  runs with fixa_v3 IN STACK.
- ELSE: baseline = C′-ON references with fixa_v3 REVERTED.
- The A-v3 verdict determines the branch. The canary operator does not choose.
- All "vs baseline" deltas below read against the selected branch.

## L — LOG bars (mechanism; window rolls on 300 TRUE CHUNK BACKWARDS —
##     v2's events-window mislabel is fixed; counter semantics identical)
- L1  kicks == stash_hits == 78/mb (312/step at d4).
- L2  stash_misses == 1/mb (4/step; the last chunk recomputes inline).
      Any other count (0 or >=2) = structural anomaly = FAIL. [AMENDED A1.4]
- L3  sweeps == 0 (halt) AND fallbacks == 0.
- L4  kick_ms_avg <= 60 ms (spec "~60 ms" dilation bar; reference inline ~41 ms);
      kick_ms_max REPORTED (no bar).
      SCOPE RULING (curie, 2026-08-10, helmholtz-delegated; interpretation, NOT
      re-thresholding — the 60 ms threshold is unchanged): the bar governs
      STEADY-STATE windows. Window 0 of a boot (first kicks ever: side-pool
      first-touch allocation, gather warmup, lazy module/context init) is a
      startup transient — the same class the house process rule excludes
      ("never profile the first step after boot"). A single non-sustained
      outlier kick (e.g. one 2946 ms kick) is likewise outside the bar's
      intent: the spec bars "SUSTAINED max >> that" as the contention signal.
      APPLICATION: L4 PASSES iff every steady-state window's kick_ms_avg
      <= 60 ms. Window-0 composition and any single-kick outlier are REPORTED
      with an explanation attempt; if a >60 ms avg appears in any steady
      window, or outlier-magnitude kicks recur across windows (sustained),
      L4 FAILS. Required data: per-window kick_ms_avg/max trajectory.

## T — TRACE bars (arm trace; checker befb52b4+, trace_processor >= v56.1)
- T1  OVERLAP CAPTURE (the v2 failure axis; v2 measured 0.1% = 0.01s/10.82s):
      OC = (side-stream kernel-time concurrent with ANY other GPU work within
      bwd windows) / (total side-stream kernel-time within bwd windows),
      per boltzmann_w3_side_stream.py.  BAR: OC >= 50%.  [AMENDED A1.1]
      Context (no bar): comm-window coverage = fraction of bwd NCCL-kernel wall
      time with a side-stream kernel co-resident (>=50% expected if the design
      lands; SM-slack result: ~60% of bwd kernel-time is 0%-occ NCCL — the
      territory exists). CONTEXT-ONLY per fermi: kicks may legitimately land
      in low-occ COMPUTE windows; a hard coverage bar can false-fail a
      working arm.
- T2  TWO-EVENT ORDERING: kicks NOT chained behind the compute tail.
      Measurable proxy: >= 90% of kicked LookaheadCheckpointFunctionBackward
      windows show the first side-stream kernel starting STRICTLY INSIDE the
      main-stream window (ts_side_first < main-window end). wait_stream-style
      chaining (side start >= main drain) fails this bar.
      REPORTED context (no bar) [AMENDED A1.2]: kick start-delay distribution
      ts_side_first - window_start, p50/p90 — the bar proves un-chaining; the
      delay says how much of each window the kick actually gets.
- T3  CONSUME-STALL at each LookaheadCheckpointFunctionBackward head
      [AMENDED A1.3 — fermi's idle-gap form]:
      stall = ts(first main-stream kernel in window)
              - max(window start, end of last main-stream kernel before window).
      BAR: mean <= 10 ms; p95 REPORTED.
      Rationale: main stream rolling straight from the previous tail into this
      window means the consume wait cost nothing even when arrival is late;
      the idle-gap form isolates true done-event wait exposure.
      REPORTED context (no bar): the arrival-time form
      ts(first main-stream kernel) - window start (queue-depth diagnostic).
- T4  (covered by L4 — kick_ms is a log/CUDA-event signal.)
- T5  HBM-BW TERM (INFORMATIONAL, no pass number): per-occupancy-bucket bwd
      COMPUTE-kernel dilation = curie's sm_slack analyzer re-run on the ARM
      trace vs anchor-318g61w, kernels matched by name, bucketed by anchor
      occupancy decile, dilation = arm_dur/anchor_dur per bucket.
      STOP CONDITION: if canary STEP WALL regresses beyond the house band vs
      the selected baseline, the arm is aborted and scored FAIL-safe regardless
      of other bars.
- T6  SendRecv-in-bwd-window accounting by CAUSE per the standing rule
      [AMENDED A1.5 — spec taxonomy]: drains / eventSyncs / exposed —
      NEVER totals.

## M — MEMORY bars (131k)
- M1  Peak reserved delta vs the SELECTED baseline <= +28 GiB, RE-MEASURED on
      the arm boot (v2's +25.8 GiB was taken under the serialized regime and
      does NOT carry over). 16-rank allocator snapshot piggybacked on the arm
      boot; snapshot verified loadable before any verdict.
      SCOPE RULING (curie, 2026-08-10): the RE-MEASURED clause binds the ARM
      side (the canary's own peak must be freshly measured on the canary boot —
      poller + piggybacked 16-rank snapshot). The BASELINE side is satisfied by
      the SELECTED baseline's existing measurement: the driver-polled C'-era
      max (203165 MiB, C'-ON, tonight, same box/stack/shape, method-matched
      poller). A fresh C'-only re-measure boot is NOT required: reserved-memory
      peaks are deterministic-allocator, shape-driven quantities with MiB-class
      cross-boot jitter, immaterial against the 3.1 GiB margin (+24.9 GiB
      measured vs +28 GiB bar). A3 considered and NOT BINDING here: the
      baseline's op-history does not move its steady-state reserved peak.
      CONDITIONS: (a) method/shape match confirmed (driver poller, 131k-d4,
      both sides); (b) arm-side poller peak and snapshot peak agree; (c) the
      snapshot decomposition (intrinsic vs retention) is REPORTED against the
      v2 structure (11.5 + ~14 GiB). If (a) fails, the 26-min C'-only
      re-measure becomes required.
- M2  trims == 0 (the ~14 GiB allocator-retention knob is OFF at 131k).

## N — NUMERICS
- N1  [AMENDED A1.6] Canary loss vs the selected baseline within the house
      band: <=2e-3 PASS / >5e-3 STOP (middle zone per standing house ruling);
      MATCHED --warmup-datums MANDATORY (mismatch = INVALID, re-run);
      20-step drift cover. 131k ONLY.
- N2  16k is HARD OFF: any 16k datum in the arm => INVALID arm, re-run.

## V — VERDICT RULE [AMENDED A1.7 — two-tier]
- TIER 1 (mechanism): ALL of L1-L3, L4(avg), T1, T2, T3, M1, M2, N1 pass,
  no STOP fired, no INVALID condition => MECHANISM PASS.
- TIER 2 (wall): quoted only after Tier 1 passes; vs the SELECTED baseline
  (section B conditional; fallback-branch reference = box-1 C'-on anchor,
  666 s class). NEVER quote uncalibrated wall.
- CANARY FAIL  <=> any bar FAIL, or STOP fired. Failing bar(s) reported with
                 numbers; no re-thresholding.
- INVALID      <=> md5 mismatch on the staged patch or any box copy; snapshot
                 unloadable; warmup-datums mismatch; 16k contamination; trace
                 fails checker; CDMC=1 on the arm boot [A1.8]; pre-boot CPU
                 suite not 40/40 [A1.9]. INVALID != FAIL: repair and re-run.
- T5/HBM and kick_ms_max and T3-p95 and T2-delay and comm-coverage and
  T3-arrival-time are reported under every outcome.

## C — Carried caveat (curie's, from the SM-slack gate)
Occupancy figures are launch-config SLOT-AVAILABILITY estimates (co-residency
feasibility), NOT measured HBM bandwidth. Therefore OCCUPANCY IS NOT THE PROOF:
T1 capture% is the proof, and T5 dilation is the measured-not-assumed HBM term.

## Logistics
- Box slot: box 1, after the A-v3 timed arm (helmholtz's sequencing).
- grothendieck stages the patch with md5 chain; curie re-verifies md5 on-box
  before boot (mismatch => INVALID, not FAIL).
- Analyzer: curie's sm_slack script (gate run) generalized for per-bucket
  matched-kernel dilation; boltzmann_w3_side_stream.py for T1.
