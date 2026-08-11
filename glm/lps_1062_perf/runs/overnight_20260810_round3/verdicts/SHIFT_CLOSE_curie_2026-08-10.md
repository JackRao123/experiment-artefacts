# SHIFT-CLOSE SUMMARY — verification lane (curie), 2026-08-10
Scope: LPS-1062 round-3 night. All verdicts issued on my own extraction from
md5-proofed artifacts; all frames pre-registered/frozen before numbers.

## VERDICTS
1. **W3-v3: FAIL (capture 0.9% vs ≥50% bar) → W3-close ratified.**
   - Gate first (pre-registered, SM_SLACK_PREREG/RESULT): the B300 bwd phase
     HAS SM slack (occ_w 15-16%, 83% of kernel time <10% occ, NCCL ~60% of
     bwd time at 0% occ) — HEADROOM, W3-v3 design-live; CONFIRMED boltzmann's
     inherited numbers within ~1pp.
   - Canary (frozen frame + A1-A3): T1 capture FAIL (0.10s of 11.29s
     concurrent; v2 0.1% → v3 0.9%). T2 PASS (99.5% in-window). L1 PASS-by-
     re-derivation (75 chunks/mb = 74 kicks + 1 inline, exact; the frozen
     78/mb constant was a DSA-layer conflation). L2/L3/L4 PASS. M1 PASS
     (+24.9 ≤ +28 GiB; headroom 20.0 ≥ 10). M2 PASS (trims 0). N1 PASS
     (near-bitwise: w0 +0.1e-3, mains ≤0.7e-3).
   - T6 wait-by-cause (the v4-or-close discriminator): kicks release on a
     COMM-TAIL-DELAYED (b) edge into drain gaps, then FRAGMENT ~5×/chunk on
     shared comm/expert FIFO slots; main queues empty at release (host
     drain-throttled) → 0.9% capture. v3 removed the whole-backlog chain;
     the residual couplings are structural. v4 levers delivered (dedicated
     kick comm stream / earlier last_bwd_end or pool versioning / A-v3
     run-ahead prerequisite / side-stream priority).
   - T5: dilation ~nil at 0.9% capture (1.0175, no bucket concentration);
     d at real capture is a v4 measurement.
   - Memory decomposition: in-flight depth bound of 2 CONFIRMED (peak live
     23.14 GiB = two ~11.5 GiB chunk graphs; MoE-pipeline saves dominate).
2. **A-v3: CLOSED — mechanism PASS · numerics PASS · composition PASS ·
   wall informational-positive (+4-5% steady 131k).**
   - Path: under-armed boots INVALIDATED (grothendieck's catch) → FIX C′
     hybrid-gate finding (all three "C′-ON" boots FORCE-only; candidate 1.5
     confirmed; then fermi's code read: CACHE-off ⇒ FORCE-inert ⇒ free-
     routing status-quo regime) → two-boot completion plan.
   - Leg (b) (first genuinely C′-engaged A-v3 boot): checker 22/23 with the
     era-calibration rows landing EXACTLY (replay 0 / allgather 300 / dtoh
     1914 — retroactive proof of the hybrid table); A-v3 rows exact
     (eventsync_a==312, drains gone); canary in-band (my own deltas vs the
     era reference); wall +4-5% steady (honest steady-vs-steady read).
   - Leg (a): composition proof PASS — V3 invariant + cache invariant held
     simultaneously on every shared window 1-22 (my per-window extraction).
   - V3 invariant replicated across THREE regimes (under-armed B/F,
     FORCE-only, C′-engaged) — strongest robustness form available.
   - memcpy finding (CORRECTED mechanism): the probe's first-pass D2H reads
     sit ON the compute stream and block the host behind the backlog
     (33/window, p50 9ms, max 132ms; GPU copies µs-scale). fermi's whole-
     backlog defect class CONFIRMED; locus corrected (direct compute-stream
     ordering, not side-stream wait_stream). Fix: pinned/async/deferred or
     device-side read. (My initial "big-copy" refutation was wrong and was
     corrected openly with fermi.)
3. **F2 lane.**
   - (c1): VACUOUS-as-screen (zero phantom-fired windows by construction);
     the −34.5→−67.8e-3 drift recorded as pure config-trajectory divergence.
   - (c2) FORMAL ADJUDICATION (re-opened gate): **PASS** on my own
     computation — 21 windows 1:1, max |dloss| 1.65e-3 (main1; box lane's
     "main0" slip corrected), all in-band; matched warmup-datums verified;
     A3 satisfied-in-intent (forced boot-warmup asymmetry documented);
     drift cover clean. Scope: fix-INERTNESS; fired-correctness rests on
     the zero-mask argument + (b) + fibonacci 1a/1b ship gate.
   - Pre-registered warmup0-excess prediction CONFIRMED (−0.95e-3, in-band)
     — positive evidence for the inertness premise.
   - W2 arm: FAIL-BY-HANG verified — PG-18 ALLTOALL deadlock, all 8 ranks,
     seq 619; NumelIn imbalanced 2..130448 vs uniform NumelOut 65536
     (routing-imbalance-triggered per-chunk deadlock; repro reframed:
     imbalanced datum mix). Regime-matched control (reference clean 701)
     makes it regime-independent.
4. **Adjudications issued under caveat**: box-3 anchor accept-with-caveat
   (cross-box warmup0 marginal; same-box re-base rule for verdict numerics).

## PROCESS FRAMEWORK BUILT (all ratified; now standing)
- Boot-region scoping (persistent-tee hygiene); A2.1 pre-drive arm-check;
  A2.1-E engagement markers + hybrid env-flag enumeration (C′ = CACHE+FORCE,
  /proc truth paste); A3 boot-history symmetry. Papercut pc_fa89e43ecbd0
  (log overwrite) → post-boot log copies now standing practice.
- WARMUP0-TRANSIENT-PATTERN characterized (excess only in cross-box /
  regime-shifted comparisons; three no-excess points; break condition pinned).

## LANE SELF-CORRECTIONS (recorded openly)
- Comm-coverage script bug caught and discarded (30% → corrected 3.9%).
- fermi refutation reversed (host-wait vs GPU-execution premise error).
- Box-lane main0/main1 attribution slip corrected.

## OPEN ITEMS FOR MORNING
- (d) perf investigation: the 5.7% ON-cost QUESTION (deflators: box-2
  anchor-less walls; possible F2_SHADOW instrumentation) — ship-gate item.
- fibonacci 1a/1b NaNx0 checksum canary = the F2 SHIP gate.
- W3-v4 decision rests with the estate (T6 anatomy + v4 levers delivered).
- A-v3 memcpy fix (fermi) — pinned/async/device-side read. Locus search
  BOUNDED (fermi's code read, bounded negative result: not in the patch
  surface; the copies are a V3-activated host read in the REPLAY's execution
  of EXISTING code — the replay-side indexer/attention forward path, which
  carries the repeat_interleave/masked_fill machinery).
- Morning capture recipe: (i) GPU-track annotations for the Lookahead
  functions (T3 was not measurable on tonight's artifact); (ii) python
  stacks (with_stack=True) targeted at the replay-side indexer/attention
  forward path — the host-read attribution needs line-level frames (tonight's
  traces are slimmed; the cpu_op chain bottoms out at
  CheckpointFunctionBackward).
