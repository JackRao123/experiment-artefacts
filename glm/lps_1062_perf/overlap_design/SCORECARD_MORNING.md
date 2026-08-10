# LPS-1062 round-3 morning scorecard — for Jack (helmholtz, orchestrator)

**Status: DRAFT — TBD slots fill as tonight's arms land. Delete this line when final.**

Goal metric: **TPS/GPU**. Frontier at nightfall: **715 tok/s/GPU @131k×d4 (46.8 s/step), 726 @16k×d32**
(golden EP16/CP16 2×8 B300 + ship NCCL env + TF32 head + DSA/RoPE host caches), peak ~202 GiB.

## Numerics disclosure (read first)

Your constraint "loss numerically the same" was adjudicated and you ratified the standard
(08-10, to kepler): optimized-vs-default diff ≈ run-to-run variance of default. Measured
stack fact: a **cross-boot bitwise floor of 0.7–1.7e-3 exists with zero patches applied**
(boot-time cuBLAS/cuDNN algo selection). All levers below hold *in-process* bitwise gates
where the design claims exactness, plus cross-boot canaries within the house band
(≤2e-3 pass / >5e-3 stop, matched warmup-datums). Nothing below perturbs loss beyond the
pre-existing boot floor.

Verdict tiers (ratified): **mechanism bars decide** whether a lever works as designed;
**wall deltas on 318g61w are informational** — the box is proven tail-bound (straggler
tails set the per-layer critical path), so slot-shaving levers under-deliver there by
construction. Ship magnitudes defer to prod-grade fabric.

Characterized tonight (WARMUP0-TRANSIENT pattern): cross-box or config-shifted comparisons
show a warmup0-only excess of +2.4…+2.8e-3 with mains always in-band; same-config
cross-boot warmup0 stays at the 0.7–1.7e-3 floor. Mains carry every verdict.

## Lever scorecard

| lever | what it does | TPS/GPU worth | memory cost | verdict / status | PR |
|---|---|---|---|---|---|
| B/F host-cache pair (`BT_DSA_CP_LAYOUT_CACHE`, `BT_THD_ROPE_HOST_CACHE`) | kills dispatcher-side host syncs | +11–13% @131k, +18.5% @16k customer shape (parity-proven, prior shift) | ~0 | SHIPPED into golden config | TBD |
| NCCL env defaulting (+ TF32 LM head) | ship config env block as code defaults | part of the +51% 256k ship config (prior shift) | ~0 | proven, needs defaulting PR | TBD |
| W1 probs-a2a second communicator (`BT_MOE_PROBS_A2A_COMM=1`) | de-serializes 900×5.13 s of latency-bound probs a2a off the token stream | mechanism PASS (900/900 off-stream, gap 11.2→9.2 ms); wall −0.4–0.6 s here; **model −3.4…−5.1 s (+7–11%) on prod fabric** | +1.3 GiB (2nd NCCL comm buffers) | SHIPPED-as-v1; residual understood (late probs-grad producer) and bounded. **PR-review finding (fixed on branch): the 2nd-communicator `new_group` call violated the torch contract on >1-EP-group topologies (would hang at 4-node EP16); all 2-node validation unaffected (single EP group); fix = use_local_synchronization + arm-time guard — flag W1 must stay OFF on multi-EP-group jobs until the fixed build is validated there** | TBD |
| C′ dispatcher replay-metadata cache (`BT_MOE_DISPATCH_REPLAY_CACHE=1`, hybrid routing-map form) | replay-side eventsync 300→0/step, dispatcher allgather 600→300 | mechanism PASS both ranks; wall ~flat here (value compounds as comm gets hidden) | ~0 (2 MB routing map ×in-flight) | PASS (verify-soak 20/20 + timed arm; drain-seconds FAILs adjudicated wait-redistribution w/ conservation proof) | TBD |
| W2 chunked MoE a2a pipeline K=2 (`BT_MOE_A2A_PIPELINE=2`) | overlaps token a2a with expert compute via per-peer views | model −3.0…−4.4 s standalone; **timed arm result: TBD** | ≈0 (views) | gate-clean at small scale (44/0/0 on canonical gate d88d8b7d), but the **TIMED ARM FAILED BY HANG**: first backward at full 131k with the C′+W1 composition deadlocked (NCCL collective timeout ~7 min, work seq 619). Reference arm clean (701 tok/s/GPU), golden mesh confirmed on both arms (A/B valid). Diagnosis so far (W2_ARM_HANG_ANALYSIS): W1's comm exonerated (16-rank, worked all forward); the stuck group is an **8-rank ranks-0-7 group that shouldn't exist at golden CP16 per the design** — either framework-created (hang surfaced on it) or lazily created by the armed path; front-runner mechanism = FIX-C replay-restore divergence across ranks ⇒ mismatched a2a splits. Morning steps written: reference-boot comm grep, group-identity code read, W2+C′ VERIFY=1 full-shape soak. Lever STOPPED, PR held closed | held closed |
| W3 lookahead recompute | overlap bwd(L) with recompute(L−1) | v2 REFUTED as implemented (kicks 99.9% serialized — wait_stream ordered each kick behind the whole compute backlog). **But the SM-slack gate came back POSITIVE: 35.9 s of bwd SM slack/step, comm windows at 0% occupancy — the serialization was v2's ordering design, not hardware.** −8…−12 s model credible again for a v3 (input-dependency kick ordering); HBM-contention caveat open | +25.8 GiB measured at 131k (~11.5 intrinsic + ~14 recoverable via allocator knob) | v2 DISARMED; **v3 canaried tonight and REFUTED on the verdict row too — capture 0.9% vs ≥50% bar** (plumbing fully correct: kicks land in the right windows 99.5%, numerics near-bitwise, memory +24.9 GiB ≤ +28 bar; the win mechanism didn't materialize; slack confirmed still present, 34.65 s). (Recorded confound, verdict-insensitive: v2's 0.1% was measured cache-ON, v3's 0.9%
cache-OFF — the v2→v3 delta crosses cache states, but both sit ~50× below the bar.)
**Wait-by-cause analysis then MAPPED the residual serialization** (W3V3_T6_WAIT_ANALYSIS.md): (i) the allocator-safety edge releases only after each chunk's comm tail drains (75% of gate-releases are token-A2A ends); (ii) kicked work fragments ~5×/chunk and re-gates on the SHARED comm/expert streams; (iii) main queues are EMPTY at release because host run-ahead is drain-throttled with FIX A reverted. **v4 path is mapped, not dead**: dedicated kick-A2A stream, allocator-edge surgery, side-stream priority — and critically, **A-v3 is a prerequisite** (restores host run-ahead so releases meet queued work). Option-6 (~1–1.5 s bwd-chain reorder) revived and building Mac-side | no PR; v4 design doc for morning |
| A-v3 (`torch.nonzero` batching, retry post-C′) | removes DSA-bwd host sync, unthrottled now that C′ landed | tonight's timed arm INVALIDATED (ran under-armed — C′/W1 off — mismatched gates; caught by our own boot-region audit). A-v3's isolated mechanism rows PASSED (312/312 eventsyncs, 1.9 ms CPU); the post-C′-regime composition measurement is the open item | ~0 | **MECHANISM PASS + NUMERICS PASS on the final clean-regime boot** (engagement proven: cache hits 1200/1200, misses 0; era-calibrated checker rows pass exactly; canary in-band). Informational wall: **+4–5% steady @131k** vs C′ (the post-C′ "~0 exposure" hypothesis was wrong in the good direction — removing the 312 DSA-bwd drains shows a real steady win on this box); 16k ramp-limited, −0.9% once ramped. **Also the L3 prerequisite for any W3-v4 capture** (host run-ahead). Known improvement: V3's probe-input D2H copies (33/window, removable). Composition soak (leg a) closed the story overnight | skeleton ready to open on the leg-(b) verdict |
| F2 phantom partitions (DP>1 deadlock fix) | correctness for any DP>1 customer + **unlocks 4+ node scaling** (EP16/CP8/DP4 ≈ 2× aggregate) | not a 2-node lever; scale-out enabler | no change @DP=1 | **VERDICT SET COMPLETE — PASS**: (a) fix-ON boots DP2 clean (READY 139 s vs pre-fix >35 min hang); (a3) kill-switch control reproduces the exact deadlock, py-spy signatures matched ⇒ causality; (b) exact Aug-9 customer deadlock shape completes, losses sane; (c2) numerically INERT at equal counts (max \|Δloss\| 1.65e-3 / 21 windows, symmetric-history flag-ON/OFF A/B; the planned DP1-golden bar was found INVALID-by-design and replaced pre-data). **Scope: phantom-FIRED numerics are design-argued (zero-mask) + (b)-sane; the 1a/1b NaN×0 checksum canary is the SHIP gate — not claimed proven tonight** | opening as draft |
| Option-6 bwd-chain reorder (probs-grad producer early) | recovers W1's bwd residual now that W3 is gone | model ~1–1.5 s | ~0 | UN-PARKED per the pre-registered stub rule, then **BUILT and CPU-proven tonight** (OPTION6_DESIGN.md + patch 6fbf3acc, tests green, W1-suite unaffected; mechanism sharpened: splits the fused sort's joint autograd node via its own row_id_map bwd op, seq-bumps the probs path past the fc1 pack, defers the early wait to the tokens reverse). Gate `BT_MOE_PROBS_BWD_REORDER=1`, requires W1, loud fallbacks. NOT booted — canary needs a slot you approve | patch ready; PR after canary |

## Stacked outlook (final, post-arms)

**Proven-and-shipping tonight**: golden config (B/F + NCCL env + TF32 head) + W1 + C′.
Model-level on prod fabric: W1 −3.4…−5.1 s + C′ ~0…−3 s compounding, on the 46.8 s step ⇒
the conservative proven stack is fermi's V0/V1 band (~41.7–43.4 s ≈ **~790–825 tok/s/GPU,
+10–15% model-level**); on these tail-bound boxes the measured wall is smaller by
construction (two-tier ruling).

**The bigger levers are pipeline, not dead** (full arithmetic + variants in
`SCORECARD_STACKED_OUTLOOK_draft.md`): W2 (−3.0…−4.4) failed its full-shape arm by hang —
diagnosis in the design doc, decision yours; W3 (−8…−12) refuted twice as implemented but
the wait-by-cause analysis mapped exactly what to change, with **A-v3 as the prerequisite**;
option-6 (~1–1.5) is built and CPU-proven awaiting a canary slot. If the W3-v4 path lands
on top of the proven stack, the original +40–60% ceiling remains live; tonight's honest
number is the +10–15% proven band.

## 4+ node scaling readiness (your F2 pivot)

DP2 validation: (a)+(a3)+(b) PASS, (c2) verdict TBD tonight. **DP4 probe: BLOCKED-BY-INFRA
tonight** — two 4×8 B300 provisioning attempts died (one node flake post-RUNNING, one
FAILED from DEPLOYING after 60+ min capacity-starved; our own fleet held 6 ali B300 nodes
during both). Everything needed to run it is READY: `f2_fix/DP4_SCALEOUT_PROBE.md` +
`expB-ep16cp8dp4.json`, mechanism bars pre-registered, W1 explicitly prohibited on
multi-EP-group topology (see W1 row). ~1 h exercise once a 4×8 box schedules — release
fleet boxes first. CP8's 3.5 GiB headroom violation @16k×d32 remains the ship blocker
to trim for the customer shape.

## PR list (all prepared/draft, NOT merged — per your order; independently slop-reviewed)

| PR | lever | state |
|---|---|---|
| Megatron-LM **#26** | B/F host caches (+FIX A parked default-OFF) | SHIP-READY per review |
| Megatron-LM **#27** | C/C′ replay cache + routing-force (stash re-keyed `id(router)` post-review) | **add an engagement/single-flag guard from open-item #1 before merge** |
| Megatron-LM **#28** | W1 probs second communicator (+`new_group` multi-EP-group fix + arm-time guard post-review) | fixed; **keep W1 OFF on >1-EP-group jobs until validated there** |
| trainers **#1000** | NCCL IB fabric env defaults (setdefault, operator-wins) | clean; retitled |
| trainers **#1001** | F2 phantom partitions (full DP2 evidence + ported test suite) | opened on verdict-complete |

Held closed: W2 (arm failed by hang — diagnosis attached), A-v3 / W3 skeletons (held /
refuted). Estate: JackRao123/experiment-artefacts — all evidence, frames, analyses, and
findings committed + pushed through the night.

## Open items at your wake (decision-ready, ranked)

1. **FINDING: C′ armed-but-not-ENGAGED on tonight's late boots**
   (`FINDING_FIXC_NOT_ENGAGING.md`): the closing arbitration proved the checker pins right
   and the boots wrong — replay eventsyncs 300 (era: 0), allgather 600 (era: 300) despite
   pasted arm-checks. **ROOT CAUSE CONFIRMED**: two-flag arming miss — all three late
   C′-boots set `BT_MOE_ROUTING_REPLAY_FORCE=1` without the underlying
   `BT_MOE_DISPATCH_REPLAY_CACHE=1` (48 "present but DISABLED (env unset)" warnings prove
   the code was in-tree and would have engaged; never-hitting and stale-clone candidates
   both refuted). Numerics verdicts survive — the affected
   boots ran the status-quo free-routing replay regime (a later code read showed the force
   half was ALSO inert without the cache gate: the checkpoint wrapper no-ops), which is the
   same regime every pre-C′ baseline ran, and the canaries' measured in-band deltas are
   direct evidence. Only the A-v3 timed wall was polluted; a clean both-flags re-run
   closed out the night. C′'s own era verdict unaffected (it
   engaged when measured). Process fix codified: engagement markers + hybrid flag-PAIR
   enumeration + `/proc` environ truth paste. **The C′ ship-PR (#27) should gain a
   single-flag/loud-composition guard so this class can't ship.**
2. **A-v3 clean-regime boot** (both C′ flags + engagement markers): re-runs the post-C′
   composition question — and it's the **W3-v4 prerequisite** (T6: host run-ahead).
   ~1.5 h on box 1 (idle-at-READY).
3. **W2 hang morning steps** (analysis doc sec 5): group-identity read, then W2+C′
   VERIFY=1 full-shape soak with NCCL_LOG=INFO + per-boot log rotation (two tooling
   papercuts filed tonight).
4. **W3-v4 + option-6**: v4 skeleton carries the full T6 anatomy; option-6 patch is
   built/CPU-proven and needs only a canary slot you approve.
5. **DP4 scale-out probe**: package ready; needs a 4×8 B300 that schedules (release
   fleet boxes first; two infra failures tonight).
6. F2 ship-gate hardening (1a/1b NaN×0 checksum canary) + per-replica pre-reduction loss
   tooling (gap found tonight); attention kill-shot probe (ULP source) still queued
   non-blocking.
7. **Boxes**: 318g61w (box 1) + wxlg05w (box 3, ⚠️ zero — one char from the dead box-2 id)
   idle-at-READY for morning arms; box 2 torn down, evidence on CPFS; teardown whitelist
   {318g61w, wxlg05w}.
