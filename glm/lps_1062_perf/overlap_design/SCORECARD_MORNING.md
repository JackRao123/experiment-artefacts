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

## Lever scorecard

| lever | what it does | TPS/GPU worth | memory cost | verdict / status | PR |
|---|---|---|---|---|---|
| B/F host-cache pair (`BT_DSA_CP_LAYOUT_CACHE`, `BT_THD_ROPE_HOST_CACHE`) | kills dispatcher-side host syncs | +11–13% @131k, +18.5% @16k customer shape (parity-proven, prior shift) | ~0 | SHIPPED into golden config | TBD |
| NCCL env defaulting (+ TF32 LM head) | ship config env block as code defaults | part of the +51% 256k ship config (prior shift) | ~0 | proven, needs defaulting PR | TBD |
| W1 probs-a2a second communicator (`BT_MOE_PROBS_A2A_COMM=1`) | de-serializes 900×5.13 s of latency-bound probs a2a off the token stream | mechanism PASS (900/900 off-stream, gap 11.2→9.2 ms); wall −0.4–0.6 s here; **model −3.4…−5.1 s (+7–11%) on prod fabric** | +1.3 GiB (2nd NCCL comm buffers) | SHIPPED-as-v1; residual understood (late probs-grad producer) and bounded. **PR-review finding (fixed on branch): the 2nd-communicator `new_group` call violated the torch contract on >1-EP-group topologies (would hang at 4-node EP16); all 2-node validation unaffected (single EP group); fix = use_local_synchronization + arm-time guard — flag W1 must stay OFF on multi-EP-group jobs until the fixed build is validated there** | TBD |
| C′ dispatcher replay-metadata cache (`BT_MOE_DISPATCH_REPLAY_CACHE=1`, hybrid routing-map form) | replay-side eventsync 300→0/step, dispatcher allgather 600→300 | mechanism PASS both ranks; wall ~flat here (value compounds as comm gets hidden) | ~0 (2 MB routing map ×in-flight) | PASS (verify-soak 20/20 + timed arm; drain-seconds FAILs adjudicated wait-redistribution w/ conservation proof) | TBD |
| W2 chunked MoE a2a pipeline K=2 (`BT_MOE_A2A_PIPELINE=2`) | overlaps token a2a with expert compute via per-peer views | model −3.0…−4.4 s standalone; **timed arm result: TBD** | ≈0 (views) | gate-clean at small scale (44/0/0 on canonical gate d88d8b7d), but the **TIMED ARM FAILED BY HANG**: first backward at full 131k with the C′+W1 composition deadlocked (NCCL collective timeout, non-default PG — W1-comm vs W2-pipeline identity unresolved). Reference arm was clean (701 tok/s/GPU). Full-shape/composition failure class — the small-shape gate could not see it. Lever STOPPED; failure analysis in the design doc; 6144 fidelity re-run moot until the hang is diagnosed | held closed |
| W3 lookahead recompute | overlap bwd(L) with recompute(L−1) | v2 REFUTED as implemented (kicks 99.9% serialized — wait_stream ordered each kick behind the whole compute backlog). **But the SM-slack gate came back POSITIVE: 35.9 s of bwd SM slack/step, comm windows at 0% occupancy — the serialization was v2's ordering design, not hardware.** −8…−12 s model credible again for a v3 (input-dependency kick ordering); HBM-contention caveat open | +25.8 GiB measured at 131k (~11.5 intrinsic + ~14 recoverable via allocator knob) | v2 DISARMED; **v3 canaried tonight and REFUTED on the verdict row too — capture 0.9% vs ≥50% bar** (plumbing fully correct: kicks land in the right windows 99.5%, numerics near-bitwise, memory +24.9 GiB ≤ +28 bar; the win mechanism didn't materialize; slack confirmed still present, 34.65 s). **Wait-by-cause analysis then MAPPED the residual serialization** (W3V3_T6_WAIT_ANALYSIS.md): (i) the allocator-safety edge releases only after each chunk's comm tail drains (75% of gate-releases are token-A2A ends); (ii) kicked work fragments ~5×/chunk and re-gates on the SHARED comm/expert streams; (iii) main queues are EMPTY at release because host run-ahead is drain-throttled with FIX A reverted. **v4 path is mapped, not dead**: dedicated kick-A2A stream, allocator-edge surgery, side-stream priority — and critically, **A-v3 is a prerequisite** (restores host run-ahead so releases meet queued work). Option-6 (~1–1.5 s bwd-chain reorder) revived and building Mac-side | no PR; v4 design doc for morning |
| A-v3 (`torch.nonzero` batching, retry post-C′) | removes DSA-bwd host sync, unthrottled now that C′ landed | tonight's timed arm INVALIDATED (ran under-armed — C′/W1 off — mismatched gates; caught by our own boot-region audit). A-v3's isolated mechanism rows PASSED (312/312 eventsyncs, 1.9 ms CPU); the post-C′-regime composition measurement is the open item | ~0 | soak PASS (B/F-regime); **UPGRADED by the W3 wait analysis: A-v3 is the L3 prerequisite for any W3-v4 capture (host run-ahead). Re-arm with full stack running as the night's final experiment** | skeleton drafted, opens on valid verdict |
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

## PR list (prepared, NOT merged — per your order)

TBD — filled in the PR phase. Branches: Megatron-LM `jackrao/lps-1062-r3-stack`
(tip 15d5679e — carries W3 v2, which must be excluded from ship PRs),
`jackrao/lps-1062-r3-macship-snapshot`; trainers `jackrao/lps-1062-r3-f2`,
`jackrao/lps-1062-f2-main-rebase`. Estate: JackRao123/experiment-artefacts@2263b82+.

## Open items at your wake

TBD — whatever didn't finish: SM-slack verdict, 6144 fidelity, DP4 probe, option-6 build,
attention kill-shot probe (ULP source, queued non-blocking).
