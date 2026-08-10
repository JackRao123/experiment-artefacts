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
| W2 chunked MoE a2a pipeline K=2 (`BT_MOE_A2A_PIPELINE=2`) | overlaps token a2a with expert compute via per-peer views | model −3.0…−4.4 s standalone; **timed arm result: TBD** | ≈0 (views) | gate-clean (44/0/0 on canonical gate d88d8b7d; 0626 FAIL was 2 harness artifacts); timed arm TBD; 6144 fidelity re-run TBD | TBD |
| W3 lookahead recompute | overlap bwd(L) with recompute(L−1) | v2 REFUTED as implemented (kicks 99.9% serialized — wait_stream ordered each kick behind the whole compute backlog). **But the SM-slack gate came back POSITIVE: 35.9 s of bwd SM slack/step, comm windows at 0% occupancy — the serialization was v2's ordering design, not hardware.** −8…−12 s model credible again for a v3 (input-dependency kick ordering); HBM-contention caveat open | +25.8 GiB measured at 131k (~11.5 intrinsic + ~14 recoverable via allocator knob) | v2 DISARMED; 16k hard OFF; **W3-v3 design in flight tonight (fermi): TBD** | TBD |
| A-v3 (`torch.nonzero` batching, retry post-C′) | removes DSA-bwd host sync, unthrottled now that C′ landed | tonight's timed arm INVALIDATED (ran under-armed — C′/W1 off — mismatched gates; caught by our own boot-region audit). A-v3's isolated mechanism rows PASSED (312/312 eventsyncs, 1.9 ms CPU); the post-C′-regime composition measurement is the open item | ~0 | soak PASS (B/F-regime); **re-arm with full stack = morning item unless a slot freed tonight** | skeleton drafted, opens on valid verdict |
| F2 phantom partitions (DP>1 deadlock fix) | correctness for any DP>1 customer + **unlocks 4+ node scaling** (EP16/CP8/DP4 ≈ 2× aggregate) | not a 2-node lever; scale-out enabler | no change @DP=1 | **(a) PASS + (a3) PASS ⇒ CAUSALITY PROVEN** (fix-ON boots DP2 clean; flag-OFF reproduces the exact deadlock, py-spy signatures matched); **(b) PASS** (exact Aug-9 deadlock shape completes, losses sane); (c) re-designed mid-flight: DP1-golden comparison INVALID-by-design (config-topology drift −34.6e-3 at warmup0 with phantoms proven-not-fired dominates the 5e-3 bar) → verdict rides on (c2) equal-count flag-ON/OFF single-variable A/B: TBD. **Scope note: the (c) suite proves fix-INERTNESS at equal counts; phantom-FIRED numerics are design-argued (zero-mask) + (b)-sane, with the 1a/1b NaN×0 checksum canary as the SHIP gate — not claimed proven tonight** | TBD |
| Option-6 bwd-chain reorder (probs-grad producer early) | recovers W1's bwd residual now that W3 is gone | model ~1–1.5 s | ~0 | UN-PARKED (minkowski, per pre-registered stub rule); design spec in progress (fermi); not run tonight unless a slot frees | design-only |

## Stacked outlook (post-W3-refutation)

TBD — recompute once W2/A-v3 arms land. Static model with W3 removed:
W1 (−3.4…−5.1 prod) + W2 (−3.0…−4.4, if arm passes) + C′ (compounding ~0…−3) + option-6
(~1–1.5, unbuilt) on 46.7 s ⇒ ~38–40 s ≈ **~840–900 tok/s/GPU (+18–26%)** model-level;
tail-bound-box measurements will read lower by construction.

## 4+ node scaling readiness (your F2 pivot)

TBD — depends on tonight's DP2 validation. If PASS: EP16/CP8/DP4 on 4×8 B300 is the
scale-out config (≈23.2K agg tok/s model, 2× the 2-node winner); CP8's 3.5 GiB headroom
violation @16k×d32 remains the ship blocker to trim. DP4 probe: TBD (needs a 4×8 box).

## PR list (prepared, NOT merged — per your order)

TBD — filled in the PR phase. Branches: Megatron-LM `jackrao/lps-1062-r3-stack`
(tip 15d5679e — carries W3 v2, which must be excluded from ship PRs),
`jackrao/lps-1062-r3-macship-snapshot`; trainers `jackrao/lps-1062-r3-f2`,
`jackrao/lps-1062-f2-main-rebase`. Estate: JackRao123/experiment-artefacts@2263b82+.

## Open items at your wake

TBD — whatever didn't finish: SM-slack verdict, 6144 fidelity, DP4 probe, option-6 build,
attention kill-shot probe (ULP source, queued non-blocking).
