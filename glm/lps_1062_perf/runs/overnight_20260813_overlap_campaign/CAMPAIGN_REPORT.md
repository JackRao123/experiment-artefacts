# LPS-1062 Overlap Campaign — Final Report (FINAL — all windows closed, incl. P4 and the conditional warm-pool probe)

Orchestrator: bayes (Claude; began as fermi, renamed in the 08-14 ~00:00
session wave). Fleet: lovelace (box mechanics, ex-doppler), kolmogorov
(B/F + adjudication, ex-bohr), pauli (memory leg, ex-jacobi), ramanujan
(contract shim, ex-lebesgue), jacobi (trace forensics, ex-kepler),
grothendieck (W1b driver + support, fresh), hausdorff (W1/option-6 staging;
lost in the wave, work survives). Campaign window: 2026-08-13 19:03 →
2026-08-15 19:03 CDT (48h). Written in plain language per Jack's standing
rule; every number's provenance is in NOTEBOOK.md and the per-window docs.

## 1. Headline

**New number of record: 1103 tok/s/GPU @ d16 131k (mfu3x ~10.1%), +12.1%
over the campaign-start record of 984.** Honest decomposition, measured
separately:

| term | size | what it is |
|---|---|---|
| env/box-class | +6.8% | the old 984 was set on a slower box/venv-era; the exonerated fresh baseline at the record env measured 1050.5 (see §4, the provenance saga) |
| B/F host-sync caches | +5.0% | the campaign's ratified lever — see §2 |

Customer-shape bonus: **B/F pays +6.2% at 16k-d32** (the shape customer
jobs actually run), confirming the docs-per-partition scaling mechanism.

## 2. The one fully-shipped lever: B/F (PR #26 — merge-ready)

BT_DSA_CP_LAYOUT_CACHE + BT_THD_ROPE_HOST_CACHE. Complete evidence chain:

- **A/B:** d16 +5.0% (1103 vs 1050.5; |Δ|=52.5 > max-spread 30 —
  distinguishable; dead-center of the three pre-registered bands). d4 warm
  +4.9%. 16k-d32 +6.2% (band +5-12%). Memory flat. Canaries clean, both
  cache gates ACTIVE on 16/16 ranks.
- **Mechanism (pre-registered, then measured):** host `nonzero` calls
  55,328 → 1,328/step (prediction: ~1,330). RoPE host time 35.3 → 1.07 s.
  Pure GPU idle 12.00 → 4.07 s — more than the calls alone explain, because
  −539k kernel launches/step collapsed the micro-gap dispatch tax
  (9.51 → 0.89 s): the long-hypothesized "launch decompression" measured
  directly. Traced-step span −8.3%.
- **Ablation:** the Aug-7 ship NCCL knobs (QPS/split/channels) are INERT on
  this topology (920 vs 926) — keep-for-safety guidance in merge item 9.

The campaign's opening "20.8 s idle problem" is now a **4.07 s problem**;
the dominant residual is ~2.66 s of step-seam python that B/F *unmasked*
(not created — H1 CONFIRMED by the R2 py-spy probe; cache maintenance
measured 0% of seam samples on both ranks). The seam is now NAMED
(R2_SEAM_PROBE.md §5): rank8's ~7.1 s/step `broadcast_object_list`
peer-loop rendezvous is the largest item (likely boundary skew wearing a
broadcast), plus a ~4.1 s full-device shape-sync
(`_communicate_shapes`) and THD packing; scoped fix shapes (async
broadcast emission, static-shape sync skip, packing prefetch) carry an
EV of ~1–2 s of the 2.66 s — roadmap items, not campaign-window work.

## 3. The overlap program: enabler shipped, prize blocked upstream

The campaign's biggest theoretical prize (15-19%: hiding the 34.3 s/step of
exposed expert all-to-all) required three legs. Status:

- **Leg (a) contract shim — DONE and hardware-validated.** Branch
  `jackrao/lps-1062-overlap-contract-shim` @1cd31535: our trainer can now
  drive Megatron's overlap executor (it never could before). Proven: boots,
  trains, loss in band; the frozen-embedding landmine fix validated at its
  first real backward; a latent chunked-LM-head dtype bug (fp32 boundary ×
  bf16 LoRA adapters) found and fixed en route. Merge posture: default-off
  + documented caveat, Jack's call.
- **Leg (b/c) memory — built, purpose-held.** Per-layer recompute dial:
  built, reviewed, 7/7 hardware correctness proofs (incl. negative control
  and bitwise recompute-equivalence under dropout); executor rungs HELD.
  Full status: MEMORY_LEG_FINAL_STATUS.md.
- **The blocker (upstream):** Megatron's combined-1F1B executor is
  **nondeterministic on DSA within a single boot** — identical data, same
  boot, per-token logprobs diverge up to 6.3 (bar: 1e-3), spikes recurring
  at DSA top-k boundary positions; aggregates stay tight (rel ~3e-5), so
  only a per-token parity gate catches it. Escalation with full reproducer:
  results/UPSTREAM_ESCALATION_DSA_EXECUTOR.md. First suspect: the DSA
  kernel-race class from the wheel saga, exposed by the executor's
  interleaved scheduling. **UPDATED 2026-08-14 (P4 noise matrix,
  escalation §7): the exposure framing is falsified — the conventional
  path measures the same per-token floor (3.7–5.5 within-boot, 95.7%
  pervasive), so the nondeterminism is schedule-independent; attribution
  re-sited to the DSA indexer top-2048 selection. The upstream ask
  stands, sharpened; read the escalation's §7 before forwarding.**

**Consequence:** the overlap prize waits on the upstream fix; when it
lands, the shim + dial mean we can A/B it immediately.

## 4. The provenance saga (why tonight's numbers are trustworthy)

- Jack's fresh-install order caught the inherited venv carrying a silent
  hadamard substitution (mystery PyPI 1.0.4.post1 vs the lock's git
  v1.1.0). Full rebuild + wheel re-bump + DSA gates before any measurement.
- The first fresh anchors came in −4% and STOPPED per protocol; root cause
  was an under-specified env (TF32 LM-head gate defaults OFF + ship NCCL
  knobs absent) — not the venv. The canonical mission env now lives in
  BOX_SCHEDULE.md; W1c boot-1 at the full env EXONERATED the venv by
  beating the old record (1050.5 > 984).
- blockK partial recompute: **does-not-fit at 131k** (two pre-registered
  misses: K=21 OOM ×2, K=25 at 265 GiB reserved vs the 255 bar). The
  memory model was corrected twice by evidence (S_eager 2.25→2.94
  GiB/layer/mb fully-eager; then the triple-metric and step-creep findings)
  and now postdicts both deaths exactly. Conditional warm-pool revival
  pre-registered fail-closed (BLOCKK_WARMPOOL_PREREG.md).
- **Two campaign-wide measurement rules learned:** (1) memory A/Bs compare
  at matched step positions or plateau (torch-reserved creeps ~+20 GiB over
  early steps); (2) never mix the three memory metrics (torch-allocated <
  torch-reserved < nvidia-smi; they differed by 12 GiB at one crash point).

## 5. W1 second-communicator + option-6 (W3/W4) — CLOSED: reverts to default-off

- **W1 correctness: all gates passed** (V0 16/16 armed + full optimizer
  step; V1 counters 1:1:2; V2 zero drift — bitwise-safe held). The
  2-EP-group port (PR #28 + guard-relax) works.
- **W1-alone perf: null**, exactly as the trace predicted before the bench
  ran (the speed payload was always option-6's; the redundant full A/B was
  stood down early).
- **option-6 (the bwd reorder): fired mechanically, lost on throughput.**
  Traces (two-rank consistent, rank0+rank8): comm1/comm2 overlap 0.5% →
  12.8/13.8%, issue point moved early identically on both stages, the
  targeted probs-grad exposure shrank — and the bench still measured
  flat-to-negative. Conversion ≈ 0. (Recovery MAGNITUDE is deliberately
  not headlined: the "vs Aug-10" comparison crosses topology/wheel
  baselines and no same-night W1-only d16 off-arm exists — jacobi's
  correction, scope marked in the read doc. The within-trace mechanism
  claims are the evidenced ones.) The wait distribution is pure peer-arrival skew (floor 0.27-0.43 ms
  vs staggered tail p90 9.8 ms), unchanged in character by the reorder —
  only its placement moved.
- **The campaign's closing synthesis (evidence at the rendezvous level,
  scope marked in W1_V1_COMM2_READ.md):** you cannot hide waiting for a
  straggler. B/F paid MORE than predicted because it removed work every
  rank did; option-6 paid nothing because it hid waiting only some ranks
  did — the slowest rank still sets the step. The residual exposed-comm
  mass (~30 s/step) is convoy/imbalance-dominated; the next real lever is
  **balance (expert/token de-staggering) — roadmap-scale**, not more local
  overlap. Every independent measurement tonight (AllGather skew signature,
  SendRecv waits, option-6's null conversion) points at this same wall.
- Process note: the W3/W4 tree swaps dropped the TF32 patch (caught,
  quarantined — internal validity unaffected; papercut filed; durable fix
  = landing PR 995).

## 6. Soak + parity gates (P4) — COMPLETE: ship stack validated, two latent defects caught and fixed en route

P4 (S2 parity → S1 60-step soak → S3 export + sync-save) ran 2026-08-14
afternoon/evening under kolmogorov (adjudication) and lovelace (driving),
with turing orchestrating. All three legs closed green — but the leg-level
story is the campaign's best argument for running soaks at all, because
both of tonight's latent defects were invisible to every earlier bench.

**S2 (parity) — bars failed, caches exonerated, a measurement first.**
The off-vs-on compare missed its bars by orders of magnitude (per-token
max-abs 3.4953 vs 1e-3, 95.7% of tokens; loss rel 9.11e-5 vs 1e-6). The
localization was rigorous and is worth reading as method: env drift
eliminated (boots env-symmetric, shas byte-exact), the LPS-1063
uninit-LSE class eliminated (fix present + guarded in the ship tree), one
adjudication error caught and retracted on the record (stale W2-era
executor files briefly analyzed as fresh controls — the provenance tells
and the correction are in the NOTEBOOK, marked in place), and then a
pre-registered four-cell noise matrix (S2_NOISE_MATRIX_PREREG.md)
produced the campaign's **first-ever direct measurement of the
conventional path's per-token determinism**: the path does not reproduce
ITSELF — within-boot repeats diverge at max-abs 3.7–5.5 (B/F OFF) and
4.96–5.31 (B/F ON), cross-boot 4.27, all ~95.7% pervasive. The off-vs-on
divergence sits inside every floor measure. **Verdict (kolmogorov): B/F
exonerated — bitwise-exact by construction, noise-indistinguishable on
hardware. The nondeterminism is the DSA indexer's top-2048 selection,
schedule-independent** — which also falsified the upstream escalation's
"conventional clean, executor exposes" framing (corrected: escalation §7,
hertz). Standing rule adopted: **all parity gates on this stack are
noise-relative** (the floor sits 3–6 thousand × above the old per-token
bar).

**S1 (60-step soak) — PASS with two flags, both resolved.** 60/60
optimizer steps at d16/131k; deep-settle throughput 1080–1091, refining
the number of record's steady-state to the **1089–1103 band** (the A/B's
1103 was the fresh-pair read of the same quantity — no new-record claim,
per the pre-registered read); loss/gn trajectories training-healthy; B/F
telemetry zero fallback/miss/stale ×60. Flag 1: memory +2.8 GiB over the
soak vs the 2 GiB bar — resolved BENIGN on the metric split
(torch-allocated flat at the reference class; all growth in
reserved-not-allocated allocator pool; slope decelerating — a
constant-rate leak does not decelerate). Flag 2: a one-step gn spike at
main54 (0.97, full recovery next step, clean logs) — an acknowledged
bar trip whose event is benign-class data variance; future-soak tooling
item: a per-step watcher instead of batched reporting. jacobi's R2 seam
probe rode an S1 idle gap undisturbed (result in §2).

**S3 (export + save) — caught the campaign's second latent defect, then
validated the real fix end-to-end.** The first `/save_state` of the
entire campaign (benches never save) WEDGED the trainer at soak-end:
async DCP save under CP>1 — the known F1 landmine — because
**`BT_SAVE_STATE_SYNC`, set on every boot per the standing rule, is read
nowhere on this tree** (async_save hardcoded on;
`megatron_config.py:325`). The workaround was a silent no-op all
campaign; the false comfort is itself the finding (silent-fallback
family, 4th member; papercuts filed; wedge evidence banked as a fresh F1
reproduction — s3_wedge_snapshot/). Recovery: the db5d1826 toggle hunk
applied + sha-verified (merge-queue item 6, re-prioritized
SHIP-CRITICAL), 5-step burst, S3 re-run — **all green**: 1094-key
complete GLM-5.2 LoRA export (the spec's 392-key bar was the Qwen3
reference; scope note ratified + spec corrected in place), L1
non-degeneracy 547/547, sync save 65.9 s vs the 5-minute bar (2.7 GB
checkpoint complete with metadata) with the toggle genuinely gating
async off, post-save resume in band (d2 loss 12.2804, gn 0.374). The F1
workaround is now real on this tree for the first time. P4 verdict
formally kolmogorov's, evidence-seen: **the ship stack (mission + B/F)
is VALIDATED.**

Post-P4 conditional probe (blockK K=25 warm-pool revival;
pre-registered, fail-closed, one boot; curie driving, hertz
adjudicating): the warmup0 gate reproduced the W1b reference exactly
(256.7/264.3 vs 256.8/265.0; canary in-band), then the first d2 main
(M=2) hit a structural CUDA OOM — stage-0 demand ramped +50 GiB with
pipeline fill (217.3 → 267.4 vs the 267.69 cap) while the worker sat
flat. The warm-pool hypothesis is refuted: the 265-class peak was M=1
structural demand, not cold-pool excess; M≥2 is infeasible at K=25/131k
(the mission runs M=16). W1b's DOES-NOT-FIT stands,
mechanism-explained: the binding term is M-scaling of stage-0
activation holdings, not a cold-pool artifact. The dial K floor stands
at 25+ with the valve stack as the coverage path. Full record:
blockk_warmpool_probe/PROBE_RESULT.md. The box then went idle-armed for
Jack's stop/keep.

## 7. Levers investigated and closed without box time (the cheap kills)

- NCCL CP-AllGather tuning: killed at a Mac-side gate — the exposure is
  ~84% peer-arrival skew (min-at-wire-floor / avg-18×-floor signature);
  no knob addresses skew, only overlap does.
- PP layer rebalance: killed — DSA top-k sharing quantizes legal splits to
  4-layer moves; the 38/40 split is already the optimal legal cell.
  Zero-cost reopen condition attached to the W1c traces (stayed dead).
- "Sync-free indexer top-k" as a standalone lever: dissolved — the leak was
  mis-sited; FIX B's cache owns 98.9% of it (confirmed by the W1c traces).
- CUDA-graphs / launch-tax surgery: scoped honestly to a roadmap item
  (LAUNCH_TAX_SCOPING.md); unlock chain runs through B/F+A-v3 anyway, and
  B/F's measured launch collapse (§2) already took the biggest bite.
- Aug-9's +18.5% convoy-regime B/F expectation: did not transfer post-M=N
  (measured +6.2%) — runahead absorbs most of the old convoy cost.

## 8. Incidents and lessons (honest ledger)

1. **Capacity war:** ali B300 full for 4+ hours; the platform silently
   reaps pending jobs at ~60 min INCLUDING half-scheduled ones (held a
   Running node 42 min, lost it). Five creation cycles to land. Papercuts
   filed (silent reap, no error_message; devbox-up 429 fragility).
2. **The 00:00 session wave** (Mac opendirectoryd outage): every session
   renamed, one lane's context lost (doppler's — recovered from disk docs),
   one wedged provisioning driver caught double-driving and deconflicted.
3. **Blocking-wait violation:** one driver blocked 60+ min on a manual
   trainer wait (sleep+ssh-stare) and went unresponsive; lane handed off at
   the boundary. Jack's zero-tolerance rule broadcast; all six sessions
   acked. wait_trainer_health.sh, backgrounded, always.
4. **The smoke lesson (orchestrator's own):** a cut-down-model smoke test
   was built, then shelved when the real box landed; the W2 dtype bug it
   would have caught for free cost a mission-box boot cycle instead.
   Opportunistic hardware de-risking has positive EV even when the real
   gate is imminent.
5. **Pre-registration paid all night:** every verdict in this report was
   accepted or refuted against bars written before the data existed —
   including three self-corrections by the fleet against their own earlier
   claims, each marked in place in the NOTEBOOK.

## 9. Merge queue for Jack (morning)

MORNING_MERGE_QUEUE.md updated in place. Ready: **PR #26 (B/F) —
unconditional** (item 7, full chain above). Staged behind decisions:
shim branch (default-off + caveat, blocked-on-upstream), A-v3 W1-free
restack, W1 port + option-6 (W3/W4 closed-negative — both revert to
default-off, §5), dial branch (held).
Box disposition (wprm693, wlxj8vw stop/keep) = Jack's call.

— bayes, 2026-08-14, through §5; §6 + current-state edits by turing
(orchestrator from ~14:1x CDT by succession; §6 closed ~18:3x CDT on
kolmogorov's leg adjudications)
