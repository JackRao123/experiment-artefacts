# W1 V1 COMM-2 READ — probs a2a second-communicator structural check

**Author:** jacobi (ex-kepler) · **Date:** 2026-08-14 · **Task:** kolmogorov
handoff per hausdorff's spec — confirm the probs a2a rides comm 2 off the main
compute stream (Aug-10-style off-stream check) on the W1 V1 mechanism traces.
**Traces:** `w3_v1_rank{0,8}.pt.trace.json` (702/749 MB, sha256 in
`pp2cp8ep8/w1c_traces.sha256`), served on :9010/:9011. d4 window (span ~42 s;
CF/CFB = 152 rank0 / 160 rank8 = 38/40 layers × 4 mb ✓).

## Verdict (one line)

**Stream separation CONFIRMED; overlap NOT realized in V1.** The probs a2a has
its own communicator/stream and the right traffic rides it — but the backward
probs-grad class (the actual W1 target from the Aug-10 map: 5.41 s/step
exposed at d16) is still ~85% time-exposed here, and the two comm streams
never run concurrently.

## (a) Distinct NCCL streams — CONFIRMED

| stream | rank0 track | rank8 track | content |
|---|---|---|---|
| compute | 4 (226k kernels + CP collectives) | 4 (239k + collectives) | main |
| **comm 1 (tokens a2a)** | 11: 840 SendRecv, 6.53 s, avg 7.8 ms | 12: 960 SendRecv, 6.73 s, avg 7.0 ms | dispatch/combine + grads |
| **comm 2 (probs a2a)** | 10: 420 SendRecv, 1.02 s, avg 2.4 ms | 10: 480 SendRecv, 1.39 s, avg 2.9 ms | **probs** |
| PP p2p | 24: 10 calls, 12.7 s (max 6.29 s) | 6: 10 calls, 1.5 s | stage-to-stage |

## (b) Probs rides comm 2 — CONFIRMED (structural)

- Call-share: comm2 carries exactly **1/3 of a2a calls** (420/840 r0,
  480/960 r8) = 3 probs-class calls per layer-mb (fwd + replay + grad).
- Payload signature: comm2 min 14–19 µs (tiny transfers ✓ 528 KB class);
  comm1 min 2.5–2.7 ms (token payloads).
- Phase split on comm2: **fwd 140/160 calls (1/layer-mb) + bwd-region
  280/320 (2/layer-mb: replay-probs + probs-grad)** — a clean 1:2. (This is
  likely the counter telemetry's 1:1:2 basis; my stream split is tokens:probs
  = 2:1 by calls. Counter owners reconcile.)
- Counter telemetry's ratio is thereby structurally confirmed in substance:
  the probs traffic exists, is separate, and fires at the expected cadence.

## (c) Temporal overlap — SPLIT VERDICT (the exception)

| comm2 class | calls | wall | hidden under compute |
|---|---:|---:|---:|
| fwd probs | 140/160 | 0.018–0.025 s | **96.9% — works** |
| **bwd probs-grad (W1's target class)** | 280/320 | **1.00/1.36 s** | **14.7% / 11.8% — NOT hidden** |

- comm1 vs comm2 mutual overlap: **0.5% / 0.4%** — the two streams' traffic is
  still serialized in time; comm2 bought stream *separation*, not concurrency.
- Longest comm2 calls (16–25 ms) overlap ~1 ms of compute and zero comm1.
- Read: the bwd-reorder's overlap is not visible in this V1 capture — either
  the reorder isn't in this build, or the probs-grad issue point still has no
  independent compute to hide behind (the GPU is otherwise idle during the
  wait). The wait itself is wait-for-peer (convoy), per the Aug-10 map's
  "~100% wait" — stream placement alone was never going to shrink it; the
  reorder was supposed to *move the wait off the critical path*. It hasn't,
  here.

## Consequences for the W1 lever (for bayes's queue)

- W1 V1 on the 2-EP-group topology (PR #28's `new_group` fix lineage):
  **mechanism structurally in, overlap prize unrealized at d4 in V1.** The
  −3.4…−5.1 s prod-fabric model number is NOT supported by this trace — the
  exposed probs-grad class persists (1.0–1.4 s in this ~1.1-step d4 window
  alone).
- Next diagnostic (cheap, source-side): does the V1 build actually contain
  the bwd reorder (issue-early/wait-late), or just the 2nd communicator? The
  trace says the wait still gates. If the reorder is in, the issue point needs
  to move earlier (before the next layer's bwd compute is enqueued).
- Standing caveat: d4 window, ~1.1 steps captured; a d16 V1 trace would
  re-size the exposed class against the Aug-10 5.41 s directly.

## Follow-up: source-side check (bayes-ordered, 2026-08-14) + option-6 canary pre-registration

**Q: is the bwd reorder in the V1 build?** NO — confirmed two ways
(kolmogorov's definitive statement + my independent re-verification):
`BT_MOE_PROBS_BWD_REORDER` is ABSENT on the local W1-lineage tip
(`jackrao/lps-1062-ship-w1` = d58a3214a, whose own log shows the W1
second-communicator guard d794ca3d2) and ABSENT on main-line 57efae08b;
present only in the option-6 chain (`jackrao/lps-1062-option6-staged`,
mcore 70710d116: transformer_engine.py/mappings.py/moe_utils.py — the exact
commits live in the fork remote, not this clone). The V1 build (w1-port @
a58990a91) lacks the reorder **by construction** — hausdorff's
never-co-armed sequencing. The V1 trace verdict is therefore exactly the
expected W1-alone shape.

**Q: is the 85%-exposed class precisely what the reorder attacks?** YES —
verbatim. The option-6 patch
(`runs/overnight_20260810_round3/overlap/patches/option6-probs-bwd-reorder.patch`)
makes the backward's probs reverse a2a the "early" sibling: it "issues its
reverse A2A and stashes the work handle in the carrier WITHOUT waiting... the
early issue's flight then overlaps the remaining backward instead of stalling
the compute stream at the early point"; the tokens reverse ("late" sibling)
waits both handles. The measured V1 class (bwd probs-grad on comm2,
1.0–1.4 s exposed per d4 window, 12–15% hidden, 0.4–0.5% comm1/comm2
concurrency) **is** that inline-wait stall. "Recovers W1's bwd residual" =
this class. Sizing consistency: 1.0–1.4 s per ~37 s d4 step here vs 5.41
s/step at d16 on the Aug-10 map — the modeled −3.4…−5.1 s at d16 is the same
class.

**Pre-registered option-6 canary trace reads (filed now, for whenever the
canary slot lands; requires W1 armed per the orders):**
- O1: bwd probs-grad on comm2 — exposure collapses from ~85% to **<20%**
  hidden-under-compute flips to >80%.
- O2: comm1/comm2 mutual overlap rises from ~0.5% to **materially >0**
  (the streams finally run concurrently).
- O3: no NEW exposed class appears at the late sibling (the late wait lands
  free); SendRecv totals flat-to-down.
- O4 (bench-side, not my trace): the modeled −3.4…−5.1 s at d16 against the
  same-window bench.
- O5 (canary, binding): loss 12.2–12.4 band + gn comparable; the reorder
  changes schedule, not numerics — drift = STOP.

## W4 ON-arm d16 mechanism closure (2026-08-14; trace released by kolmogorov)

**Trace:** `w4_on_d16_rank0.pt.trace.json` (2.82 GB, sha256 logged, :9012);
rank8 pending transfer at write time (cross-check to follow — see end).
**Marked confound (bayes):** this trace is the **no-TF32 class** (swap
confound) — irrelevant to the mechanism fractions below; noted for the
record. Span 145.01 s (vs fe127's 133.4 — confound + arm; perf is
bench-side, not this read). CF/CFB = 608/608 ✓ d16 rank0. Stream map:
comm1 (tokens) = track 10 (3,360 SendRecv, 25.30 s); **comm2 (probs) = track
9 (1,680, 4.17 s)**; PP = track 24 (34 calls, 34.68 s). a2a total 5,040 =
fe127's exact count; comm1:comm2 = 2:1 ✓ (W1's stream split intact at d16).

### O1–O3 reads vs the pre-registered bars

| read | bar (pre-registered) | measured (W4 on-arm r0) | verdict |
|---|---|---|---|
| O1 bwd probs-grad hidden-under-compute | collapse to **>80% hidden** | **26.6% hidden** (1,120 calls, 4.05 s wall) — V1 d4 was 12–15% | **PARTIAL — bar not met** |
| O2 comm1/comm2 mutual overlap | materially >0 (from ~0.5%) | **12.8% of combined** (3.78 s) | **MET** |
| O3 no new exposed class at the late sibling | none | none — comm1 87.5% exposed (V1 ~90%), the standing lever-1 prize, not a new class | **MET** |
| (reference) fwd probs on comm2 | hidden | 560 calls, 0.12 s, **97.6% hidden** | ✓ (as V1) |

### The closure answer: **FIRED, PARTIAL — neither clean binary**

Not "never fired" (no arming bug): comm1/comm2 concurrency appeared
(0.5%→12.8%), the bwd hidden fraction doubled (13%→26.6%), and the firing
position moved into the backward — comm2 bwd calls land at **p50 = 0.69 of
the enclosing layer-backward span** (p10 0.45 / p90 0.87), i.e. the
issue-early half of the early/late protocol is visibly in effect.

Not "fired and shrank to bar" either: **73% of the bwd probs-grad wall still
finds an empty compute queue.** The discriminator inside the firing-position
data: early-half fires (n=309) complete in **0.27 ms avg** (fully hidden —
peer already there); late-half fires (n=811) carry **4.89 ms avg** waits.
So the residual exposure is the **wait-for-peer/convoy tail on late-half
fires** plus thin remaining-compute cover after ~0.7 of the layer backward —
not the missing-reorder class. Sizing (cross-baseline, caveated): the Aug-10
map's 5.41 s/step exposed probs-grad vs ~2.7–3.0 s/step exposed here →
**roughly half the modeled prize recovered** (model −3.4…−5.1 s). The report's
option-6 post-mortem sentence, verbatim-ready: *"The reorder fired and its
early/late protocol is trace-visible, but the probs-grad exposure only
roughly halved — the residual is convoy stagger on late-half fires, which
issue-point placement alone cannot remove."*

Caveats: (i) no same-night W1-only d16 off-arm exists (V1 was d4) — the
recovery estimate crosses baselines (Aug-10 golden EP16/CP16, old wheel) and
is caveated accordingly; (ii) rank8 cross-check pending (transfer in flight);
(iii) no-TF32 confound marked, mechanism fractions unaffected.

### Reconciliation sentence (bayes-ordered, on record)

> **Exposure halved (~2.7 s recovered vs the Aug-10 class) yet the bench is
> flat-to-negative, so CONVERSION ≈ 0: the reorder moved local waits off the
> compute stream, but step time is set by the slowest rank's convoy, and a
> locally-hidden wait is not throughput.** The same lesson as W1c, inverted:
> B/F removed work *every* rank did (conversion > 1 via launch
> decompression); option-6 hid waiting that only *some* ranks do. The
> residual exposed-comm mass is convoy/imbalance-dominated, and the next real
> lever is **BALANCE (expert/token de-staggering), roadmap-scale** — not more
> local overlap.

**Evidentiary scope (honest split):** the trace supports the mechanism half
at rendezvous granularity — the probs-grad wait distribution is pure
peer-arrival skew (peer-ready floor 0.27–0.43 ms vs staggered tail p90
9.8 ms / p99 15.7 ms; top-10 longest waits own only 0.16 s — the mass is in
the tail), unchanged in character by the reorder; only its placement moved.
The conversion fact (bench flat-to-negative) is bench-side (bayes/lovelace).
**Not trace-testable tonight:** per-rank step-time spread ON vs OFF — no W4
off-arm pair exists, and PP lockstep makes step spans equal across ranks by
construction; the convoy evidence here is at the rendezvous level, which is
the level the sentence's mechanism claim actually needs.

### rank8 cross-check (landed 2026-08-14 ~14:2x; 3.01 GB, sha256 logged, :9013)

Span 145.04 s = rank0's 145.01 s (same step ✓). CF/CFB = 640/640 ✓ (40
layers). Stream map: comm1 = track 11 (3,840 = 640×6 exact), **comm2 =
track 10 (1,920 = 640×3 exact)**, PP = track 6 (34). 2:1 split ✓.

| read | w4 rank0 | w4 rank8 | consistency |
|---|---|---|---|
| O2 comm1/comm2 mutual overlap | 12.8% | 13.8% | ✓ both stages |
| O1 bwd probs-grad hidden | 26.6% | **18.7%** | ✓ same shape, stage 1 worse |
| fwd probs hidden | 97.6% | 97.2% | ✓ |
| firing position p50 in bwd span | 0.69 | 0.68 | ✓ identical issue point |
| early-half fire duration | 0.27 ms | 0.22 ms | ✓ peer-ready floor |
| late-half fire duration | 4.89 ms | **6.42 ms** | stage 1's convoy tail longer |
| comm2 exposed wall / step | ~3.0 s | **~4.6 s** | stage 1 carries more of the residual |

**The closure verdict is two-rank consistent: FIRED, PARTIAL, convoy owns the
residual — on both stages, with stage 1 modestly worse** (18.7% vs 26.6%
hidden; late-fire waits 6.42 vs 4.89 ms). Direct stagger note: both ranks
fire at the same relative point (p50 ≈ 0.68–0.69) yet wait differently — the
wait is set by the *peers'* arrival, and the EP group's other 6 ranks are
unprofiled tonight (`BT_PROFILE_RANKS=0,8` = one rank per stage): the
convoy's existence is trace-visible, its per-rank composition is not.

**Correction to the rank0 section's sizing language (honest chronology):**
the "roughly half the modeled prize recovered" estimate crossed baselines
(Aug-10's 5.41 s/step was a different topology/wheel, and its 640-call figure
is the 40-layer stage — the honest same-stage W4 read is stage 1's ~4.6 s
exposed, i.e. only ~−16% against that number). The well-evidenced same-night
claims are the within-trace ones: O2 concurrency appeared (0.5% → 13–14%),
the issue point moved into the backward (p50 0.68–0.69), and the early/late
fire-duration split (0.22–0.27 ms vs 4.9–6.4 ms) shows the residual is
peer-stagger. The exposure-reduction *magnitude* lacks a same-night W1-only
d16 baseline (V1 was d4) — stated, not overclaimed.

## Method note

Stream = perfetto track_id. Exposure/overlap via window-function union
coverage (cover(A∪B) − cover(A)); phase tagging by CF/CFB containment. No
enclosure joins (the :9001/:9005 lesson). Queries: w1_streams.py /
w1_overlap.py in the opencode tmp dir (transient); reproducible from this
doc's definitions.
