# IDLE RESIDUAL MAP — measured host-side classes vs the staged lever stack

**Author:** kepler · **Date:** 2026-08-13 · **Task source:** fermi assignment 2 (trace+source, Mac-side)
**Trace:** `fe127_d16_rank0.pt.trace.json` (133.350 s step, rank 0), served on
**http://localhost:9002** (kepler's trace_processor, pid 29344 — fleet trace
server going forward; the old :9001 was retired by fermi after my join wedged
it — see IDLE_WINDOW_DECOMPOSITION.md §4).
**Companion:** `IDLE_WINDOW_DECOMPOSITION.md` (bucket definitions, gap method).
**Stack under reconciliation:** FIX B (`BT_DSA_CP_LAYOUT_CACHE`, PR #26) + FIX F
(`BT_THD_ROPE_HOST_CACHE`, PR #26) + FIX C′ (PR #27) + FIX A-v3 (PR #29,
restacked `jackrao/lps-1062-ship-av3-wo-w1` per bohr's REBUILD_NOTE) — the
Aug-9 dispatcher-hostsync campaign (`runs/overnight_20260809_dispatcher_hostsync/`).

---

## 0. TL;DR

- The 55,328 `aten::nonzero` calls/step decompose cleanly by site:
  **54,720 (98.9%) are the DSA CP layout builders = FIX B's target**;
  **608 (1.1%) are the bwd `topk_length` compaction = FIX A/A-v3's target** —
  but the 608 carry **27.0 s (79%) of the nonzero host CPU** while the 54,720
  carry only 6.98 s.
- Measured against GPU-idle leak (the 20.23 s from the decomposition): the
  **staged stack covers 8.09 s (40%)** — B 6.20, F 0.86, A-v3 0.77, C′ 0.26.
  **Residual 12.14 s is genuinely new territory** (dispatch tax, seam python,
  a2a/cat host blocks, misc).
- My earlier "attack 1 (indexer sync-free top-k, 4–7 s)" was **mis-sited**: the
  leak is not in the top-k kernel path, it's the CP layout builders — whose
  correct fix is B's *cache*, already built. Attack 1 dissolves into lever 3.
- **B/F at d16 pre-registered prediction (mine, from the leak numbers):
  central +4–7% (point +5%), floor +1.5–2.5%, ceiling +9–11%, <+1.5% refutes.**
  Details + comparison to gauss/bohr's bands in §4.

## 1. Q1 — site-level coverage of the nonzero class (and friends)

### 1.1 The 55,328 nonzero calls, by site (trace containment: parent op + CF/CFB phase)

| site class | source site | patch | calls/step | host CPU | idle leaked |
|---|---|---|---:|---:|---:|
| `index_wrapped` fwd — boolean-mask index in the CP layout builders, live forward | `dsa_layout.py:158-161` et al. (`seq_starts[nonzero]` motif = `aten::index → aten::nonzero`) | **FIX B** | 27,360 | 3.34 s | 2.95 s |
| `index_wrapped` bwd — same builders, recompute replay | same | **FIX B** | 27,360 | 3.64 s | 3.25 s |
| `bare` bwd — `torch.nonzero(topk_length > 0)` row compaction | `dsa_cudnn_kernels.py:2130` | **FIX A / A-v3** | 608 | **26.99 s** | 0.77 s |
| **total** | | | **55,328** | **33.97 s** | **6.97 s** |

Call-count arithmetic checks: 27,360 = 38 layers × 16 mb × 45/pass (45 = 5
sites × (cp_size+1) = 5×9 at CP8; Aug-9's 85/pass = 5×17 at CP16 — the builder
cost scales with CP). 608 = 38 × 16, one compaction per layer-backward.

**Asymmetry worth stating loudly:** the numerous class is cheap per call
(~56 µs p50) but leaks ~89% of its host time as GPU idle (6.20/6.98) — it
fires in launch-bound regions where the queue is already shallow. The 608-call
class is enormous per call (~44 ms avg) but leaks only 2.8% (0.77/27.0) — it
fires under deep backward runahead that absorbs the drain. "Absorbed" (erdos's
map) is true of the big drains and false of the small ones.

### 1.2 Adjacent classes (not nonzero, same stack)

| class | source site | patch | calls/step | host CPU | idle leaked |
|---|---|---|---:|---:|---:|
| heavy `aten::to` >1 ms (blocking DtoH), fwd 774 / bwd 836 / none 20 | THD RoPE bookkeeping `rope_utils.py:222-239` | **FIX F** | 1,630 | **37.04 s** | 0.86 s |
| `cudaEventSynchronize` under CF/CFB (dispatcher replay throttle) | token_dispatcher replay `d2h_event.synchronize()` | **FIX C/C′** | 1,120 (560 fwd + 560 bwd) | 13.83 s | 0.26 s |
| `cudaStreamSynchronize` (generic; mostly *inside* the nonzero/to/item calls above — do not double count) | — | — | 64,940 | 31.97 s | (contained) |

### 1.3 Stack totals (union, deduplicated)

| stack subset | idle covered (≥1 ms gaps) | (micro gaps) | total |
|---|---:|---:|---:|
| B (layout cache) | 3.45 | 2.75 | **6.20 s** |
| F (RoPE cache) | 0.66 | 0.20 | **0.86 s** |
| A-v3 (bwd nonempty) | 0.60 | 0.16 | **0.77 s** |
| C′ (replay eventSync) | 0.12 | 0.13 | **0.26 s** |
| **B+F+A-v3+C′ union** | **4.84** | **3.25** | **8.09 s** |
| **residual (not covered)** | 4.97 | 7.17 | **12.14 s** |

Surviving nonzero after B+A-v3 land: ~700 calls/step (first-miss builds per
microbatch layout; Aug-9 measured 26,520→~340 at d4/78L — tonight's d16/38L
first-miss count ≈ 340 × (16/4 mb) × (38/78 layers) ≈ 660), i.e. **~1.3% of
calls survive, carrying <0.1 s host**. The class is dead.

**The stack's hidden prize is host-time removal, not idle removal:** B+F+A-v3+C′
collectively delete ~85 s/step of host drag (6.98 layout + 37.0 RoPE + 27.0
compaction + 13.8 replay-sync). Only 8.09 s of that shows as pure GPU idle
today; the rest rides runahead. That is exactly the F6 "launch-pipeline
decompression" mechanism — and it means the stack's wall win can exceed its
static idle coverage (Aug-9: idle −6.7 s ≈ wall +7.3 s).

## 2. Q2 — is "indexer sync-free top-k" the same work as A-v3?

**No — and it isn't new work either: my attack 1 was mis-sited.** The
decomposition shows the leaked idle does not come from the indexer top-k
kernel path (the `indexer_top_k` cutlass kernels are GPU-side and fine); it
comes from (i) the **CP layout builders** (6.20 s leak) — whose correct fix is
B's *result cache on the per-microbatch carrier*, not sync-free compute — and
(ii) the **bwd row compaction** (0.77 s leak) — which is A-v3's exact site
(`dsa_cudnn_kernels.py:2130`), where A-v3's mechanism (async first-pass
nonempty flag + arange substitution) is the right shape and is already built
(PRs #26/#29, restacked wo-W1). So attack 1 = B (89%) + A-v3 (11%), both
staged. **The genuinely new levers are in the residual (§3).**

A-v3's direct-idle EV today is 0.77 s (0.6%) — consistent with Aug-9's
"≈0, parked" verdict. Its 27 s host class is 97% absorbed at d16 runahead
depth. Dependency chain stands and is *strengthened* by this measurement:
A-v3 stays parked until B/F+C′ land and the absorption regime shifts; then
re-measure (its exposure is a sequela of C′ removing the dispatcher throttle,
per PATCH_NOTES §A-v3).

## 3. Q3 — the 10.41 s diffuse dispatch tax: what dies with the stack?

| component | s/step | dies with |
|---|---:|---|
| micro-gap time overlapping B/F/A-v3/C′ host classes | 3.25 | the stack (mostly B: 2.75) |
| micro-gap time overlapping a2a/cat host blocks | ~1.5 | NEW lever: pre-sized persistent a2a buffers + bwd-cat rework (adjacent to lever-1 contract shim, but this time is pure idle — disjoint from the 34.3 s exposed-SendRecv prize) |
| generic per-kernel dispatch tax | ~4.4 | only kernel-count reduction (fusion) or CUDA-graph-class surgery. **Partial assist from B**: B removes ~150–200k launches/step (54,720 nonzero+index plus follow-on kernels; Aug-9 measured −150k at d4/78L/CP16) ≈ 19% of the step's 1.07 M GPU ops → est. −0.8–1.5 s of the generic tax |
| misc micro (repeat_interleave/cumsum/item-adjacent) | ~1.2 | partially B (the layout-builder arithmetic family rides along) |

Honest total for "what B/F alone does to the 10.41 s": 2.95 s direct (B+F
micro share) + ~1 s launch-count relief ≈ **3.5–4 s**, with the caveat that
launch-count relief is an estimate (gap mean 11.6 µs × launches killed), not a
direct measurement.

## 4. Pre-registered predictions — bohr's B/F d16 A/B (vs 984 tok/s/GPU, 133.2 s step)

Filed BEFORE the A/B runs. Mechanism budget: B+F direct idle coverage 7.06 s
(6.20+0.86) × conversion factor + launch-decompression relief.

- **Central +4–7% (point +5%; step 124.0–127.9 s; 1,018–1,052 tok/s/GPU).**
  Assumes direct-leak conversion 0.7–1.0 (Aug-9's convoy-regime conversion was
  ~1:1 — idle −6.7 s ≈ wall +7.3 s; d16 runahead is deeper, so I shade below
  1) + 1–2.5 s launch relief.
- **Floor +1.5–2.5%** if re-absorption eats most of the direct leak (the F6
  dynamic: host runs ahead, the next sync's wait grows) and only launch-count
  relief lands.
- **Ceiling +9–11%** if conversion is 1:1 AND micro relief compounds.
- **< +1.5% refutes** the leak-causality model (would mean the 7.06 s
  idle-overlap is correlational, not causal) — escalate; the residual map's
  premise needs rework.
- Relation to gauss/bohr's on-record bands (+3–6% central ~+4%, floor
  +0.5–1%, REBUILD_NOTE §5): consistent; my central is ~1 pt higher because
  the leak is measured directly on this trace rather than inferred from the
  builder-window estimate (~7.7 s ≈ 5.7% — close to my 7.06 s).
- **OUTCOME (2026-08-14, W1c):** mechanism verification EXACT (nonzero
  55,328→1,328 vs predicted ~1,330; idle −7.9 s same-box; micro-tax
  9.51→0.89 s via −539k launches = the F6 decompression measured directly).
  Trace-span proxy −8.3% step ≈ +9% tput-equivalent — at the ceiling of this
  band; bench A/B is the throughput authority. See W1C_TRACE_READS.md.
- d4 rung (bohr's primary): no new prediction from me — the trace analyzed
  here is d16; the d16 leak structure (layout builders scale with mb count)
  implies the d4 exposure is smaller in absolute terms but the launch-drag
  share is larger at d4's shorter step. gauss's +0–5% stands.

## 5. EV-ranked corrections/additions to the lever queue

1. **Lever 3 (B/F+C′+A-v3 stack) — revised UP at d16.** Measured static
   coverage 8.09 s pure idle (6.1%) + ~85 s host-drag removal + launch relief
   ≈ EV +4–7% at d16 (pre-registered §4). The CAMPAIGN_ORDERS framing ("at d16
   mostly absorbed — worth more at smaller d") is *wrong for the layout-builder
   class*: it leaks ~89% of its host time at d16. The A/B is the top
   trace-validated idle lever; bohr's rebuild (PR #26 + pointer chain) is
   ready. C′ rides along (small direct EV 0.26 s but removes the dispatcher
   throttle = the runahead-deepener, and is A-v3's prerequisite).
2. **Lever 2 (idle-window attack) — mostly collapses into lever 3** (fermi's
   call confirmed quantitatively: 8.09/20.23 s = 40% of the idle budget is
   staged-stack territory). The genuinely NEW residual levers (12.14 s):
   - **R1. generic dispatch tax ~4.4 s** — fusion / kernel-count reduction /
     CUDA-graph-class surgery. Hard, no single window, but the largest
     untapped line.
   - **R2. seam untraced python 1.9 s** — one py-spy probe at the step seam
     names it (cheap); then async-logging/prefetch/glue fixes. Seam measured
     is a floor (trace ends at the ProfilerStep boundary). **Post-W1c update:
     the seam reads 2.66 s on-arm and is 65% of the residual idle; under the
     favored H1 framing the B/F arm did not ADD seam work — it removed the
     launch backlog that hid a ~constant 2.6–2.7 s python (probe-decidable;
     R2_SEAM_PROBE.md).**
   - **R3. MoE a2a dispatch host block 1.20 s + bwd `aten::cat` 0.71 s** —
     `new_empty`→`cudaEventQuery` 100–160 ms spins; pre-sized persistent a2a
     buffers. Disjoint from lever 1's exposed-SendRecv prize; fix path shares
     the contract-shim territory.
   - **R4. misc in-step syncs ~1.1 s** (aten::item/repeat_interleave/misc not
     stack-covered) — mop-up after R1–R3.
3. **A-v3 — stays parked** (0.77 s direct EV; dependency chain confirmed
   on tonight's trace). Re-measure only after B/F+C′ land.
4. **Latent register (not current idle):** the absorbed host classes (bare-bwd
   nonzero 27 s, RoPE 37 s, replay eventSync 13.8 s, generic streamSync 32 s)
   ride runahead today. Every lever that compresses GPU time shrinks
   absorption capacity — after the stack + lever 1 land, re-run this exact
   decomposition (the scripts are reusable) to catch re-exposure early.

## 7. Rank-8 / stage-asymmetry check (fermi assignment 3)

**PRE-REGISTERED CAVEAT (filed before any l3 query):** the only d16 rank8
trace Mac-side is `l3_d16_rank8.pt.trace.json` (with `l3_d16_rank0` as the
same-run control) — an **OLD-wheel** capture (pre-1.27.0 bump). Per the
campaign's wheel rule, absolute times from it are suspect (the corrupt wheel
changed timing); **site classes and call counts are structural and should
hold**, ratios/leak fractions are indicative not binding. The fe127 d16 rank8
trace does not exist Mac-side. Predictions filed before measurement:

- P1: layout-builder nonzero count on l3 rank8 ≈ rank0 ±5% (54,720 class;
  per-layer builders are stage-symmetric in count — stage 1 has 38 layers too
  under PP2's 38/40-ish split; if stage 1 carries 40 layers, +5%).
- P2: the bare-bwd compaction class (608 on rank0) exists on rank8 at similar
  count (every stage runs backward) but possibly different leak (stage-1
  runahead depth differs).
- P3 (wait direction): the >100 ms SendRecv calls on rank0 sit in bwd phase =
  stage 0 waits on stage 1 (consistent with the Aug-10 30× combine-imbalance
  note — hotter experts on stage 1). If instead they sit in fwd/seam, the
  imbalance runs the other way.
- P4 (haircut verdict, pre-registered decision rule): if rank8's stack-covered
  leak fraction is within ±30% of rank0's 8.09 s-equivalent (normalized per
  step second), NO haircut to the B/F band; if stage 1 leaks materially less,
  the band's central drops by the stage-0-only overcount (B/F win is set by
  the SLOWEST stage — if stage 1 doesn't stall on host syncs, removing
  stage-0 stalls only helps until stage parity).

### 7.1 Results — class decomposition, l3 rank0 vs rank8 (same run, old wheel)

l3 verified **unpatched** (nonzero counts ~55–58k, not the post-B ~700) — valid
as a same-wheel control for class structure. Spans: rank0 138.10 s, rank8
136.32 s.

| class | l3 rank0 (stage 0) | l3 rank8 (stage 1) |
|---|---|---|
| layout-builder nonzero (idx-wrapped) | 54,720 calls · 8.86 s host · **8.07 s leak** | 57,632 calls · 3.79 s host · **2.96 s leak** |
| bare bwd compaction nonzero | 608 · 27.6 s host · 0.72 s leak | 640+32 · 31.7 s host · 0.17 s leak |
| RoPE heavy `aten::to` >1 ms | 1,659 · 38.0 s host · 0.89 s leak | 1,930 · 41.2 s host · 0.83 s leak |
| **total idle** | **22.21 s** (≥1 ms 11.69 / <1 ms 10.53) | **12.74 s** (≥1 ms 1.91 / <1 ms 10.83) |
| SendRecv total | 5,074 calls · 40.2 s · 16× >100 ms | 5,794 calls · 42.7 s · 18× >100 ms |

- P1 (counts symmetric) ✓ — rank8's 57,632 = 40 layers × 16 mb × 45 × 2,
  confirming stage 1 carries **40 layers** vs stage 0's 38 (PP2 split 38/40).
- P2 (bare-bwd exists on rank8) ✓ count, ✗ leak — rank8 absorbs it even
  better (0.17 vs 0.72 s).
- **The layout-builder leak is strongly stage-asymmetric: stage 0 leaks 2.7×
  more (8.07 vs 2.96 s) on the same call count.** Mechanism: stage 1's host
  runs the builders during its PP-recv slack (its GPU wouldn't have work
  anyway); stage 0's host runs them just-in-time against deep compute queues,
  so its stalls starve the GPU. Dispatch-tax density is stage-symmetric
  (<1 ms aggregate: 10.53 vs 10.83 s; ~7.7% of span both).

### 7.2 Results — SendRecv >100 ms structure (wait direction)

fe127 rank0 (new wheel): 10 calls = 8.8 s. Structure: **5,037 ms at +5.9 s
(step start)** and **2,353 ms at +123.2 s (drain)** dominate; mid-step
100–400 ms. l3 rank0: same shape (5,199 ms at +7.7 s; 2,695 ms at +127.9 s).
l3 rank8 (stage 1): 2,048 ms pre-first-fwd at step start, then a **mid-step
cluster of 0.66–1.67 s waits** (+57.1, +65.1, +73.2, +81.6, +98.2 s).

Reading (P3): **bidirectional, asymmetric in kind.**
- **Step boundary: stage 0 waits on stage 1** — the ~5 s start wait (stage 1
  still draining/optimizing the previous step when stage 0 wants to send mb1)
  and the ~2.4–2.7 s drain wait (stage 0 finished, waiting on stage 1's last
  backwards). ≈7.4 s of fe127's 34.3 s exposed SendRecv is this boundary skew.
- **Mid-step: stage 1 waits on stage 0** — stage 1's big recv waits coincide
  (normalized-fraction alignment, ±3 s windows) with clusters of 20–160 ms
  host-stall gaps on stage 0 (366–747 ms per 6 s window at the matching
  fractions). Consistent with the coupling: **stage 0's indexer host stalls
  delay activation delivery → stage 1's recvs wait.** Stage 0's idle is not
  slack; it is the laggard signature. (Inference, not proof: cross-node
  clocks, old-wheel trace, normalized alignment.)
- The 100–190 ms "bwd"-enclosed SendRecvs on both stages are the EP a2a
  combine imbalance (Aug-10's 30× note) — smaller per call, numerous.

### 7.3 Verdict — haircut or not (P4 decision rule)

My pre-registered rule triggers on the raw numbers: rank8's stack-coverable
leak (~3.96 s) is ~59% below rank0's (~9.7 s), outside ±30%. **But the rule's
concern — "stage-0 fixes only help until stage parity" — is inverted by the
measured wait direction:** mid-step, stage 0 is the laggard (stage 1 waits on
*it*), so stage 0's layout-builder leak sits **on the step's critical path**,
and removing it should convert to wall ~directly (and compound through the
reduced stage-1 waits). The boundary skew (stage-1 tail) is a separate,
seam-class territory that my B/F band never claimed.

**VERDICT: NO stage-asymmetry haircut to the B/F d16 band (central +4–7%,
point +5%, stands).** If anything the pipeline coupling makes the stage-0
leak worth more than its local idle. Caveats restated: l3 is old-wheel
(absolute times suspect; class structure and asymmetry direction are the
transferable content); fe127 d16 rank8 does not exist Mac-side — a fixed-wheel
rank8 capture on the B/F A/B box would retire this caveat cheaply
(`BT_PROFILE_RANKS=0,8` already the standing trace discipline).

## 6. Method note (exact queries)

Same gap-union method as IDLE_WINDOW_DECOMPOSITION.md §1.4 (Q1). Site
classification: python-side containment sweep — each `aten::nonzero` slice
classified by nearest enclosing slice on its track: `aten::index` parent ⇒
layout-builder class; bare ⇒ bwd compaction; enclosing
`CheckpointFunction`/`CheckpointFunctionBackward` ⇒ fwd/bwd phase. Idle
overlap per class: exact interval-merge sweep of class slices against the
898,031-gap list (≥1 ms / <1 ms split reported separately). Export + analysis
scripts: `/var/folders/.../opencode/resid_r1..r4.py` (transient); the queries
are reproduced inline in this doc and the companion. Stack-union numbers are
deduplicated (classes are disjoint in time; union == sum within 0.01 s).
