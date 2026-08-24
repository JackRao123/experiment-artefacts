# 262k on PR #1070 vs tip of main — results and 131k↔262k trace comparison

2026-08-23 overnight (box wgm8row, 2×8 B300). Method, bars, and provenance in
README.md / PREREG.md. All bars were pre-registered before any data landed.

## Verdict first

**PR #1070 works at 262k.** It boots (PP2/CP8/EP8, TP1, 2×8), completes every
driver window at d2 and d4, loss and grad-norm canaries sit in family with the
131k rows and the Aug-9 262k anchor (loss 12.30–12.35, gn 0.36–0.45), no OOM,
no NCCL abort, and worst-GPU memory never exceeded 207 GiB of 268.6 —
~61 GiB of headroom even at d4. It is also **faster than tip of main at 262k:
+23% at the 524k-token operating point, +67% at d4**, clearing the
pre-registered +8% bar in both cases.

## Headline table (driver = profile_driver_new.py, control windows only)

| run | code | layout | tok/step | tok/s/GPU | step | mfu3x | peak rank0 / worst GPU |
|---|---|---|---|---|---|---|---|
| A-anchor-262k (2026-08-09, reference) | main @ df-era | PP1/EP16/CP16 | 524k | 630 | 52.0s | 9.0%* | — / 263.7 GiB |
| R1 | main @ 9b039d6b | PP1/EP16/CP16 | 524k (d2) | **312** | 105.1s | 3.3% | 180 / 198 GiB |
| R2 | PR #1070 @ d8b9648f | PP2/CP8/EP8 | 524k (d2) | **383** (+23%) | 85.5s | 4.0% | 187 / 203 GiB |
| R3 | PR #1070 @ d8b9648f | PP2/CP8/EP8 | 1049k (d4) | **522** (+67%) | 125.5s | 5.5% | 192 / 207 GiB |
| (131k ref) PP2 golden | PR-era tree | PP2/CP8/EP8 | 524k (d4) | 918 | 35.7s | — | ~175 GiB |

*The anchor's mfu3x predates the LoRA-corrected FLOPs model — not comparable.

R3 caveat: its two control windows were 596 and 464 tok/s/GPU; warmup and
traced windows sat at 588/581, so three of four windows cluster near ~590 and
the 522 mean is dragged by one slow window. Treat R3 as 522 (conservative,
driver headline) with ~590 plausible; either way it clears every bar.

**Flag — R1 vs the Aug-9 anchor (630 → 312), now mostly resolved from the
recorded bench rows** (`lps1062_bench/exp*.json`, 2026-08-07 opt-night, same
layout and era): the clean-env 262k baseline then was **417–445** tok/s/GPU
(`exp00-baseline`/`exp00b-memprobe`), and the ship-env soak (TF32 head +
NCCL QP/channel knobs) was **629** (`exp06-ship-soak`) — the Aug-9 anchor's
630 matches the ship-env number exactly. So ~2/3 of the 630→312 gap is
environment, not code. The residual clean-vs-clean gap (417–445 then vs 312
on today's tip) remains unexplained — candidates are 15 days of main and any
recompute-policy difference (tonight's runs inherit the config default
`recompute.granularity="full"`; the exp-era notebook says "full recompute
stands" at 256k, so likely the same) — and still needs a one-variable A/B,
which was out of tonight's prereg. The main-vs-PR comparison is unaffected:
R1/R2/R3 share code-default recompute and a clean env by construction.

## 131k ↔ 262k side-by-side (recorded rows, no reruns)

131k rows are the recorded bench JSONs from the box
(`lps1062_bench/*.json`, Aug 12–14 campaign); 262k rows are tonight's.
All 2×8 B300, GLM-5.2-FP8, LoRA r32, driver control windows. "Peak" =
worst-GPU nvidia-smi MiB from the run's poller. Env differs between eras
and is flagged per row — the 131k campaign rows carry the TF32-head knob
(worth ~+7% at 131k-d4: 550→589 pre-M=N), tonight's runs are clean env.

**PR #1070 layout (TP1/PP2/CP8/EP8) — matched by datums (same pipeline M):**

| datums (M) | 131k (mn rows, TF32 env) | 262k (tonight, clean env) | 262k/131k per-token |
|---|---|---|---|
| d2 (M=2) | 764 tok/s/GPU, 173 GiB | 383 tok/s/GPU, 203 GiB | 0.50× |
| d4 (M=4) | **918 tok/s/GPU, 175 GiB** | **522 tok/s/GPU, 207 GiB** | 0.57× |
| d8 (M=8) | 1000 tok/s/GPU, 182 GiB | not run | — |
| d16 (M=16) | 1052 tok/s/GPU, 194 GiB | not run | — |

(The matched-tokens pairing — 131k-d4 vs 262k-d2, both 524,288 tok/step —
reads 918 vs 383, but that comparison is confounded: the 262k side pays an
M=2 bubble the 131k side doesn't. Matched-M is the fair per-token view.)

**Main layout (TP1/PP1/EP16/CP16):**

| row | seq | tok/s/GPU | worst-GPU peak | env / tree |
|---|---|---|---|---|
| A131-131k-d2 | 131k | 649 | 229 GiB | campaign canonical (TF32+NCCL), Aug-13 tree |
| A131-131k-d4 (anchor family 616–630) | 131k | 620 | 197 GiB | same |
| exp00-baseline / exp00b | 262k | 417 / 445 | 242 / 257 GiB | clean env, Aug-7 tree |
| exp06-ship-soak | 262k | 629 | 258 GiB | ship env (TF32+NCCL), Aug-7 tree |
| A-anchor-262k (Aug-9) | 262k | 630 | 258 GiB | matches ship-env number |
| **R1 (tonight)** | 262k | **312** | **198 GiB** | clean env, main tip 9b039d6b |

Memory reading: on the PR layout, 131k→262k costs ~+30 GiB of worst-GPU peak
at matched M (173–194 → 203–207 GiB) — activation growth under full
recompute — leaving ~62 GiB headroom at d4. Cross-era memory on the main
layout is noisier (different trees/buffers): the Aug-era 262k rows sat at
242–258 GiB (≤12 GiB headroom — the reason lighter recompute was ruled
memory-infeasible then), while tonight's tip peaks at 198 GiB, ~59 GiB leaner
on the same layout; the mechanism behind that shift is part of the same
open A/B as the throughput residual above.

Throughput reading: at matched M the PR layout keeps 50–57% of its 131k
per-token throughput at 262k (the wait growth quantified below). The main
layout, in contrast, is nearly context-flat per token in comparable env
(649 @131k-d2 → 630 @262k, both TF32+NCCL-era rows) — coherent with its
bottleneck: EP a2a bytes per token don't depend on context length, and
cross-node a2a is already its ceiling at 131k. Flat-but-low loses to
faster-but-decaying at both contexts measured: the PR layout beats the main
layout like-for-like at 131k (918 vs ~620 at d4) and at 262k (522 vs 312
clean-env; the anchor's 630 needed the ship env, whose knobs are additive
candidates on top of #1070, not alternatives to it).

## Trace comparison, 131k ↔ 262k (rank-0 kineto, traced window = one step)

Reference 131k trace: `~/perf_profiles/lps-1062/pp2cp8ep8/mn_d4_rank0`
(PP2/CP8/EP8, d4, post-M=N-fix — same layout and M as R3). All numbers are
GPU-kernel time on rank 0 (node 0, PP stage 0); NCCL in-kernel time includes
peer-wait, and the tables below use that fact deliberately.
Profiles on this tree are rank-0-only (ProfilingConfig.rank_set={0}), so
stage-1/node-1 skew is not directly measurable — that limitation is noted
where it matters.

### Same layout, same M (PP2/CP8/EP8 d4): 131k vs 262k (R3)

Normalized to a 524k-token half-step for the 262k column so rows compare
per-token cost at 2× context:

| component | 131k d4 (36.3s step) | 262k d4 (per 524k tok) | growth |
|---|---|---|---|
| PP p2p wait (`nccl:coalesced`) | 8.8s | 14.3s | 1.6× |
| EP a2a (intra-node) | 7.2s | 11.1s | **1.54×** |
| non-NCCL compute (all kernels) | 17.8s | 22.0s | 1.24× |
| — of which DSA fwd+bwd+indexer | ~4.2s | ~4.4s | ~1.05× |
| — of which GEMM | 5.3s | ~4.9s | ~0.9× |
| CP KV allgather | 0.23s | 0.58s | 2.5× (tiny) |
| control step per 524k tok | 35.7s | 62.8s | **1.76×** |

Reading: at 2× context on the same layout, per-token cost grows 1.76×, and
**essentially none of it is compute** — GEMMs are token-count-bound (flat),
and DSA is top-2048 sparse so attention stays ~flat instead of the O(s²)
blowup dense attention would give. The growth is (1) pipeline wait and
(2) all-to-all residency:

- **EP a2a 1.54× per token.** The per-call p50 is bandwidth-flat (4.5ms at
  16k-token chunks → 7.5ms at 32k chunks, sublinear per byte). The inflation
  lives in the tail: p90 11.7ms → 29ms. The a2a kernels are absorbing more
  rank-arrival skew per call at the bigger chunk size, not moving bytes
  slower. (Same conclusion the 131k campaign reached: the collective itself
  is fine; the wait is the cost.)
- **PP p2p wait 1.6× per token at matched M.** Absolute p2p wait is ~flat
  (28.6s at d4-262k vs 28.7s at d2-262k — see below) but each 262k
  microbatch takes ~2× longer, so at matched M the bubble costs ~2× per
  step, ~1.6× per token after the longer step amortizes fill/drain edges.

### The d2→d4 lever at 262k (R2 vs R3, PR layout)

| component | d2 (M=2) | d4 (M=4) |
|---|---|---|
| PP p2p wait, absolute | 28.7s | 28.6s |
| PP p2p wait, share of step | ~33% | ~14% (per 524k: 14.3s) |
| EP a2a per 524k tok | 25.4s | 11.1s |
| EP a2a p99 per call | 1.26s | 0.34s |
| tok/s/GPU | 383 | 522 (~590 in tight windows) |

The M=2 bubble matched theory almost exactly ((PP-1)/(M+PP-1) = 1/3 of an
85.5s step ≈ 28.5s predicted, 28.7s measured), and the d2 a2a "inflation"
mostly collapsed at d4 — those were bubble-coupled waits (stage-local ranks
arriving at collectives out of phase around pipeline stalls), not bandwidth.
Same conclusion as the 131k campaign: **datums are the bubble amortizer;
d2 is the worst legitimate operating point for PP2 at 262k.**

### Ranked bottlenecks per config

**131k PP2/CP8/EP8 d4 (the shipped PR row, 35.7s step, 918 tok/s/GPU):**
1. PP p2p wait 8.8s (~24%) — fill/drain + stage skew wearing SendRecv.
2. EP a2a 7.2s (~20%) — intra-node, tail-skew dominated.
3. CPU-blocked host syncs ~16s CPU-side (deviceSync 8.8 + memcpy 8.2 +
   streamSync 7.4), partially shadowed by GPU work; known ~21s exposed on
   rank 8 from the campaign's rank-8 trace.
4. GEMM 5.3s, DSA ~4.2s (healthy floor).

**262k PP2/CP8/EP8 d4 (R3, the PR at 262k, 125.5s step, 522 tok/s/GPU):**
1. PP p2p wait 28.6s (~23%) — same share as 131k; scales with microbatch
   duration. Levers: more datums (d8+), VPP, or the overlap roadmap.
2. EP a2a 22.2s (~18%) — per-token 1.54× the 131k cost, all in arrival-skew
   tail, not bandwidth. Lever: the a2a-overlap / balance roadmap items.
3. Non-NCCL compute 43.9s (~35% — the useful floor; DSA+indexer ~8.8s,
   GEMM ~9.8s, rest elementwise/permute/recompute overhead from
   full-recompute default).
4. Step-end coalesced allreduce 3.8s single call (straggler absorption;
   0.5ms at 131k — worth an eye, likely embedding-grad/telemetry skew).
5. GPU idle-by-sum 16% — CPU-blocked structure (streamSync 35.9s +
   deviceSync 28.9s host-side per step at d4).

**262k main tip EP16/CP16 d2 (R1, 105.1s step, 312 tok/s/GPU):**
1. **EP a2a 55.0s = 54% of the step** — EP16 spans both nodes, so every
   MoE dispatch/combine crosses the inter-node fabric; p50 36ms/call
   (7× the intra-node p50 at the same chunk size), plus tail to 1.19s.
   This is the structural cost the PR's PP-outermost thesis removes: with
   PP2/EP8, expert traffic stays on NVLink and only PP crosses nodes.
2. True serialization ~18s (17.7% idle-by-sum; host streamSync 70s/step
   CPU-side — the dispatch path is sync-heavy at CP16).
3. Compute 28.5s (GEMM 6.8, DSA 6.6, indexer 1.6 — indexer/DSA re-run
   under full recompute; plus 1.9s of f32 SIMT GEMMs at 117ms/call — 8
   calls, non-tensor-core, worth identifying).
4. f32 TREE AllReduces 2.9s (max 1.8s) + CP16 KV allgather 2.2s.

### What 262k changes vs 131k, in one paragraph

Compute is not the story — DSA's fixed top-2048 window keeps attention
essentially flat per token, and GEMMs don't care about context. What doubles
context actually buys is *wait*: every communication edge (pipeline p2p,
all-to-all arrival, step-end reductions) absorbs more skew because each
microbatch unit of work is twice as long, and on main's EP16/CP16 layout the
a2a additionally pays the inter-node fabric on every MoE layer. The PR's
layout (PP outermost, EP/CP intra-node) is the right shape at 262k for the
same reason it was at 131k, and its remaining costs (bubble ~23%, a2a skew
~18%) are exactly the two levers already on the LPS-1062 roadmap (datums/VPP
for the bubble, overlap/balance for the a2a).

## Caveats
- Rank-0 traces only (main has no BT_PROFILE_RANKS); stage-1 skew unseen.
- Both 262k runs use main's default full recompute; 131k reference trace's
  tree predates tonight's checkout (PR-era, same layout/M at d4).
- Kineto adds overhead (−3.5% to +23% traced-vs-control tonight), so shares
  are computed within the traced window, headlines from untraced controls.
- Parity/determinism not re-tested at 262k; the noise-relative rule from the
  131k campaign presumably applies. Save/export paths untested at 262k.
- R3's 464-tok/s control window is unexplained (one of four windows);
  optim_step took 1.1s there vs 0.1s typical — possibly a background flush.

## Artifact map
- Driver JSONs/runlogs/trainer logs: this folder (R1/R2/R3 prefixes).
- Traces + memory pickles: `traces/` (sha256-verified after transfer).
- Per-GPU nvidia-smi peaks: `R{1,2,3}_mem_max_node{0,1}.txt`.
- Box copies: `tj-wgm8row:/root/.cache/user_artifacts/lps1062_262k/`.
- Query battery + raw outputs: session scratchpad `tq/` (q1–q11).
