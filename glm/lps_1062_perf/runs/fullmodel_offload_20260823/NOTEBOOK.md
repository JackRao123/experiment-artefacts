# Full-model offload overnight (2026-08-23)

Jack's orders (~00:30 PT): run the FULL GLM-5.2 on the 16-GPU devbox
`wgm8row`; goal = `offload` within **5%** of `baseline` on the full model;
also benchmark vs the previous PP2/CP8/EP8 numbers; record kineto profiles.
Shut down all other devboxes (DONE: `wdpok4w` and stale `q0g9z43` stopped via
`truss train stop`; only wgm8row lives).

Jack's thesis under test: the proxy's forward was too fast to hide copies
because 1 GPU has no all-to-all; at EP8/CP8 the a2a stretches per-layer wall
time while per-rank copy volume shrinks (EP8 splits the moe payload), so the
copies should hide. The proxy mission (see
`../debug_proxy_20260821/results/overlap_trace_20260822/MISSION_WITHIN2PCT.md`)
closed at −17.6% with the forward host-sync × saturated-copy-engine collision
as the proven floor; this run tests whether that collision dissolves at
production topology.

## Arm definitions (Jack's standing names)

- `baseline` = no recompute, no offload (`recompute: selective, modules: []`)
- `offload` = baseline + 4-group activation offload (core_attn, moe_act,
  qkv_linear, attn_proj; fraction 1.0; **max_inflight_offloads 4** — the
  arm-3b uncapped-valve OOM lesson) + env `BT_OFFLOAD_PREFETCH_DEPTH=6`,
  `BT_OFFLOAD_H2D_UNCHAINED=1` (the proxy-validated fix, incl. the
  backward-entry prefetch in schedules.py)
- `fullrecompute` = full/uniform recompute (production's current setting)

## Setup

- Box: wgm8row, 2×8 B300 (275 GiB/GPU). Tree: SHARED-FS worktree
  `devboxes/qkpox9w/trainers` = lps1062-scale `c226338a` + the 3-file patch
  (`bt_offload_backward_prefetch.patch`, in
  `../debug_proxy_20260821/results/overlap_trace_20260822/`). Its shared
  `.venv` works cross-box. Model: `zai-org/GLM-5.2-FP8` (shared HF cache).
- Topology: TP1/**PP2/EP8/CP8**, LoRA r32, d4 datums/window (matches the
  918@d4 record shape). ES **on** for all arms (memory-tight; matched pairs;
  the proxy's ES-tail finding noted as a caveat, not applied here).
- Launch overlay: box `/root/.cache/user_artifacts/lps1062_full/`
  (`start_trainer_full.sh` → `run_trainer_node_full.sh` with SRC→patched
  tree, NVTE latch, knob defaults pinned). Driver:
  `lps1062_262k/profile_driver_new.py` (warmup → traced → N controls;
  headline = control windows). Health waits per the box's embedded rule:
  its `wait_trainer_health.sh`, re-run each 180 s checkpoint with log reads.
- Memory envelope (why there is no baseline_131k arm): no-recompute at 131k
  needs ~2.94 GiB × 70 sets ≈ +206 GiB on rank 0 → ~358 GiB, over the
  275 GiB card. At 32k it's ~+51 GiB → fits; 64k marginal. So the within-5%
  verdict comes from 32k (and 64k if time allows); 131k runs offload vs
  fullrecompute only.

## Boot/arm log (append per arm)

| # | arm | boot | windows | result |
|---|---|---|---|---|
| 1 | fullrec_32k d4 | HEALTH_UP in ~12 min (4th checkpoint) — **patched tree validated at PP2/16-GPU multinode** | warmup 449; traced 545 (+6.6% kineto); controls 582/580/578/584 | **581 tok/s/GPU**, step 14.1 s (131,072 tok), peak 110 GiB, loss 12.29-12.30 gn 0.34-0.45 (matches the 262k canary band) |
| 2 | baseline_32k d4 | HEALTH_UP round 5 (~15 min incl. no-recompute warmup) — **fits: peak 228 GiB** (envelope said ~201; 47 GiB headroom) | pass 1 controls 731/740/526/647 (noisy — one −29% outlier window); pass 2 with 8 controls run on the same boot | pass 1 aggregate **649 tok/s/GPU**, step 12.6 s; vs fullrec 581 → recompute costs ~10-15% at this shape |
| 2b | baseline_32k pass 2 (same boot, 8 controls) | — | 724/707/716/707/706/685/702/703 — clean | **baseline reference = 706 tok/s/GPU median** (step 11.6 s); within-5% bar for offload = **≥671** |
| 3 | offload_32k d4 (K6+unchained, valve 4) | HEALTH_UP round 7 (~21 min — one-time pinned-pool first-touch; NUMA placement verified correct per GPU; NVTE latch OK both nodes) | warmup 93 (pool build); controls 367/382/289/471/522/471/418/477 — warming across windows, steady tail ~470-520 | aggregate **411 tok/s/GPU**, step 19.9 s; **−33 to −42% vs baseline 706 — far off the 5% bar.** Peak 233 GiB ≈ baseline's 228: no worst-GPU saving (suspect last-stage/LM-head binds peak on both arms, or valve/prefetch re-residency). +8.3 s/step ≈ envelope one-way copy time (35 layers × ~1.4 GiB × d4 ÷ 28 GB/s ≈ 6.9 s) → roughly half the bidirectional copy volume is exposed. Jack's a2a-hiding thesis NOT confirmed at 32k/d4. Next: trace analysis before any lever (no-guessing rule). |
| A | offload_32k, valve 16 (knobs on) | healthy (one slurm detour: fast rotation left node drained "Kill task failed" → scontrol resume; queue-aware watcher added) | controls 347/423/335/420/494/472/459/473 | aggregate 420, tail ~460-495 — **statistically identical to valve 4 (411, ~470-520): VALVE EXONERATED** — its ~664 drain-waits/step complete too fast to cost wall time at 10% engine duty |
| B | offload_32k, valve 16, **knobs OFF** (legacy scheduling) + telemetry | healthy round 7 | controls 360/410/273/481/481; aggregate 384 (telemetry-tainted, same regime) | **misses persist under legacy: 12.8%, again perfectly uniform across every group** (attn_proj 76/594, core_attn 304/2376, moe_act 175/1365, qkv 570/4455). My knobs worsen the rate (→20.4%) but are NOT the cause. **Verdict: pre-existing PP2/d4 defect in the machinery's backward-order prediction — whole chunks mispredict; ~150-200 misses/step/rank, each a main/autograd-thread full stream sync (~1.8 s/step measured on stage 1).** |
| 4 | offload_131k d4 | SKIPPED deliberately — pointless until the PP2 backward-order miss bug is fixed | — | — |
| 5 | fullrec_131k d4 | healthy round 5 | controls 792/791/792/787 (ultra-tight) | **790 tok/s/GPU**, step 41.5 s (524,288 tok), peak 145 GiB. = 86% of the 918@d4 record on the campaign tree+ship env; coherent — this branch lacks TF32 LM-head + dispatcher caches (+11-13%), so same-hardware ratio checks out (790×1.13≈893). Box left clean (trainer stopped, GPUs 0 MiB). |

## Trace findings, offload_32k rank-0 (traced window; structure valid, absolutes kineto-inflated)

1. **Offload copies are NOT the direct cost.** Rank-0 D2H = 1,095 copies /
   0.97 s busy; H2D = 1,095 / 1.01 s — ~1 s per direction against a 19.9 s
   control step (~10% engine duty). Zero demand misses (no H2D on the main
   stream). My pre-run copy-volume envelope (~6.9 s) was wrong — CP8/EP8
   shrink per-rank payloads harder than estimated.
2. **The dominant loss is pipeline peer wait:** rank 0 idles in
   `ncclDevKernel_SendRecv` + `cudaDeviceSynchronize` holes of 5.26 s /
   2.0 s / 1.5 s (baseline: same 11 waits total 2.25 s; offload: 6.81 s).
   **Stage 1 (other node: 40 MoE layers + LM head) is the slow side under
   offload, and rank-0-only profiling can't see it.**
3. Secondary rank-0 signals: `cudaMemcpyAsync` issuance 0.27 s → 5.50 s
   (~10k small dispatcher pinned readbacks + 1.1k offload copies; per-call
   issue 27 µs → 494 µs — enqueue contention, kineto-inflated but
   differential), dispatcher `cudaEventSynchronize` 140 calls 0.70 → 1.98 s
   (the proxy's readback class at scale), valve waits real but minor
   (43/181 main-stream gaps end at D2H completions).
4. Next: diagnosis boot with `BT_PROFILE_RANKS=0,8` (supported on this
   tree — models/src/loops_models/profiling.py) + valve telemetry, to see
   stage 1 directly and read drain_firings per rank.

## Diagnosis boot (BT_PROFILE_RANKS=0,8 + valve telemetry, same offload config)

Windows (telemetry-tainted, same regime as arm 3): warmup 105, traced 283,
controls 324/452 tok/s/GPU. Telemetry counters (cumulative, iter=6,
per-rank; identical on both stage leaders):

- **Valve cap 4 fires on ~90% of commits**: per group-name commits 171-186,
  drain_firings 151-166, max_pending pinned at the cap. The main stream
  blocks on a D2H completion event at nearly every group commit —
  "an undersized valve costs throughput invisibly" (campaign memory),
  realized at ~664 drain waits/step/rank.
- **Demand misses at exactly 20.4% for every group** (attn_proj 80/392,
  core_attn 320/1568, moe_act 200/980, qkv 600/2940 — uniform = structural,
  ~200 misses/step/rank, each a reload on the main stream + a full stream
  synchronize). Suspect: the backward-order prediction (and possibly my E7
  backward-entry/top-up prefetch, validated only single-chunk) targets the
  wrong in-flight chunk under PP2/d4 1F1B.
- Rank-0 + rank-8 traces captured to
  `lps1062_full/traces/diag-offload-32k-d4/` (both nodes) for stage-1
  analysis.

Night plan from here (one variable per boot): Boot A = valve 16, knobs on
(valve lever). Boot B = valve 16, knobs off + telemetry (does the miss rate
vanish with legacy 1-deep prefetch → E7 misfires at PP2).

**Rank-8 (stage 1) trace, diagnosis boot** (structure; absolutes tainted):
critical-thread host blocking ≈ 12.7 s of a ~29 s traced step —
`cudaMemcpyAsync` issuance 11,779 calls / 7.24 s (worse than rank 0's
5.5 s), demand-miss `cudaStreamSynchronize` on the **autograd thread**
168 calls / 1.83 s (the misses, live on backward), dispatcher
`cudaEventSynchronize` 160 calls (= 40 layers × d4) / 0.99 s, and
`cudaHostAlloc` 15 / 0.82 s (pinned pool still growing mid-run — shape
churn, the proxy's forward-stall class). Both stages exchange p2p waits
(deviceSync ~7 s each side of the traced window). The offload copies
themselves: ~1.05-1.1 s per direction — still not the direct cost.

Ops note: rotating boots too fast leaves srun PENDING behind the old job's
epilogue and the box wait-script misreads it as TRAINER DIED — wait for
`squeue` to empty before redispatch (queue-aware watcher added).

## MORNING CONCLUSIONS (2026-08-23, as of ~06:30 PT)

**Goal status: offload within 5% of baseline on the full model — NOT
reached, and the blocker is identified, mechanism-proven, and different
from the proxy's.**

Three-way at 32k/d4 (PP2/EP8/CP8, full GLM-5.2, matched tree/env):

| arm | tok/s/GPU | peak GiB |
|---|---:|---:|
| baseline (no recompute) | **706** (8 clean controls) | 228 |
| fullrecompute | 581 | 110 |
| offload (any valve/knob combo tried) | 384-420 aggregate, ~470-520 best windows | 232-233 |

Findings chain (each step measured, per the no-guessing rule):

1. Copies are innocent: ~1 s per direction against ~20 s steps (~10%
   engine duty), zero misses attributable to bandwidth. Jack's a2a-hiding
   premise is CORRECT for raw copy volume — CP8/EP8 shrink per-rank
   payloads so far that hiding them is trivial. What's expensive is not
   the copying; it's the machinery around it.
2. Valve exonerated by A/B: cap 4 → 16 changed nothing (411→420 agg
   within noise), despite firing on ~90% of commits.
3. **Demand misses are the primary offload-specific cost: uniform-rate
   misprediction of the backward reload order at PP2/d4** — 12.8% legacy
   / 20.4% with my prefetch knobs, whole-chunk-shaped, ~150-200 misses per
   step per rank, each a full stream synchronize on the consuming thread
   (~1.8 s/step measured on stage 1), compounding into 5.3 s/2.0 s/1.5 s
   pipeline peer-wait holes.
4. Secondary: `cudaMemcpyAsync` issuance inflation on the critical thread
   (0.27→5.5-7.2 s traced) and the dispatcher `cudaEventSynchronize`
   class (140/160 calls = layers×d4 exactly), plus mid-run pinned-pool
   growth (`cudaHostAlloc` 15×/0.8 s on stage 1 — shape churn).
5. **Offload peak (232) ≈ baseline peak (228): no worst-GPU memory saving
   at this config** — unexplained in detail (suspect stage-1/LM-head
   binding + valve/prefetch re-residency); memory pickles captured for
   follow-up.

**What would make offload viable at PP2** (in order): fix the
backward-order bookkeeping for interleaved 1F1B chunks (the miss storm is
a correctness-of-schedule bug, not a tuning knob — misses are silent
correctness-preserving but throughput-fatal); then re-measure; then the
memcpy-issuance and dispatcher-sync classes (shared with the proxy
findings) are next. Until the order fix lands, offload on PP2 topologies
is not competitive: fullrecompute beats it by ~40% at 32k while saving
MORE memory (110 vs 232 GiB).

My 3-file patch: knobs behaved as designed (backward-entry auto-off at
depth 0; no regression vs legacy in B), but top-up prefetch AMPLIFIES the
pre-existing misprediction at PP2 (20.4% vs 12.8%) — recommendation:
gate `on_backward_entry`/`top_up_reloads` to single-chunk (PP1) pending
the order fix.

## WEIL MISSION (2026-08-23, handoff from bernoulli): fix the PP2 backward-order miss bug

Goal: offload within 5% of baseline 706 tok/s/GPU @32k/d4 full model. Iteration
vehicle = PP2-on-2-GPU debug proxy. Working dir on box: `lps1062_pp2fix/`.

- Boot R1a (pp2-repro-legacy-v1, glm52-debug-0d4m, PP2/EP1/CP1, 16k, d4,
  legacy knobs, telemetry): **BOOT FAILED — model constraint, not infra.**
  PP2 engaged ("initialized pipeline model parallel with size 2") but 0d4m has
  DSA `index_topk_freq: 4` (only layer 0 computes top-k) and cross-layer top-k
  sharing cannot cross the PP boundary at the 2+2 split (validator
  `_validate_dsa_index_share_pipeline_split`). The prebuilt
  `glm52-debug-0d2m-pp2` snapshot solves it with `index_topk_freq: 1`.
- Built `glm52-debug-0d4m-pp2` = copy of 0d4m with `index_topk_freq: 1`
  (each layer computes its own top-k; same 4 layers / shapes; 2 layers/stage).
- Instrumented module prepared (logging-only, env `BT_OFFLOAD_MISS_DEBUG=1`):
  ordered event stream `BT_PP2FIX r<rank> i<iter> s<seq>` with events
  STARTB/COMMIT/ISSUE/PREGATE/MISS/ENTRY, chunk ids = forward creation order.
  Purpose: localize which chunk/position misses and whether missed groups
  poison later reload slots (code-reading suspect: a demand-missed group stays
  in `_groups_to_reload`, so each subsequent start-node wastes its slot
  re-issuing an already-consumed empty group — a whole-chunk cascade;
  hypothesis, to be confirmed by the event stream, not assumed).
- Observation from existing telemetry (diagnosis boot): miss counts
  80/320/200/600 are exact multiples of per-group tensor-count ratios
  2:8:5:15 → misses are whole-GROUP-shaped (~38-40 group instances per name
  per window), not necessarily whole-chunk.

### R1 (pp2-repro-legacy-v2): PP2 proxy does NOT reproduce steady misses — but exposes the regime

Boot healthy (world 2, PP2), 6608 tok/s/GPU @16k/d4, loss 11.9509, gn 2.8e-4,
telemetry per-iteration. Findings (all read off cumulative counters):

1. **ONE cached chunk handler only.** Commits grow exactly +1 per group-name
   per iteration (= 1 offloaded layer-group per name after margin-4 on a
   2-layer stage). Cause: chunk handlers are created ONLY while the manager
   `_is_warmup`, and warmup = the trainer's internal startup fb which uses
   **one datum** (backend.py run_startup_warmup, `data=[datum]`); manager
   reset() runs at the END of each forward_backward (schedules.py ~839/2080)
   so post_warmup fires right after the internal fb. **In every d4 window,
   microbatches 2-4 are completely unmanaged (no offload at all).** On the
   full model this is the likely explanation for "offload saves no memory"
   (232 vs 228 GiB) — only ~1/4 of activations were ever offloaded.
2. **All demand misses live in the warmup iteration itself** (4/14/10/30 =
   exactly 2 groups/name × per-group tensor counts 2/7/5/15): warmup-mode
   reload picks `_groups_to_reload[-1]` = the just-consumed group → the whole
   warmup chunk demand-misses by design (one-time). **Zero steady misses.**
3. A miss is set purely by autograd-thread bookkeeping order (state mutation
   is synchronous; H2D is async but the tuple→tensor swap happens at issue
   time), so the full model's steady 12.8% is a DETERMINISTIC order
   misprediction, not a timing race.
4. **Full-model overlay never sets BT_WARMUP_SEQ → full-model boots warmed
   up at seq=64, 1 microbatch** while steady = 32k/d4 (proxy runner set
   warmup=steady, which is why the proxy is clean).

### R3 launched (pp2-warm64-v1): one-variable test — BT_WARMUP_SEQ=64 vs
16384 steady, instrumented module (BT_OFFLOAD_MISS_DEBUG event stream)
deployed to the qkpox9w tree (original backed up as .bak_preweil, md5
74b3e0f5...). If steady misses appear → full-model mechanism reproduced and
localized by the event stream in the same boot.

### R3 (pp2-warm64-v1): warmup-seq mismatch REFUTED as the miss mechanism (at CP1/EP1)

BT_WARMUP_SEQ=64 vs steady 16k, otherwise identical to R1: miss counters
IDENTICAL to R1 (4/14/10/30 frozen after warmup; zero steady). Event stream
(instrument validated): iter 1 = the by-design warmup cascade on chunk 0
(58 misses + 7-8 EMPTY reload issues — the wasted-slot cascade predicted from
code reading, observed directly); iters 2+ = only chunk 0 exists, 26-28
events/iter, zero misses, PREGATE always front=None. Conclusions:
- The 1-chunk steady chain is clean on the proxy even with a warmup-graph
  seq mismatch → the full model's steady 12.8% needs EP8/CP8/layer-count.
- The 1-chunk regime itself (coverage defect) is confirmed twice over.

### FIX-W deployed (backend.py, Baseten code — not vendored Megatron)

`run_startup_warmup` now resets the PipelineOffloadManager singleton after
the internal 1-datum warmup fb (only when the manager was engaged), so the
FIRST REAL forward_backward becomes the machinery's warmup: chunk roster =
real microbatch count, backward order recorded on the real-seq graph, byte
stats real. Baseline/fullrec arms unaffected (manager never engaged → no-op).
Box tree edited via scp, backup `backend.py.bak_preweil`. Known residual
limitation (documented, not fixed here): a later window with MORE datums than
the first real window would still under-manage; durable fix would be
on-demand chunk creation.

### R4 (pp2-fixw-v1): FIX-W VALIDATED on the proxy — multi-chunk steady state clean

One variable vs R3 (FIX-W on). Event stream: first real window = manager
warmup with ALL 4 chunks created (chunk0-3 all present; whole-window miss
cascade by design, 290 misses incl. the old instance's internal-warmup 58).
**Steady iterations: 0 misses, 0 empty issues on BOTH stages.** Stage 0
(rank 0): PREGATE fires=True at every chunk boundary — the cross-chunk
pre-reload chain works; last stage (rank 1): PREGATE fires=False everywhere
(next chunk's forward hasn't run — by 1F1B) and chunks self-start from their
dynamically-kept resident top-layer groups. Loss 11.950888 (matches R1/R3 to
~1e-6). Peak alloc 82.3 → 76.7 GiB (saving materializes). Throughput 5985 vs
6608 (R1) tok/s/GPU — expected, now offloading 4-6× the volume (commits
4-6/name/iter vs 1). Proxy acceptance (ladder #1) met.

### F1 launched (full-offload-32k-d4-fixw): full model 32k/d4, FIX-W +
arm-3 knobs (K6+unchained, valve 4, ES on), telemetry EVERY=1, BT_PP2FIX
instrument on ranks 0,8, profile_driver_new --control-repeats 8 (same
invocation as arm 3's 411 for direct comparison; bar = ≥671, peak should
drop well below 233 GiB with 4× offload coverage). Dispatcher
`lps1062_pp2fix/run_full_arm.sh` (bounded health rounds, queue-aware).

### F1 OUTCOME: BOX KILLED BY HOST OOM — second latent defect exposed (pinned-pool blowup)

Timeline (UTC, from Grafana/Loki on ali-apse7-prod-1): trainer healthy 18:08;
driver started (memory profile + first real window = manager warmup under
FIX-W); node-0 container working set ramped 545 GiB (18:18) → 1003 (18:19:45)
→ 2007 (18:21:30) → **2641 GiB plateau** (18:23-18:27); node-1 same shape to
2593 GiB; pods died ~18:31 ("Exceeded 1 retry limit with 1 attempts,
terminating job"); job → TRAINING_JOB_FAILED. My earlier ssh drops = the box
dying, plus a stale ControlMaster socket masking it.

Mechanism (measured ramp + code): warmup-mode offload copies EVERYTHING
(margin/fraction/dynamic-keep don't exist until post-warmup; min_tensor_size
0), through the pinned pool whose keys pad dim-0 up to
BT_OFFLOAD_POOL_ROW_BUCKET=8192 rows — at 32k/CP8 local seq = 4096 rows, so
EVERY activation inflates ≥2× (small tensors 10-100×), and buffers free only
at reload. Result ≈ in-flight chunks × padded full coverage ≈ 2.6 TiB/node >
host budget. The old regime survived only by managing 1 seq-64-shaped chunk;
its observed mid-run cudaHostAlloc growth was this same failure in slow
motion. NOTE: this is not warmup-only — steady multi-chunk coverage has the
same padded-pool demand; it must be engineered down regardless.

Levers (in order): BT_OFFLOAD_POOL_ROW_BUCKET 8192→1024 (payload ~0.4-0.8
TB/node payload ×~1.1 padding → fits); pool-bytes telemetry (add to valve
dump — never fly blind into host OOM again); min_tensor_size 0→1MiB variant
if needed; fraction as last resort. 131k full-coverage payload (~1.7-3.3
TB/node) does NOT fit at any padding — 131k will need fraction/module
selection regardless (was true before this mission).

Box: recreating via devbox-up 16 b300 ali (all state on shared FS: qkpox9w
patched tree incl. FIX-W + instrument, lps1062_full overlay, lps1062_pp2fix,
glm52-debug-0d4m-pp2 snapshot — nothing lost).

### New box: tj-q8yz15w (2×8 B300, ali; replaces dead wgm8row; one old node
b300-1-5abzeeir-0002 reused + b300-1-ana8db87-0004). All shared-FS state
verified intact (qkpox9w tree with FIX-W, overlays, snapshots). Long-running
box work now dispatched via nohup on the box (ssh-drop immunity — the relay
drop during F1 was the box dying, but the pattern was fragile regardless).

### R5 (pp2-fixw-pool-v1, new box): pool guard + bucket 1024 validated

Proxy behavior identical to R4 (misses = warmup-only 16/56/40/120, frozen;
commits 24/36; peak_alloc 76.7 GiB; 5637 tok/s/GPU — different node pair).
New pool telemetry line works: stage ranks 11.5 / 22.9 GiB, 18 keys.
Module now also carries BT_OFFLOAD_POOL_MAX_GB (loud RuntimeError instead of
host OOM; runners pass 200 GiB/rank) and BT_OFFLOAD_POOL_ROW_BUCKET=1024 in
both runners.

### F1b launched (full-offload-32k-d4-fixw2): full model 32k/d4, FIX-W +
K6/unchained + valve 4 + bucket 1024 + pool cap 200 + telemetry + instrument
ranks 0,8; profile_driver_new --control-repeats 8. Watch: misses ≈ 0 steady,
pool GiB/rank, peak drop vs 233, tok/s vs bar 671.

### F1b ops detour (new-box gotcha): slurm lead ≠ ssh login node

run_trainer_node_full.sh picks rank-0 = scontrol-sorted first host. On
q8yz15w the sorted first is the REUSED old node b300-1-5abzeeir-0002
(= alias tj-q8yz15w-1), while ssh login lands on b300-1-ana8db87-0004 — so
127.0.0.1:8001 health checks and drivers on the login node see nothing, and
the dispatcher hit BOOT_TIMEOUT while the trainer was actually healthy.
Salvaged: resume_full_arm.sh nohup'd ON the lead (tj-q8yz15w-1); trainer
boot reused (uvicorn was up at ~21 min). RULE for this box: run health waits
+ drivers on the scontrol-sorted lead node. (On wgm8row login==lead by
coincidence.) Watch cadence now 10-min bounded rounds per Jack's order.

### F1b RESULTS (full-offload-32k-d4-fixw2): MISS BUG FIXED AT FULL TOPOLOGY; pool churn is the remaining blocker

**1. The backward reload-order miss storm is FIXED on the full model.**
Event stream (ranks 0 and 8, PP2/EP8/CP8, 32k/d4): iter 1 = manager warmup,
all 4 chunks created, all-miss by design (5625/6000 events incl. old-instance
internal warmup); **iters 2-5 (traced + 2 controls): ZERO misses, ZERO empty
issues on both stage leaders.** Commits ≈ 121/name/iter = 4 chunks × ~30 —
full coverage (old regime: ~31 = 1 chunk). The cumulative 25.1% miss
fraction in raw telemetry is warmup-window-only. The old steady 12.8-20.4%
class is eliminated by FIX-W.

**2. Pool cap tripped (by design, host SAFE): 199.9 GiB/rank, 123 keys, at
control2.** Trajectory (avg/max GiB per rank per iter): warmup→94.9/158.9 →
131.4/174.2 → 137.5/184.7 → 143.3/196.7 — decelerating but never
converging: **+6 GiB/rank/window steady churn** (MoE routing jitter mints
new exact-shape pool keys; bucket 1024 traded padding for key churn). Any
cap eventually trips; uncapped = slow crawl to host OOM (the wgm8row killer,
and the historical mid-run cudaHostAlloc class). Hazard note: the raise on
one rank wedged the other 15 in a collective — driver would have sat until
its 1 h timeout; caught at the 10-min watch round (Jack's new cadence rule),
scancel + drained-node resume (5abzeeir, the known "Kill task failed"
gotcha).

**3. Perf numbers from F1b are NOT valid** (126-140 tok/s/GPU controls):
confounded by pool cold-allocation/first-touch during windows, kineto traced
window, telemetry EVERY=1, and the miss-debug instrument.

**4. Fix implemented: family-reuse pinned pool.** Pool key (ndim, trailing
dims, dtype); allocate = best-fit smallest free buffer with dim-0 rows ≥
requested (offload/reload already slice buf[:real_rows], so oversized reuse
is safe by construction); 0-dim families isolated. Unit-tested on the box
(best-fit reuse, 0-dim isolation, cap trip): POOL_UNIT_TEST_OK. Expect pool
≈ in-flight high-water and FLAT across windows. R6 (pp2-familypool-v1)
validating on the proxy now; then F2 = full model, instrument OFF,
telemetry on, 8 controls, driven from the slurm lead (tj-q8yz15w-1).

### F2 (full-offload-32k-d4-fixw3): fixed machinery measured — memory goal
landed, throughput still −40%

Windows: warmup 37 (one-time 219 s pool build), traced 240, controls
353/284/150/401/411/431/425/418 — **steady tail ~401-431, median ~418**
(aggregate 320 dragged by early windows). **Peak 129 GiB vs baseline 228 —
offload now saves ~100 GiB/GPU.** Misses 0 steady (cumulative = warmup
only); **family pool CONVERGED: 17 keys, 138-148 GiB/rank, flat at iter 11,
no cap trip**; loss parity vs arm-3 window-for-window to ~1e-4.

Rank-0 (stage 0) trace, traced window 35.4 s wall (kineto-inflated):
- Host: **cudaMemcpyAsync ISSUANCE 10.4 s / 14,531 calls (714 µs/call!)**
  on the main thread — the dominant class; autograd-thread
  cudaStreamSynchronize 152 / 4.9 s (TE GroupedLinear m_splits readback
  class); deviceSync 11 / 2.6 s; dispatcher cudaEventSynchronize 140
  (= 35 layers × d4) / 2.1 s; main streamSync 8086 / 1.8 s; hostAlloc 5 /
  0.85 s (pool tail growth).
- GPU: SendRecv 17.6 s (pipeline peer wait, symptom of both sides
  stalling), Memcpy 4.0 s HtoD + 3.9 s DtoH (the ×4 offload volume,
  as predicted), compute only 3.6 s. busy≈wall (no overlap slack).

Next levers by leverage: (1) **ES OFF** — proxy-proven ES×offload
heavy-tail/issuance interaction; ES was only ever on for memory headroom,
and peak is now 129/275 GiB → ES unnecessary. (2) valve 16 (A/B in flight,
full_offload_32k_v16.json). (3) then m_splits + dispatcher classes (proxy
mission already sized paths). Watch script hardened with ServerAlive bounds
(one round wedged on a dead ssh read).

### v16 A/B (full-offload-32k-d4-fixw-v16): valve exonerated at 4× coverage

Tail 420/427/435/421/424 (median ~424) vs F2 valve-4 tail median ~418 —
identical within noise; drain rate fell 90%→58% of commits with no wall
effect. Valve is NOT the lever at full coverage either. Peak 129 GiB again;
loss parity window-for-window.

### ES-off A/B launched (full-offload-32k-d4-fixw-esoff): identical to F2
except PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False (valve 4). Trace
rationale: 10.4 s/step main-thread cudaMemcpyAsync issuance (714 µs/call) +
proxy-proven ES×offload heavy-tail interaction; ES only existed for memory
headroom and peak is now 129/275 GiB.

### ES-off A/B: EXONERATED (Jack called it)

Tail 413/420/426/427 (median ~423) vs ES-on ~418/~424 — identical within
noise. expandable_segments is NOT the issuance mechanism. Formally closed.

### Trace decomposition of the 10.4 s memcpy-issuance class (fixw3 rank 0)

Methodical chain, each step queried from the same trace:
1. Distribution: p50 = 6 µs (normal); **113 calls > 1 ms carry 98.4%**
   (10.21 of 10.38 s); worst single call blocks **1.21 s**.
2. All 113 are `aten::_to_copy` → `aten::copy_` → 4-byte (one 24-byte)
   **Memcpy DtoH (Device → Pageable)**; GPU-side copy 5 µs.
3. Per-call blocked time == GPU main-stream lag exactly (CPU waits until
   the main stream reaches the copy — pageable DtoH is stream-ordered and
   synchronous).
4. During the worst 1.21 s window the ENTIRE GPU was idle (~3.7 ms busy
   across ALL tracks incl. NCCL + copy streams) — the rank is stalled on a
   dependency that isn't running locally (peer-rank / event dependency),
   i.e. a cross-rank convoy amplified by per-layer host syncs, NOT a local
   backlog. CUDA_DEVICE_MAX_CONNECTIONS is unset everywhere (default 8) —
   the =1 foot-gun is excluded; queue-sharing with 10+ streams remains a
   secondary suspect.
5. Call-site localized: `tokens_per_expert.tolist()` in
   megatron/core/transformer/moe/experts.py (581 fused-fp8 / 678
   grouped-main / 1257 sequential). The dispatcher's staged-DtoH machinery
   (cuda_dtoh_point, d2h_event — the 140-call / 2.1 s eventSync class) is
   SUPPOSED to land tokens_per_expert on CPU first; a device tensor at
   tolist() means some branch bypasses it. Note
   `maybe_move_tensor_to_cpu` stages into PAGEABLE cpu memory with
   non_blocking=True (degraded for pageable dst) — suspicious in itself.
6. Backward twin: autograd-thread cudaStreamSynchronize 152 / 4.9 s —
   same family, to be localized after forward.

### White-box probe deployed (BT_MOE_TPE_DEBUG=1): logs device/type/config
at each tolist() site + dispatcher staged-move points (first 24/process).
experts.py + token_dispatcher.py patched on the qkpox9w tree (backups
.bak_preweil). New fast vehicle: **4-GPU PP2/EP2 proxy**
(pp2ep2-04-offload-16k.json, runner now takes a gpus param) — a2a +
m_splits + offload on one node, ~10-min cycles. pp2ep2-probe-v1 in flight.

### Probe result: ATTRIBUTION CORRECTED — the slow class is the dispatcher's
own staging, not experts.py tolist

EP2 proxy + full-model probe both show experts.py receives tokens_per_expert
already on **cpu** (fp8=None — the FP8 checkpoint dequantizes to bf16
compute, so the fp8/fused paths never run; permute_fusion=True,
dispatcher=alltoall, cuda_graph=none). Re-reading the trace with this: the
113 slow ops are `aten::_to_copy` = `.to("cpu")` = **the dispatcher's
`maybe_move_tensor_to_cpu` staging calls in _maybe_dtoh_and_synchronize** —
tolist() on a cpu tensor makes no CUDA call at all. Mechanism (complete):
staging writes to PAGEABLE host memory with non_blocking=True, which CUDA
cannot make async → each staging call blocks the MAIN thread until the
dtoh stream (chained behind the lagging main stream) executes the copy —
up to 1.2 s per call, ~10 s/step at 32k under offload. (The white-box probe
caught my wrong attribution — exactly its job.)

### FIX implemented (env-gated BT_MOE_PINNED_DTOH=1):
- moe_utils.maybe_move_tensor_to_cpu_pinned: stages into pinned buffers
  cached per (dispatcher instance, role key) → truly async enqueue; host
  reads stay ordered by the existing d2h_event.synchronize() at
  cuda_sync_point (unchanged).
- Aliasing hazard closed at the source: _AllToAll stores the split-sizes
  OBJECT in ctx for backward (upstream safe only because each staging made
  a fresh array). The three ep a2a call sites now pass
  `_split_sizes_as_list(...)` — immutable python lists materialized at USE
  time (post-sync, race-free) under BOTH staging modes.
- Files: moe_utils.py + token_dispatcher.py on qkpox9w tree (backups
  .bak_preweil). A/B: pp2ep2-pinned-v1 (proxy correctness) in flight, then
  full-model one-variable arm.

### Pinned-DtoH A/B: REFUTED — and the premise was wrong at the torch level

Full-model pinned arm tail 407/435/430/427/428 (median ~429) = same noise
band as F2 (~418) / v16 (~424) / esoff (~423). Trace comparison shows WHY:
the DtoH copy mix is IDENTICAL in both arms (398 pageable + 12,828 pinned)
— **`.to("cpu", non_blocking=True)` already allocates pinned destinations
in this torch**, so the dispatcher staging was never the pageable source
and BT_MOE_PINNED_DTOH changed nothing (proxy parity confirmed the code is
correctness-neutral; the a2a `_split_sizes_as_list` hardening is kept — it
closes a real aliasing hazard for any future buffer reuse). The 132 slow
pageable 4-byte copies (~10.8 s) come from an UNIDENTIFIED site that calls
a synchronous `.cpu()`/`.to("cpu")` per layer-chunk. Also notable: in the
pinned arm's trace the main-thread wait redistributed (deviceSync 2.6→9.8 s,
eventSync 2.1→2.5 s, memcpyAsync ~11 s) with wall unchanged — consistent
with all of these being SYMPTOMS of one upstream stall (main stream stalled
while the whole GPU idles), not independent costs.

### New instrument: BT_CUDA_SYNC_DEBUG=warn (backend.py, env-gated) —
torch's sync-debug mode warns with the python call site on EVERY
synchronizing CUDA op, deduped per site. Diagnosis boot
full-offload-32k-d4-syncdbg (2 repeats) in flight to NAME the caller of the
per-layer pageable sync, instead of another attribution guess.

### Sync-debug names the callers — and the fix already exists (ship's host caches)

BT_CUDA_SYNC_DEBUG=warn (one diagnosis boot) named every synchronizing call
site, deduped per site. The per-layer ones: **dsa_layout.py:159-161,188-189
(DSA packed-CP layout builders — boolean-mask nonzero + D2H per layer)**,
**rope_utils.py:222,224,239 (THD RoPE cu_seqlens .tolist()/.item() per
layer)**, dsa_cudnn_kernels.py:2130; per-microbatch packed_seq_params.py:45,61;
per-step training_runner.py:531-551. These are exactly the classes the
dispatcher-opt campaign already fixed on the ship stack:
**BT_DSA_CP_LAYOUT_CACHE + BT_THD_ROPE_HOST_CACHE** (+11-13% @131k,
parity-proven; this branch documented as lacking them). Patch found at
pp2cp8ep8/results/BF_hostcache_port_mcore_57efae08b.patch, applied CLEAN to
the qkpox9w tree (backups .bak_preweil; gates log at WARNING). The convoy
mechanism now reads coherently: per-layer host syncs gate each rank's next
a2a enqueue; under offload the per-layer stream time jitters → rank skew →
every sync pays the slowest rank's lag (baseline 0.27 s vs offload ~10 s for
the same sites).

### A/B in flight: full-offload-32k-d4-hostcache = F2 config + both cache
knobs ON (one boot, two knobs — they attack the same named class and were
shipped together; if the result needs decomposition we split after).

### HOST CACHES LAND: +17-19% — offload now ~497 tok/s/GPU

full-offload-32k-d4-hostcache (both gates confirmed ACTIVE): controls
473/312/193*/486/485/508/497/500 — **tail median ~497, step 19.5→16.4 s**
(*the recurring mid-run outlier window class, present in every arm). Loss
parity window-for-window; peak 130 GiB. Arm ladder at 32k/d4 now:
384-420 (broken machinery) → ~420 (machinery fixed) → **~497 (+ host
caches)**; bar 671; fullrecompute 581.

Trace (this boot): the memcpyAsync class is GONE from the stall list
(launches 90,745→24,249 — the RoPE per-sequence reads were those thousands
of copies); main-thread streamSync 8086→254 calls. Remaining main-thread
blockers: **dispatcher d2h_event.synchronize (cudaEventSynchronize) 140 =
layers×d4, 7.3 s traced** (now the top class — staged copies wait behind
main-stream lag), deviceSync 11/4.1 s, autograd streamSync 152/2.3 s
(backward twin, halved).

### Next A/B in flight: valve OFF (max_inflight_offloads null,
full_offload_32k_uncapped.json) + caches on — the valve's main-stream
waits fire on ~90% of commits and are a prime source of the lag the
dispatcher sync pays; the earlier v4→v16 A/B was masked by the 10 s RoPE
class. Safety: family-pool cap 200 GiB/rank + 146 GiB GPU headroom.
Ops: slurm jobs submitted while a node is drained stick in PD (Reason=None)
after resume — remedy is scancel + fresh dispatch; and NEVER combine pkill
with a dispatch naming the same script in one ssh command (self-match kills
the shell — bit us twice).

### Valve-off A/B: exonerated (third time) — tail ~505 vs ~497, marginal;
uncapped valve ran max_pending 38 with peak still 130 GiB (family pool +
natural bounds hold). Valve is definitively not the lag source.

### Trace comparison BASELINE vs OFFLOAD+CACHES (same 850 SendRecv calls):

| class | baseline (11.6 s step) | offload+hc (16.4 s) |
|---|---|---|
| SendRecv | 3.51 s | **13.24 s (×3.8)** |
| Memcpy | 0.04 s | 7.88 s (the offload copies) |
| compute | 3.49 s | 3.48 s (unchanged) |
| AllGather | 0.85 s | 2.27 s |

SendRecv distribution: baseline p50/p90/p99 = 1.0/2.5/6.1 ms; offload =
1.5/7.8/**443 ms** — p50 +54% but p99 ×73 → **heavy-tail rank-skew convoy
at the collectives, NOT raw bandwidth** (a uniform bandwidth hit would move
the median, not explode the tail). Remaining per-layer host coupling =
dispatcher d2h_event.synchronize (140/step, 7.3 s traced — data genuinely
needed on host to launch the a2a).

### A/B in flight (job 37): drop moe_act from offload modules
(full_offload_32k_nomoe.json; keep caches + valve off). RATIONALE: moe_act
is the one PER-RANK-VARIABLE copy volume (routing-dependent) — prime skew
source. PRE-REGISTERED: if routing-variance skew is the mechanism, SendRecv
p99 collapses toward baseline and tok/s jumps well past 550 (memory rises
but stays ≪ 228); if aggregate bandwidth, gain only ~ proportional to
removed bytes (~10-15%).

### NO-MOE OFFLOAD LANDS: ~596 tok/s/GPU — offload now BEATS fullrecompute

full-offload-32k-d4-hc-nomoe (caches + valve off + modules
core_attn/qkv/attn_proj): tail 577/597/597/578/596 — **median ~596, step
13.7 s**; peak 171 GiB (up from 130; still 57 GiB under baseline 228);
loss parity window-for-window. **Pre-registered prediction CONFIRMED:
SendRecv p99 collapsed 443 ms → 12 ms (~baseline scale)** — routing-variance
rank skew from the moe_act copies was the collective-stretch mechanism.

Ladder @32k/d4: broken 384-420 → machinery fixed ~420 → +host caches ~497
→ **+no-moe-offload ~596** | fullrecompute 581 | bar 671 | baseline 706.
OFFLOAD NOW BEATS FULLRECOMPUTE ON TPS (596 vs 581) at 171 vs 110 GiB.

Remaining classes (nomoe trace): deviceSync 11 calls/8.4 s traced (the
pipeline-bubble class — count fixed, duration = stage imbalance; baseline
pays 2.25 s of the same), autograd-thread streamSync 152/2.0 s = the
per-layer BACKWARD `torch.nonzero(topk_length > 0)` sync in
dsa_cudnn_kernels (box line 2130; fast path `all_sparse_bwd_rows_nonempty`
off at this config), dispatcher eventSync residue.

### Probe boot in flight (job 40): BT_DSA_BWD_PROBE=1 logs which fast-path
condition fails + whether any row is ACTUALLY empty (first 3 calls). If no
row is ever empty, the compaction (and its nonzero sync) is skippable under
a parity-validated flag — the bar-crossing candidate (596 + ~1.5-2 s/step
reclaim ≈ 660-700).

### Backward nonzero-skip lands: steady ~615; outlier class identified

Probe: the active DSA path is FusedSparseAttentionFunc (my first probe sat
in the unused fused-indexer class — corrected); its backward ALWAYS took the
empty-row compaction (torch.nonzero host sync per layer) because the fast
path was never wired. Empirical check: **any_empty_row=False in 24/24
samples (rows=4096)** → BT_DSA_ASSUME_NONEMPTY_ROWS=1 skips compaction.
Result: autograd streamSync class GONE from the trace; tail 616/621/609 →
**steady ~615**; loss/gn parity in band (numerically clean, as the probe
predicted). NOTE for productization: the flag asserts no-empty-rows — safe
for packed causal training (every row attends itself); needs a guard or
upstream fast-path wiring before mainlining.

**The recurring outlier window is ALWAYS control2** (54.6/42.4/43.1/76.2 s
across arms) — right after the kineto flush ("can take minutes") →
suspected profiling artifact, absent from clean runs (baseline's clean
pass-2 had none). Headline must come from a no-trace protocol.

Ladder @32k/d4 steady: 420 → 497 (host caches) → 596 (no-moe) → **~615
(+nonzero skip)** | fullrec 581 | bar 671 | baseline 706. Step 13.3 s vs
11.6 s baseline (−1.7 s to find). Remaining classes: pipeline bubble
(deviceSync 11/7.0 s traced; stage 1 = 40 layers+head also offloads MORE),
dispatcher eventSync 2.7 s, SendRecv still 2.4× baseline.

### A/B in flight (job 46): delta_bytes_across_pp_ranks = 32 GiB (stage 1
keeps 32 GiB on GPU → offloads less → bubble shrinks). PRE-REGISTERED:
+3-6% and deviceSync excess shrinks if stage imbalance is the bubble.

### Delta A/B crashed → THIRD latent machinery bug found and fixed

delta_bytes_across_pp_ranks=32GiB crashed the full model ("Chunk mismatch"
assert in on_group_commit_backward) and wedged the job (assert on one rank,
15 in collectives — scancel + node resume). Root cause (static + proxy
repro at 10-min scale, 4-GPU PP2/EP2 with delta=64MiB → SAME assert):
`finish_all_groups`' quick path (`_groups_to_reload`==0 ∧
`_groups_to_offload`==0 ∧ index>0) misfires MID-FORWARD on a chunk whose
EARLY groups are all policy-disabled (delta disables from the queue front →
nothing ever pends reload) → pop_forward_chunk advances to the NEXT chunk's
handler mid-forward → chunk mapping deranged. Margin never triggers it
(tail groups). FIX: quick path now requires
`index >= _max_group_size` (forward actually finished). Deployed;
crash-then-clean pair running on the proxy. Jack's directive stands:
target = TPS parity with baseline 706, not just the 671 bar.

### finish_all_groups fix proven crash-to-clean on the proxy; delta REFUTED

Proxy pair: delta config crashed with the same "Chunk mismatch" (repro),
fixed module ran the identical config clean (5673 tok/s/GPU, proxy band).
Ops note: the crashed proxy left a wedged trainer holding 59 GiB on GPU 0 —
proxy crashes need a pkill -9 + GPU check before rerun.
Full-model delta32b (fixed module): tail 624/610/608 ≈ nonzero arm's ~615 —
**delta stage-balance prediction REFUTED at 32 GiB** (stage-offload
imbalance is not the bubble driver, or mis-sized); parity held; run
turbulence high (control2 outlier 120 s — worst; one 19 s optim).

### Clean-protocol headline run in flight (job 53): best config
(nomoe + caches + nonzero-skip + valve off), bench_driver2c, NO
kineto/memory profilers, 10 controls. KEY MEASUREMENT-SIDE POINT: baseline's
706 came from a clean pass; every offload arm so far carried profilers +
the control2 flush artifact — the honest gap may be smaller than the
profile-laden tails suggest.

### CLEAN PROTOCOL: offload steady ~741 median (up to 806) — profilers were
costing 15-20% on every profiled arm

full-offload-32k-d4-CLEAN (nomoe + caches + nonzero-skip + valve off,
bench_driver2c, NO kineto / NO memory profiler, 10 windows):
304/666/751/91*/588/731/741/794/731/806 — **steady tail (main5-9)
731/741/794/731/806, median ~741**; peak 175.6 GiB alloc; losses in the
established parity band. (*one 89.6 s outlier window persists without
profilers — 1/10, separate class, disclosed; not the control2 flush
artifact.)

MEASUREMENT CAVEAT driving the next run: baseline's 706 was measured WITH
the memory profiler active (profile_driver protocol). The parity verdict
needs baseline through the SAME clean protocol —
full-baseline-32k-d4-CLEAN in flight (job 55). If clean baseline ≈706-780,
offload is at/near Jack's parity target; note offload at 175 GiB also
relieves allocator pressure vs baseline's 228/275 GiB, so genuine parity or
better is physically plausible.

### CLEAN BASELINE RECALIBRATES EVERYTHING: baseline ≈ 1005, offload ≈ 741

full-baseline-32k-d4-CLEAN (no profilers, NO cache envs): steady tail
1005/1000/1009/923/1010 — **median ~1005 tok/s/GPU, step 8.2 s** — the old
706 was profiler-taxed by +42% (baseline suffers the tax more: shorter
steps + more allocator events for the memory ring). Peak 236.6 GiB. The
outlier-window class hits baseline too (main4 14.6 s) → box/fabric-level,
NOT offload's.

HONEST SCOREBOARD (clean protocol, 32k/d4, same tree):
| arm | steady median | peak GiB |
|---|---:|---:|
| baseline (no recompute, no offload) | **~1005** | 237 |
| offload best-so-far (nomoe+caches+nonzero+valve-off) | **~741** | 176 |
| fullrecompute (profiled 581 — needs a clean rerun for fairness) | TBD | 110 |

Gap to Jack's parity target: **−26%** (was masked at −13% by asymmetric
profiler tax). All prior profiled numbers remain valid for A/B DIRECTION
but not for absolute headline. Every future arm runs the clean protocol.

### In flight (job 59): min_tensor_size 0 → 1,048,576 elements (upstream
default) — config had 0, so every tiny tensor offloaded (copy/event/
machinery cost, ~zero memory benefit). One change, clean protocol.

### min_tensor_size 1M: +3-4% — clean offload now ~772 median (best 819)

CLEAN-mts tail 744/746/772/802/819, parity in band, peak 175.8 GiB
(unchanged — tiny tensors were all cost, no benefit). Clean gap to
baseline-parity: −23% (10.4 vs 8.2 s steps). Outlier note: offload clean
runs show one ~90-120 s window per 10 (baseline's outlier was only
14.6 s) — offload amplifies the box-level hiccup class ~10×; investigate
after the main gap (suspect: a host/pinned interaction turning a hiccup
into a convoy stall).

### In flight (job 62): fraction 0.5 on top of best config. PRE-REGISTERED
DISCRIMINATOR: exposed-copy-bound → ≥ half-gap recovery (~+10%);
machinery/dependency-bound → ≤ +3%. Memory cost ~+30 GiB (fine vs 237).

### FRACTION 0.5: +15% — exposed-copy prediction CONFIRMED; the horizon rule

CLEAN-f05 tail 801/871/950/886/953 — **median ~886, best 953**; peak 207
GiB; parity in band. Both fractions pay ≈ exactly ONE direction's copy time
unoverlapped (f1.0: +2.2 s ≈ full one-way; f0.5: +1.1 s ≈ half) — and
fraction's implementation explains it: it disables the TRAILING (top-layer)
groups per chunk, whose reloads gate backward START and can never hide
(the classic offload-horizon rule). So fraction ≈ the horizon knob.
Clean ladder: 741 → 772 (mts) → **886 (f0.5)** | baseline 1005 (−12%).
Memory: 207 vs baseline 237 (nomoe already gave up the biggest group; to
recover savings later: moe_act back with uniform padded copies).
Outlier: ALWAYS main3 in clean offload runs (89.6/102/133.9 s) — systematic
position ⇒ suspected last-big-pool-growth window; amortizes if one-time;
verify with a long run later.

### In flight (job 67, after another stuck-PD scancel+redispatch): f0.75
knee search. Expect ~820-850 at ~+55 GiB saving if the copy-exposure is
linear in trailing-group volume.

### Fraction knee curve (clean, steady median / peak GiB):
f1.0 772/176 · f0.75 861/191 · f0.5 886/207 · baseline 1005/237.
Non-linear: the TOPMOST quarter of trailing groups cost −89 tok/s (horizon
effect strongest at the very top of the stack); f0.75→f0.5 buys only +25
for +16 GiB. Residual at f0.5 ≈ −12% ≈ one direction's copy time again.
main3 outlier in every clean offload arm (127 s here) — consistent
systematic class, parked. Stuck-PD scancel+redispatch needed twice more
(jobs 65, 70 — pattern: srun submitted immediately after a previous srun's
teardown; consider a 60 s settle before dispatch in the runners).

### In flight (job 72): f0.5 + BT_OFFLOAD_PREFETCH_DEPTH=16 — DISCRIMINATOR:
H2D-reload-side exposure → depth recovers it; D2H-side → no change.

### Depth 16 at f0.5: +3-4% — clean ladder now 917 vs baseline 1005 (−8.8%)

Tail 770/862/941/917/948 (median ~917; best windows in baseline's band).
H2D-reload exposure partially confirmed by the discriminator; d32 A/B in
flight to find the depth ceiling. Clean Pareto so far (median / peak GiB):
f1.0-nomoe 772/176 · f0.75 861/191 · f0.5 886/207 · f0.5+d16 **917**/207 ·
baseline 1005/237.

### d32: tail median ~943 (best 952) — −6.2% from baseline parity

Depth ladder at f0.5: d6 886 → d16 917 → d32 **943** (diminishing ~+3%/
doubling). Clean Pareto: f0.5+d32 943 @ 207 GiB vs baseline 1005 @ 237.
Runner now settles 90 s post-teardown (stuck-PD churn ended).

### Outlier-window attribution: pool REFUTED, auto-NUMA the live suspect

Pool telemetry: family pool converges at iter 2 (51.6/98.0 GiB, 10 keys,
FLAT) — no growth anywhere near the outlier window ⇒ pool-growth hypothesis
refuted. New evidence: **kernel auto-NUMA balancing is ON on both nodes
(numa_balancing=1) with 56.6e9 PTE updates / 14.4e9 hint faults / 1.0e9
migrations** — a scanner storm over our pinned-heavy address space is the
prime suspect for the one-giant-window class (and possibly steady tax).
/proc/sys is read-only in the pod → in-code fix candidate: mbind() the pool
buffers (kernel skips non-default-policy VMAs for NUMA hinting; also
hardens placement). Attribution first: numaprobe rerun in flight with a
10 s vmstat sampler to align counter spikes with window boundaries.

### OUTLIER ATTRIBUTED (measured, aligned): kernel auto-NUMA migration storm

numaprobe run: main3 outlier window spans epoch [1787550141, 1787550283];
the vmstat sampler shows a sustained migration storm [1787550152,
1787550282] — exactly inside it (58-90k pgmigrate + ~1M hint faults per
10 s for ~130 s), then settles for the rest of the run. One-time per
process lifetime → amortizes in production; excluded-with-disclosure in
headlines. FIX (in our control; /proc/sys read-only in pod): mbind() the
pinned pool buffers — auto-NUMA skips VMA-policied regions + placement
enforced by policy. Implemented env-gated BT_OFFLOAD_POOL_MBIND (default
ON with NUMA bind), deployed. Fleet ask for infra later: numa_balancing=0
on B300 training nodes.

### In flight: d48 boot (also carries mbind — two changes but SEPARABLE
observables: tail←depth, main3-outlier←mbind; noted per the A/B rule).

### Depth ceiling + noise floor reached; d48 confounded run

d48(+mbind) tail ~844 with THREE storm windows — regressed, but note the
numaprobe run was itself a second d32-no-mbind sample with tail ~890 (vs
943 first sample): **run-to-run noise ±4-5% now exceeds lever sizes beyond
d16**. Working band: f0.5 + d16-32 ≈ 890-945 vs baseline 1005 (single
sample — baseline needs replicates too before finer claims).
d32+mbind+sampler deconfound boot in flight (tail: depth stats; outliers +
sampler: mbind effect). No mbind failure warnings in d48 boot (policy
applied); storm may also roam torch's other pinned/anon regions.

## MISSION CLOSED (2026-08-23, Jack's wrap order) — FINAL STATE

**Verdict:** all three machinery bugs fixed and proven; offload rose from
55% to 89-94% of the TRUE baseline; the remaining 6-11% is quantified and
attributed (reload horizon + per-layer dispatcher host dependency + copy
residue + ±4-5% run noise). Box tj-q8yz15w left clean (queue empty, GPUs 0).

**Final clean-protocol numbers (32k/d4, PP2/EP8/CP8, full GLM-5.2):**
- baseline (no recompute, no offload): ~1005 tok/s/GPU, 237 GiB (1 clean run)
- offload speed point (3 groups, f0.5, mts 1M, d16-32, caches, valve off):
  **~890-945 across replicates**, 207 GiB (saves 30)
- offload memory point (f1.0 nomoe): ~772, 176 GiB (saves 61)
- 4-group f1.0 (max savings, 130 GiB / saves 107): clean TPS unmeasured;
  profiled-era ~497-equivalent; needs moe-uniform padding to be fast.
- All old profiler-protocol numbers (706 baseline, etc.) are ~15-42% taxed.

**Fixes deployed on the qkpox9w tree** (all env-gated or backed up as
*.bak_preweil; candidates for the PR):
1. backend.py: offload-manager reset after the 1-datum startup warmup
   (FIX-W — the original miss-storm root cause) + BT_CUDA_SYNC_DEBUG hook.
2. fine_grained_activation_offload.py: family-reuse pinned pool +
   BT_OFFLOAD_POOL_MAX_GB loud cap (host-OOM class) + finish_all_groups
   mid-forward fix (delta crash) + mbind on pool buffers (auto-NUMA) +
   pool/reload telemetry + BT_OFFLOAD_MISS_DEBUG event stream.
3. dsa.py + rope_utils.py: ship host-cache port (BT_DSA_CP_LAYOUT_CACHE,
   BT_THD_ROPE_HOST_CACHE).
4. dsa_cudnn_kernels.py: BT_DSA_ASSUME_NONEMPTY_ROWS (needs a guard or
   upstream fast-path wiring before mainlining) + probe.
5. token_dispatcher.py/moe_utils.py: _split_sizes_as_list aliasing
   hardening + pinned-staging variant (inert — torch already pins).
6. Ops: run_clean_arm.sh / run_full_arm.sh / run_pp2_arm.sh runners,
   watch_f1b_round pattern, 90 s slurm settle, EP2/PP2 proxy vehicle
   (glm52-debug-0d4m-pp2 + lps1062_pp2fix/).

**Open items (next session):** moe_act uniform-padded offload (recover the
107-GiB point at speed); replicate baseline 2-3×; d32+mbind deconfound
(dispatch script ready: dispatch_d32mbind.sh); infra ask: numa_balancing=0
on B300 fleet; productize ASSUME_NONEMPTY guard; 131k ladder vs
fullrecompute 790; one fresh-subagent review of the box-bound diff before
PR (lightweight-review policy).

## Reference numbers (different trees/envs — ratio context only)

- PP2/CP8/EP8 @131k record: 918 tok/s/GPU d4 / 1052 d16 (box w56lorq,
  campaign tree + env; full recompute era).
- Baseline of record @131k d2: 645 tok/s/GPU (full recompute).
- This box, pr1070 tree @262k: R2 383 d2, R3 522 d4 (full recompute).
