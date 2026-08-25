# postmerge_20260824_131k — GLM-5.2 131k PP2/EP8/CP8 on tip of main (post-#1070)

First current-state reference trace after PR #1070 merged (`71a9f3b7`). Per the
2026-08-24 prune decision: this supersedes the campaign-era 131k traces; pairs
with `../overnight_20260822_262k_pr1070/` R3 for the same-tree 131k↔262k
comparison.

## Setup

- Box: devbox q4grmdq (`tj-q4grmdq`), 2×8 B300 (275 GiB/GPU), ali plane.
  Nodes: b300-1-izksekdp-0001 (leader/HTTP :8001 — SSH via `tj-q4grmdq-1`),
  b300-1-pjps8ep6-0013 (`tj-q4grmdq`; NOTE: the bare alias lands on the
  NON-leader node).
- Code: `devboxes/q4grmdq/trainers` @ **71a9f3b7** (origin/main tip = the
  #1070 merge). Venv pre-existing from the box's f3fb2fb5 provision (uv.lock
  untouched by #1070). Clean env: no TF32-head knob, no NCCL ship env —
  matches the R1/R2/R3 convention.
- Config: `trainer_131k_main_pp2cp8ep8.json` — TP1/PP2/EP8/CP8, LoRA r32,
  flash attn, weight_sync disabled, max_seq_len 131072.
- Driver: canonical `tools/profile_driver_new.py`, `--datums 4` (= pipeline
  M=4; DP=1) → 524,288 tok/step, `--control-repeats 2`.

## Result

| run | tok/s/GPU | step | mfu3x | peak mem | canaries |
|---|---:|---:|---:|---:|---|
| postmerge-131k-d4 | 371 (controls 349, 395) | 88.4 s | 3.4% | 143 GiB | loss 12.309–12.317, gn 0.35–0.46 ✓ |
| postmerge-131k-d4-settle (4 controls, hot server) | **324 — settled band ~290–395, noisy** (354, 342, 292, 316) | 101.1 s | 3.0% | 144 GiB | loss ↓12.263–12.306, gn 0.35–0.44 ✓ |

**Settled headline: ~330±40 tok/s/GPU @131k/d4 on merged main, clean env** —
no clean steady state (controls degrade within a run), vs the scale branch's
ultra-tight 787–792. ~2.3× slower AND ~±15% run-to-run noisier.

Reference points: campaign record 918 @131k/d4 (campaign tree + ship env);
fullmodel fullrec 790 @131k/d4 (scale branch c226338a, clean-ish);
R3 522 @262k/d4 (PR tree, 2× tokens).

## Trace read (rank 0, 106.5 s traced window; kineto +20.5%)

- **Compute healthy**: DSA bwd 5.5 ms/call = exactly half of R3's 262k
  per-call (11.5 ms) — the packed-CP DSA offset fix (#1134, in main but NOT
  in the R2/R3 trees) is perf-neutral. nvjet GEMM 5.4 s total.
- **68% of the window is NCCL**: SendRecv 57.8 s (1270 calls, avg 45 ms,
  max 23.5 s), AllReduce 7.0 s (one 6.1 s step-end call), **Broadcast 6.6 s
  (11 calls @ 600 ms — on the PR tree the identical 11 calls were 26 µs)**,
  AllGather 1.3 s, ReduceScatter 0.8 s.
- Same skeleton as R3 (identical collective call counts) — the delta is all
  wait time: rank 0 (stage 0) idles for stage 1. Stage-1 heavies on main:
  FP32 SIMT LM head (TF32 PR still deferred) + whatever the scale branch had
  that didn't merge.
- **Unexplained gap**: 371 vs the scale branch's 790 at the same shape is NOT
  accounted for by compute or the TF32 head (the branch lacked it too).
  Needs the stage-1 view (rank-8 trace — main has no BT_PROFILE_RANKS) or a
  scale-branch 131k trace to diff against.

## Debug-model bisect (2026-08-24 late) — `release_cached_vram` is the regression

Vehicle: `glm52-debug-1d1m` (1 dense + 1 MoE layer, everything else
full-size), 1 GPU, PP1/CP1/EP1, 8192×d4 = 32,768 tok/step (matches the full
run's per-GPU token load), full recompute, driver = profile_driver_new.py,
3 controls. Boot ~75 s per arm. Kit on box: `/root/.cache/user_artifacts/lps1062_debug_ab/`.

| arm | tok/s/GPU | fb step |
|---|---:|---:|
| main @ 71a9f3b7 | 8,041 (noisy: 6965/8810/8621) | ~4.1 s |
| branch c226338a (lps1062-scale) | 10,812 (tight) | 3.0 s |
| **main + `BT_DISABLE_VRAM_RELEASE=1`** | **10,810 (tight: 10938/10697/10796)** | **3.0 s** |

**`release_cached_vram` is the regression.** It exists only on main (added
after the branch forked, LPS-1065): it wraps `finalize_model_grads`, so every
step does `torch.cuda.synchronize()` + `torch.cuda.empty_cache()` (also once
after `dp_loss_gather`). The branch never had it. Disabling it on main closes
the ENTIRE debug-scale gap to the branch (10,810 vs 10,812).

Box patches (box `q4grmdq` checkout only, NOT committed anywhere):
`models/src/loops_models/profiling.py` reads `BT_PROFILE_RANKS` into
`rank_set`; `server-interface/.../cuda_memory.py` early-returns when
`BT_DISABLE_VRAM_RELEASE=1`.

Full-model verification (main + `BT_DISABLE_VRAM_RELEASE=1` at 131k
PP2/EP8/CP8): see below.

## Full-model confirmation — 771 tok/s/GPU

Main @ 71a9f3b7 + `BT_DISABLE_VRAM_RELEASE=1`, 131k×d4 PP2/EP8/CP8, same box,
same driver (`main-131k-d4-novr`): **771 tok/s/GPU** (controls 762/776/775,
tight; peak 144 GiB; canaries clean) vs 330 broken / 794 branch. The per-step
flush accounts for **97% of the gap**; the residual ~3% is noise-level (the
loss-path divergence may or may not be real — not worth chasing at this size).

The rank-8 "stage 1 is slow" picture was the convoy: the per-step
`empty_cache()`+`synchronize()` stalls every rank's allocator at each step
boundary; the pipeline then amplifies it, and stage 1's loss-path host reads
park on the stalled GPU. Stage 1 was never the root cause.

Also measured on the branch recheck: 794 tok/s/GPU reproduces exactly on this
box (controls 795/794/794). And the campaign-era env knobs are confirmed
immaterial at 131k PP2 (ship env 552 vs 550 without, campaign tree).

**Fix for main (separate PR):** `release_cached_vram` must not run per step.
Options: run once after boot/warmup only; gate on actual free-memory pressure;
or make it opt-in via env. It exists for LPS-1065 (NCCL/cuBLAS out-of-pool
allocations) so the safety case needs preserving — but at 131k PP2 the peak
was identical with it off (144 GiB).

## Rank-8 (stage 1) trace — why stage 1 is slow

Captured via a box-local patch: `ProfilingConfig.from_env` now reads
`BT_PROFILE_RANKS` (comma-separated) into `rank_set` (main has no knob;
patch lives only in the box checkout `devboxes/q4grmdq/trainers`). Run
`postmerge-131k-d4-r08` (334 tok/s/GPU — same regime) recorded ranks 0+8.

Rank 8 (stage 1: 40 layers + LM head + loss), 97.6 s wall, 79.7 s busy:

- **Same host-sync storm as rank 0, plus its own extras.** `aten::nonzero`
  32.9 s CPU-blocked (rank 0: 32.2 s — the dispatcher/DSA host-sync class,
  B/F never merged). On top: `aten::to`+`_to_copy` 21.2 s (rank 0: 8.4 s),
  `aten::copy_` 8.2 s (3.5 s), `cudaMemcpyAsync` 7.0 s, and **242
  `aten::item` scalar reads, 6.6 s** — the loss-path host round-trips only
  the last stage does.
- **The FP32 SIMT LM head lives here**: 32 `cutlass3x_sm100_simt_sgemm_f32…`
  calls, 3.7 s GPU (the "8×117 ms" class the 262k analysis flagged on main).
- Rank 8's own Broadcast time is 0.36 s vs rank 0's 6.6 s — i.e. rank 8
  posts the step-end scalar broadcast ~6.6 s **late**; rank 0 just waits.
  The lateness is the CPU-bound loss path, not network.
- Every PP handshake is wrapped in `cudaDeviceSynchronize` (12 calls,
  19.8 s on rank 8 / 30.3 s on rank 0) — a device-wide sync per stage-boundary
  send/recv; one slow handshake stalls the whole node.

Mechanism: stage 1's per-microbatch cost = its layers' compute **plus** a
fixed host-roundtrip block (loss reads, dtype shuffles, FP32 head). Those
fixed costs sit on the pipeline critical path (stage 0 can't get grads until
stage 1's CPU finishes), and they amortize 2× better at R3's 262k — which is
why per-token throughput fell 2.3× at 131k while compute kernels stayed
healthy. Candidates that attack exactly this: the B/F dispatcher host-sync
caches (never merged), the TF32 LM head (still deferred), and whatever else
lps1062-scale carries that #1070 didn't.

## Artifacts

- `postmerge-131k-d4.json` — driver output (windows + aggregates).
- `postmerge-131k-d4.runlog` — driver stdout.
- `traces/*.pt.trace.json.gz` — rank-0 kineto trace (73 MB gz / 733 MB raw).
- `traces/memory.rank0.pickle.gz` — rank-0 memory snapshot (130 MB raw).
- `mem_max_*.txt` — nvidia-smi per-GPU peaks, both nodes.
- Box-side kit: `/root/.cache/user_artifacts/lps1062_131k_postmerge/`
  (boot_trainer.sh, mem_poller.sh, driver copies, config).
