# Activation Placement Plan — recompute vs offload vs GPU

Decided by Jack 2026-08-20 (with hilbert). Goal: **maximize throughput** on
GLM-5.2 PP2/CP8/EP8 @131k by eliminating most of the activation
recomputation tax. Stacks on checkpoint **PR #1070**.

---

# FOR HUMANS

## The picture

Every transformer layer saves ~2.94 GiB of intermediate data ("activations")
per microbatch that the backward pass needs. Today we throw almost all of it
away and re-run the forward math to regenerate it — that re-run costs **46 s
of every 133 s step (~35%)**. This plan splits the layer's data three ways:

| Module (part of the layer) | Decision | Size (GiB/layer/mb) | Why |
|---|---|---|---|
| **Expert (MoE) block** (expert activations + combine saves) | **OFFLOAD to CPU** | ~1.2 | Worst thing to recompute — re-running it re-ships every token between GPUs (the all-to-all). Biggest memory bar. Cheap to stream over PCIe. |
| **Attention projection inputs** (qkv-proj + out-proj) | **OFFLOAD to CPU** | 0.4 | Clean single tensors; moving them is what buys the GPU its safety margin. |
| **Core attention** (sparse-attention kernel + indexer state) | **RECOMPUTE** (keep as today) | 0.7 saved | Cheapest thing to re-run: pure local math, no token shipping, ~8–16 ms/layer. Storing it needs 56 GiB we don't have; offloading it spends PCIe margin to save only ~5 s/step. Stock config. |
| **Norms / residual / router glue** + layer inputs | **STORE ON GPU** | ~0.9 | Dozens of tiny tensors per layer — per-tensor offload bookkeeping costs more than the bytes. Fits in GPU headroom. |

## What we get

- Recovers **~39 of the 46 s** of recompute per step → **~+40% throughput**
  (984 → ~1,400 tok/s/GPU; ~1,540 on the record environment).
- Also deletes ~10 s of the exposed all-to-all for free (the recompute
  pass's share of it). Remaining a2a (~19 s) is the next lever afterward.

## What has to be true first (the gates)

1. **One-boot memory census** — the "glue" size is partly inferred
   (0.44–0.89 GiB range). GPU storage works at the midpoint, is tight at
   the high end. Measure before freezing the split; if high, the biggest
   glue tensors move to the offload bucket.
2. **Pinned-memory pool fix** — the one offload trial ever run lost 30% to
   allocating fresh pinned buffers every layer (3-line patch to frozen
   vendored code; needs the freeze-exception process).
3. **Two new hook sites** — dispatcher-combine and the projection inputs.
   (Expert activations already have working, proven hooks.)
4. **First hardware validation of the backpressure valve** (committed, 
   never booted).

## What we're deliberately NOT doing

- **Not offloading core attention** — phase-2 A/B once offload is proven:
  it's a config flag worth ~+5%, but it halves the PCIe safety margin, so
  it doesn't go in the first build.
- **Not compressing the stash to FP8** — fits uncompressed; keep as margin.
- **Fallback if hooks slip:** the block+K dial (keep 25 layers stored,
  zero code, ~12–13%) — same direction, smaller step.

---

# FOR AI AGENTS

Rehydration: read [[lps-1062-pr1070-baseline]] memory first. Evidence
notebook: `pp2cp8ep8/analysis/notebooks/recompute_cost_analysis.ipynb`
(executed; per-module time/memory/deal charts). This section is dense by
design; the human section above is the decision of record.

## Baseline & stacking

- Checkpoint: PR #1070, branch `jackrao/lps-1062-pp2-mn-combined`
  (basetenlabs/trainers), 6 commits, base main@f6226626. Stack new PRs on
  this branch until merge, then retarget main. Offload plumbing commits
  f2407a10 / 6d8b22da / 4e7d5d3e (ActivationOffloadConfig + backpressure
  valve `max_inflight_offloads`) sit on the old campaign branch —
  cherry-pick them onto the stack as its first PR; valve has ZERO hardware
  evidence post-commit.

## Constants of record (fe127_d16_rank0 trace + W1b fits + benches)

- Step 133.34 s @131k d16 (984 tok/s/GPU fixed wheel; 1089–1103 record
  band). Passes: true-fwd ~44 s, recompute 46.41 s, true-bwd 35.29 s.
  1F1B canonical, in-flight I=2, PP bubble 7.2% (floor 5.9%).
- EP a2a 28.55 s resident, 0% compute overlap (verified kernel-level);
  recompute's share 9.78 s (1680 of 5040 ops = exactly 1/3). PP p2p 8.55 s
  ≈ all fill/drain bubble. Pure idle 20.23 s (nonzero D2H ~7 s, launch tax
  ~10.4 s, seam ~2 s). CP AG/RS 7.49 s ON the compute stream.
- Memory: S_eager ≈ 2.94 GiB/layer/mb-set, S_ckpt = 0.19. Census
  (GiB/layer/mb): core-attn internals 0.70; moe_act 0.75–1.15; combine
  ~0.20–0.25; qkv-in 0.20; proj-in 0.20; expert-fc1 0.00 (LoRA-frozen, TE
  saves no input); remainder 0.44–0.89 (BY SUBTRACTION — census gap,
  gate #1). Peaks @131k d16 stage-1: ~183 GiB smi / ~164 torch-reserved;
  +20 cold-pool burst; +60 boot#1 transient; cap ~267.7 GiB. Reserved
  creeps +21 GiB early steps then plateaus — matched-step/plateau reads
  only.
- Recompute time per MoE-layer-mb (76.3 ms wall): a2a 17.5, elementwise
  9.7, expert-gemm 7.0, main-gemm 5.8, CP-AG 4.7, dispatch-sort 4.5,
  indexer 3.3, attn-fwd 3.7, CP-RS 2.9, cat 2.8, idle 14.3. Deal ratios
  (GiB saved / ms): projections ~0.11, glue ~0.11, core-attn 0.05–0.10,
  MoE block ~0.027.
- Bandwidth (tj-wlmlkeq direct bench, huygens 2026-08-20; confirms
  campaign PCIE_OFFLOAD_BW_BENCH.md): per-GPU pinned D2H 57.3 / H2D 55.7
  GB/s solo; single-GPU bidi 48.3+48.3; all-8 unidirectional 456 D2H /
  439 H2D aggregate, NO per-GPU collapse; all-8 bidi ~29/dir/GPU
  (campaign box; unmeasured on tj-wlmlkeq); NUMA remote <1% single-GPU
  (multi-GPU cross-socket UNMEASURED — do not rely); pageable 5× worse.
  Host: 2× Xeon 6767P, 3.93 TiB physical, **cgroup memory.max = 2.42
  TiB** (size against this, not physical). GPUs 0–3 → NUMA0, 4–7 → NUMA1;
  NVMe RAID0 27.9 TB on NUMA1. Bench source: /tmp/cuda_host_bw.cu on box.

## The configuration

- `recompute: {granularity: selective}` (core_attn checkpointed, all else
  eager) — replaces full recompute.
- Fine-grained activation offload ON for: moe_act group (hooks exist,
  proven at 32k), dispatcher-combine (HOOK TO BUILD — single
  off_interface wrap in token dispatcher, force-release discipline),
  attention qkv-proj + out-proj inputs (HOOK TO BUILD — attention.py hook
  sites are not on GLM-5.2's AbsorbedMLA path). Do NOT enable
  offload_core_attention in phase 1 (phase-2 A/B, ~+5%, halves PCIe
  margin). expert_fc1 group: drop from module lists (inert under LoRA).
- Pinned pool: patch the MoE-groups-off-pool hardcode
  (fine_grained_activation_offload.py:362-366 class) — pad-to-max makes
  shapes constant, pool-safe. Freeze-exception required (precedent:
  gauss's cache-clear probe).
- Placement: pinned host buffers NUMA-local per huygens; large async
  copies; valve `max_inflight_offloads` tuned so stash never queues past
  ~2 layers; prefetch order within a layer's bwd = MoE tensors first
  (needed first: bwd order is MoE-bwd → attn-bwd), projections last;
  double-buffer 1 layer ahead (~5 GiB GPU reserve).

## Arithmetic (stage 1 = binding, 40 layers × I=2 = 80 sets)

- Offloaded: ~1.6 GiB/layer-mb (1.2 MoE + 0.4 proj) → stash ~22 GB/s
  during fwd (72 ms/layer), prefetch ~24 GB/s during bwd (~66 ms/layer
  incl. retained core-attn recompute) — ≤50% of solo budget, safe even at
  the 29/dir all-8-bidi seam floor. Node ~180 GB/s vs 440+ measured.
- GPU-stored: 0.19 input + 0.44–0.89 glue ≈ 0.63–1.08 → 50–86 GiB + ~5
  prefetch buffers + 20 burst vs ~85–100 headroom: OK at census-mid,
  TIGHT at census-hi → gate #1 (census boot) decides; contingency =
  offload the largest glue tensors.
- Retained recompute: core-attn ~5–7 s/step. Projected step ≈ 94 s ≈
  1,400 tok/s/GPU (+40%); +5% more available in phase-2 attn-offload A/B.
- CPU pinned: 80 × 1.6 × 8 ranks ≈ 1.0 TiB/node, NUMA-split ~512
  GiB/socket, vs 2.42 TiB cgroup — OK, leave room for dataloader.

## Validation ladder (standing rules apply)

1. Census boot: `torch.cuda.memory._record_memory_history` +
   BT_PEAK_MEM_REPORT on PP2 @131k d2 — closes the glue row; no PP2
   allocator snapshot exists today (256k PP1/CP16 snapshots at
   `~/perf_profiles/lps-1062/glm52-b300-s256k/` transfer imperfectly).
2. Selective-recompute-only boot (no offload): memory must match E1-class
   prediction; banks nothing but validates the scope switch.
3. Offload arms, one variable at a time: moe_act → +combine → +proj;
   matched-step memory reads; step-time bar pre-registered per arm.
4. Parity: NOISE-RELATIVE bars only (per-token floor 3.7–5.5 on this
   stack; never 1e-6/1e-3). DSA tests green + canonical env block before
   anchors. BT_SAVE_STATE_SYNC=1 is a NO-OP until queue item Q1 merges —
   avoid /save_state on soaks until then.
5. One fresh-subagent review per PR, no round-trips
   ([[lightweight-review-policy]]).

## Pointers

- Evidence notebook: `analysis/notebooks/recompute_cost_analysis.ipynb`
- Campaign benches: `results/PCIE_OFFLOAD_BW_BENCH.md`,
  `runs/overnight_20260813_overlap_campaign/MEMORY_LEG_DECISION.md`
- Queue/ledger: `pr_shaping/MORNING_MERGE_QUEUE.md` (Q1 save-toggle, Q2
  B/F caches precede or ride alongside this stack)
- Trace: `~/perf_profiles/lps-1062/pp2cp8ep8/fe127_d16_rank0.pt.trace.json`
