# Activation Placement Plan — recompute vs offload vs GPU

Decided by Jack 2026-08-20 (with hilbert). Goal: **maximize throughput** on
GLM-5.2 PP2/CP8/EP8 @131k by eliminating most of the activation
recomputation tax. Stacks on checkpoint **PR #1070**.

**Execution owner: banach (handed off by hilbert, 2026-08-20).**

---

---

# AMENDMENTS (banach, 2026-08-20 — read before acting on anything below)

Current state, code locations and verification status live in
**`ACTIVATION_PLACEMENT_STATE.md`**. Execution handed to **conway**.

The body of this plan is still the design of record, but these numbers and
decisions in it are now SUPERSEDED:

**1. The offload set is `moe_act` + `attn_proj` only. There is no combine arm.**
The `moe_combine` group captures ~0 bytes on this path. Its 0.20-0.25 GiB/layer
census row was derived from tensor shapes, not measured against our dispatcher
and Transformer Engine path: TE's mask-map unpermute with no merging probs saves
only `row_id_map` + `pad_offsets`, sub-megabyte index tensors below
`min_offloaded_tensor_size`. The big dispatcher-seam tensors (~1.6 GiB each) are
transient — freed in forward, never saved — so there is no better hook site.
`qkv_linear` is also deferred (its tensor is shared with the core-attention
checkpoint; offloading it either double-stores or puts an H2D transfer inside
core-attn's recompute path).
So: **offloaded ~0.95-1.35 GiB/layer-mb, not 1.6.** Glue absorbs the combine
bytes (it was derived by subtraction): **0.64-1.14, not 0.44-0.89.** Worst-case
projected peak on the binding stage: **~227 GiB against a ~248 GiB effective
ceiling — ~21 GiB margin**, versus ~35 GiB as planned. Gate #1 matters more now.

**2. NUMA-local pinning is GATE-CLASS, not a placement detail.** All-8
bidirectional, sustained, per GPU: local 27.5/28.8 GB/s (meets the ~22/24
demand); **interleaved 15.6/16.8 — BELOW demand, and interleaved is what an
unbound process gets by default**; remote 7.4/7.8. Cross-socket penalty is 73%
at 8 GPUs vs <1% at 1 — the campaign's single-GPU figure does not generalize.
Also structural: at 8 GPUs the node total caps near 450 GB/s *regardless of
direction mix* (directions split one ceiling, unlike the additive single-GPU
case). `numactl` is absent on these hosts, so binding is in-allocator.
Data: `results/ALL8_BIDI_OFFLOAD_BW_BENCH.md`.

**3. The 1F1B bandwidth model.** F and B compute do not overlap on a rank;
"in-flight 2" means two microbatches' activations are HELD, not two phases
running. Sustained regime is all-8 UNIdirectional (~2.4x margin); bidirectional
occurs only at phase seams. Phase-2 core-attn offload is CONDITIONALLY alive —
it fits unidirectional but not the seams, so it depends on phase discipline,
decided by rung-3 exposed-stall measurement.

**4. Ladder rescope.** Selective-recompute-only does NOT fit at 131k (~292 GiB
vs ~248) — selective recompute and the offload are ONE PACKAGE and there is no
selective-only fallback (block+K remains the fallback, a different mechanism).
Rung 2a moved to 32k; rung 2b (deliberate 131k OOM) is conditional and probably
skipped; **rung 2c added** — full phase-1 config at 32k as a pure BRING-UP pass
(crashes and wiring only, NOT tuning; no throughput claim) that also smoke-runs
every measurement instrument. Rung 2a/2c can run single-node at PP1.

**5. Rung 5 is a MATCHED d16 PAIR on our own tree; the deliverable is the
RATIO.** Not one run against the historical record band. TF32 is cancelled per
Jack ("TF32. I don't want that.") — it cancels out of a matched pair anyway. No
comparisons to the 984 / 1089-1103 band anywhere; absolute figures are context
only. An absolute ship-config number is a separate follow-on, not this
workstream.

**6. Layer geometry.** GLM-5.2 is not uniform: `mlp_layer_types = 3 dense + 75
sparse` over 78 layers, and `index_topk_freq = 4` (one leader computes DSA top-k,
next three share it — 11 leaders on stage 0, 10 on stage 1). With the 38/40
split, rank 0 = 3 dense + 35 MoE at in-flight 2 (70 MoE sets); rank 8 = 40 MoE at
in-flight **1** (40 sets) plus loss/logits. The stages bind on DIFFERENT things:
rank 0 on activation-set count (drives offload volume and PCIe demand), rank 8 on
total peak (drives whether the resident bucket fits). Never average per-rank
census figures; MoE-only denominators are 70 and 40.

**7. Census hygiene.** Identify tensors in allocator snapshots by ALLOCATION
SITE, never by size signature (the DSA topk stash, [1,16384,2048] int64 =
268435456 B, aliases a different tensor at 256k/CP32). That stash is ~5.9 GB on
rank 0 / ~2.7 GB on rank 8 and is paid TODAY under full recompute — it is not a
cost of this change and must be its own census row, not folded into glue.
Project net-new resident with GLUE ALONE: the 0.19 GiB/set inputs are already in
today's peak (S_ckpt), so glue+input double-counts.

**8. `NVTE_CPU_OFFLOAD_V1=1` must be in the LAUNCHER env.** TE latches it at
import; the bridge validator reads it lazily. Set worker-side after TE imports
and validation PASSES while TE silently keeps the V0 path. Absent entirely fails
loud. Boot log reports env and latched value; `env=1 latch=0` is a FAILED boot.

**9. Code is committed, not patched.** The vendored megatron-core changes are
real commits on `basetenlabs/Megatron-LM` @ `jackrao/lps-1062-activation-offload`
(the "frozen tree / freeze-exception" framing was over-cautious — it is our own
fork and its current pin is already one of our commits), pointer-bumped through
`basetenlabs/Megatron-Bridge`, surfaced in **trainers PR #1074** (draft). The
`bt_offload_*` patch files are deleted. Workflow: commit on the laptop, push,
`git pull` on the box — do not scp code.

**10. Nothing has run on GPU.** Three provisions failed on GPU capacity
(`FailedScheduling: Insufficient nvidia.com/gpu`, autoscaler cannot add nodes) —
NOT the platform regression an earlier note claimed; that diagnosis is retracted.
No fallback accelerator exists: B200's 180 GB is under the ~200-227 GiB projected
peak even after offload.

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
  (984 → ~1,400 tok/s/GPU; ~1,540 on the record environment). That is the
  ONE number — ~10 s of it is the exposed all-to-all the recompute pass
  was re-shipping (a component of the 39 s, not an extra win on top).
- **The two changes are ONE PACKAGE** (established 2026-08-20): at 131k
  the offload is load-bearing, not polish — selective recompute alone
  stores 2.24 GiB/layer/mb and would put rank 0 ~45 GiB OVER the memory
  ceiling. There is no "selective-only" configuration at the mission
  length, and therefore no partial fallback inside the package; the only
  fallback is the block+K dial (~12–13%), a different mechanism.
- After this lands, the remaining exposed all-to-all is ~19 s — the next
  lever.

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
5. **NUMA-local pinned buffers — a gate, not a tuning knob** (bench
   2026-08-20): with all 8 GPUs offloading at once, host buffers on the
   wrong CPU socket collapse bandwidth by 73%, and the OS-DEFAULT
   spread-across-both placement by 42% — **below the feature's
   requirement**. Get this wrong and phase 1 silently fails by ~30%.
   The box has no numactl, so binding must happen inside the
   pinned-buffer allocator itself (rides the pool patch), and boot must
   verify actual page placement, not assume it.

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
- Indexer leaders (freq-4 rule: a layer computes its own selection iff
  1-based idx ≤3 or (idx−3)%4==0): 11 leaders on stage 0, 10 on stage 1
  → shared-selection stash ~5.9 GB (rank 0, ×2 in-flight) / ~2.7 GB
  (rank 8). Census gotcha: that stash's byte size is ALIASED at other
  seq/CP geometries — identify by allocation site, never size signature.
- Bandwidth (tj-wlmlkeq direct bench, huygens 2026-08-20; confirms
  campaign PCIE_OFFLOAD_BW_BENCH.md): per-GPU pinned D2H 57.3 / H2D 55.7
  GB/s solo; single-GPU bidi 48.3+48.3; all-8 unidirectional 456 D2H /
  439 H2D aggregate, NO per-GPU collapse; all-8 bidi NUMA-local
  27.5+28.8 /GPU (node ~450 total, direction-agnostic). NUMA AT 8 GPUS
  IS BINARY (resolved 2026-08-20, ALL8_BIDI_OFFLOAD_BW_BENCH.md):
  local 27.5/28.8; INTERLEAVED (the unbound-process DEFAULT) 15.6/16.8 —
  BELOW the 22/24 demand; cross-socket 7.4/7.8 (−73%, and per-GPU
  fairness breaks down under clamp). The single-GPU "<1% remote penalty"
  is real but generalizes to nothing. Pageable 5× worse.
  Host: 2× Xeon 6767P, 3.93 TiB physical, **cgroup memory.max = 2.42
  TiB** (size against this, not physical). GPUs 0–3 → NUMA0, 4–7 → NUMA1;
  NVMe RAID0 27.9 TB on NUMA1. Bench source: /tmp/cuda_host_bw.cu on box.

## The configuration

- `recompute: {granularity: selective}` (core_attn checkpointed, all else
  eager) — replaces full recompute.
- Fine-grained activation offload ON for: moe_act group (hooks exist,
  proven at 32k), the combine-side saves — **hook goes on the UNPERMUTE
  step, NOT FusedCombine** (carnot 2026-08-20: on the mission path,
  MoEFlexTokenDispatcher/DeepEP's FusedCombine saves nothing — handle
  only; a combine-site wrap would be silently inert and quietly forfeit
  the ~0.2 GiB/layer bucket) — and the attention out-proj input (HOOK TO
  BUILD; clean site). The **qkv-proj input is CONDITIONAL**: it is
  hidden_states, which the core-attention checkpoint scope may also
  save — offloading it risks either double-storage (saving illusory) or
  an H2D transfer in the critical path of core-attn recompute (defeating
  why core-attn is in the recompute bucket). carnot characterizes the
  multi-save behavior (storage-pointer identity + whether the hook layer
  dedups same-storage saves); DEFAULT POSTURE until then: offload
  out-proj only (0.2 GiB, not 0.4). Do NOT enable offload_core_attention
  in phase 1 (phase-2 A/B, ~+5%, halves PCIe margin). expert_fc1 group:
  drop from module lists (inert under LoRA — confirmed at both levels:
  LoRA target list + TE GroupedLinear frozen-weight path).
- TAIL-LAYER EXCLUSION (added 2026-08-20, banach's finding): do NOT
  offload the last K=2 layers of each stage — they are stashed last and
  needed first (backward runs layers in reverse), so their round trip is
  pure waste; on the LAST stage (0 warmup, F(mb)→B(mb) back-to-back) it
  is a serialized ~145 ms/mb round trip with only the loss path as cover.
  Cost: a few GiB. Check first whether f2407a10's `fraction`/margin
  machinery already implements trailing-group retention before writing
  new code. Rung 3's seam-window stall measurement confirms it works.
- Pinned pool: patch the MoE-groups-off-pool hardcode
  (fine_grained_activation_offload.py:362-368, OffloadTensorGroup.__init__,
  forces expert_fc1/moe_act/fused_group_mlp off the pool) — pad-to-max
  makes shapes constant, pool-safe. Freeze-exception required (precedent:
  gauss's cache-clear probe). SAME exception also carries a valve
  telemetry counter (in-flight count + drain-fire frequency, zero-cost
  when quiet): the valve BLOCKS the compute stream rather than skipping
  (undersized valve = throughput cost, never memory — safe for gate #1)
  but today logs only its configured cap once at build, so tuning would
  otherwise be trace-only. Two hunks documented separately in the
  exception so either reverts alone.
- Placement: pinned host buffers NUMA-local per huygens; large async
  copies; valve `max_inflight_offloads` tuned so stash never queues past
  ~2 layers; prefetch order within a layer's bwd = MoE tensors first
  (needed first: bwd order is MoE-bwd → attn-bwd), projections last;
  double-buffer 1 layer ahead (~5 GiB GPU reserve).

## Arithmetic (binding stage = STAGE 0: 35 MoE layers × I=2 = 70 MoE sets)

RESOLVED 2026-08-20 (banach, architectural argument): the last stage
(stage 1) is in-flight **I=1**, and S_eager stays ~2.94 — layers are
uniform MoE architecture (only layers 0–2 are dense, all on stage 0), so
per-layer set size is a layer-type property; the alternative (stage-1
S≈5.5) would contradict stage-0's clean measured slope (5.8/layer =
2×2.9). All plan figures err ~5–11% conservative as written:
- Offloaded, stage 0: 70 MoE × 1.6 + 6 dense × 0.4 ≈ 114 GiB/step-window
  (plan's 128 conservative); stage 1: 40 × 1.6 = 64 GiB.
- GPU-stored, stage 0 (binding): 76 sets × 0.63–1.08 ≈ 48–82 GiB.
- CPU pinned is ASYMMETRIC: node 0 ~0.9 TiB, node 1 ~0.5 TiB (vs 2.42
  TiB cgroup each) — size and NUMA-split per node, not per the old
  uniform 1.0 figure.
- Per-GPU stash/prefetch rates UNCHANGED (~22/24 GB/s — per-layer rates).
Stage 1 has HALF the memory pressure AND the F→B tail hazard → it needs
offload least where its timing is tightest — strengthens the K=2 tail
exclusion; consider being generally more conservative on stage 1.
Rung-1 census (per-rank, per-layer-type) confirms rather than decides.

- Offloaded: ~1.6 GiB/layer-mb (1.2 MoE + 0.4 proj) → stash ~22 GB/s
  during fwd (72 ms/layer), prefetch ~24 GB/s during bwd (~66 ms/layer
  incl. retained core-attn recompute). Node ~180 GB/s vs 440+ measured.
- BANDWIDTH OPERATING MODEL (clarified 2026-08-20 after banach's
  challenge): non-interleaved 1F1B serializes F and B phases per rank
  (trace-verified) and the node is CP-lockstep, so the SUSTAINED regime
  is all-8 UNIDIRECTIONAL (measured 55–57 GB/s/GPU, no collapse →
  ~2.4× margin). Bidirectional occurs only at phase seams
  (prefetch-ahead during F tail + stash drain into early B): ~60–130 ms
  per seam at 1–2-layer double-buffering ≈ ~5% duty. The 29/dir
  all-8-bidi figure is the seam/naive-scheduler regime, NOT the design
  point — but it assumes a phase-disciplined scheduler (valve-enforced,
  zero hardware evidence). PP p2p rides the same PCIe switch but
  ~0.1 GB/s — negligible.
  MEASURED (banach's bench, 2026-08-20, tj-wlmlkeq, cross-validated
  against huygens within 0.6% — full table
  results/ALL8_BIDI_OFFLOAD_BW_BENCH.md): the model HOLDS. Unidirectional
  56.8/54.6 per GPU vs 22/24 demand (2.3–2.6×). All-8 bidi NUMA-local:
  27.5 D2H + 28.8 H2D per GPU — the seam regime ALSO meets demand
  (paced run at exactly 22/24 achieved 99.8/99.5% of target, 0.75 ms
  median latency, no standing queue). Structural fact nobody's model
  had: at 8 GPUs the node total is ~450 GB/s REGARDLESS of direction mix
  (directions split the ceiling, unlike single-GPU where they nearly
  add) — seam capacity is inherently ~half of unidirectional. VERDICT:
  phase 1 proceeds, no design change. Phase-2 core-attn offload is
  CONDITIONALLY alive (doubled demand ~44/48 fits unidirectional, not
  the seam) — strictly dependent on phase-disciplined scheduling, judged
  by rung 3's exposed-stall measurement.
- GPU-stored: 0.19 input + 0.44–0.89 glue ≈ 0.63–1.08 → 50–86 GiB + ~5
  prefetch buffers + 20 burst vs ~85–100 headroom: OK at census-mid,
  TIGHT at census-hi → gate #1 (census boot) decides; contingency =
  offload the largest glue tensors.
- Retained recompute: core-attn ~5–7 s/step. Projected step ≈ 94 s ≈
  1,400 tok/s/GPU (+40%); +5% more available in phase-2 attn-offload A/B.
- CPU pinned: 80 × 1.6 × 8 ranks ≈ 1.0 TiB/node, NUMA-split ~512
  GiB/socket, vs 2.42 TiB cgroup — OK, leave room for dataloader.

## Validation ladder (Jack's calls, 2026-08-20)

Standard for every rung: **131k seq len, d2 (2 microbatches)** — 1F1B holds
max 2 microbatches in flight regardless of M, so d2 peak memory = d16 peak
memory (campaign-verified; the d2→d16 gap was step-position allocator
creep). Run each leg past the memory plateau (~+21 GiB over early steps) or
the peak reads low. Tool: `lps_1062_perf/tools/profile_driver_new.py`
(warmup → 1 kineto-traced step → untraced control windows) — measures
throughput + max memory. **Traces on ranks 0 AND 8** (both stage leaders):
NO driver change needed — cherry-pick trainer commit `ad39a97d`
(BT_PROFILE_RANKS env, default rank-0-only) into the stack and set
`BT_PROFILE_RANKS=0,8`. The driver's memory_profile mode writes
`memory.rank<N>.pickle` allocator snapshots on ALL ranks.

1. Baseline census boot (current full-recompute config, pre-offload):
   memory_profile ON → all-rank pickles close the glue census row (gate
   #1; no PP2 allocator snapshot exists today). Also banks the d2
   baseline throughput/peak for the A/Bs. Run the baseline arm TWICE
   (same seed, same data order) — the repeat establishes this box's
   within-boot noise floor for the numerical gates (campaign floor
   3.7–5.5 per-token is the prior, not the bar).
2. RESCOPED 2026-08-20 (godel found rung 2 as originally written cannot
   run: selective-only at 131k OOMs — rank 0 projected 292 GiB vs 247.7
   effective ceiling, corroborated by E1's historical 258.8 OOM. The
   ladder previously contradicted the plan's own conclusions section on
   this; fleet caught it before a boot was burned.)
   **2a. Selective-recompute-only at 32k** (fits with wide margin) —
   TWO gates: (a) memory must match the model's 32k prediction (with
   rung 1's 131k census this gives two points on the memory-vs-seqlen
   line — stronger than one); (b) NUMERICAL: per-token loss vector +
   aggregate grad-norm vs a matched 32k full-recompute baseline arm,
   same seed/data/steps, noise-relative vs a repeat-pair floor.
   Correctness of the scope switch is a per-layer property — sequence
   length is irrelevant to it — so 32k isolates the variable cleanly;
   32k chosen over 64k for the precedent link to the prior
   selective+offload trial. Precondition from carnot's code read
   unchanged: if shared layers RECOMPUTE indices rather than saving
   them, a Mac unit test pins correctness BEFORE this boot.
   **2b (conditional). Deliberate selective-only boot at 131k to
   confirm the OOM lands at the predicted number** — run ONLY if rung
   1's measured census leaves the projection ambiguous; if run,
   pre-register the predicted OOM figure first. Costs one failed step.
   **2c. Full phase-1 offload configuration at 32k — a BRING-UP rung,
   not a measurement** (added 2026-08-20, banach): eight never-booted
   pieces (valve, pool fix, in-allocator NUMA binding, placement
   verification, valve counters, new module vocabulary + validator,
   unpermute hook, projection hook) must not all first-boot at 131k
   prices. Pass = it boots, every hook shows engaged in the build-time
   probe log, placement verification reports NUMA-local, valve counters
   report sane values, nothing crashes. EXPLICITLY not a throughput
   claim — 32k throughput says nothing about the 131k win and must not
   be reported as if it does. Scope honesty: 2c de-risks CRASHES and
   WIRING only; the valve's real backpressure and the NUMA saturation
   penalty only engage at 131k/all-8 load — tuning stays at rung 3.
   TRIPWIRE (observational, not a gate): the prior 32k trial measured
   −30% step cost from pinned allocation at this exact length — if the
   pool fix does not visibly recover most of that here, STOP and
   investigate before any 131k arm; the pool premise is load-bearing.
   2c also smoke-runs the instruments themselves (census tool, trace
   classifier incl. the new Memcpy copy class, both-rank tracing).
   Risk downgraded (carnot, 2026-08-20): the prior 32k trial
   (trainer_pp2cp8ep8_32k_selective_offload_novpp.json) already ran
   selective recompute WITH offload to completion, and indexer group
   mechanics are seq-len-independent — so the combination is exercised,
   not novel. Caveats keeping the gate: that trial was throughput-only
   (never parity-checked — silent mis-attention runs to completion), and
   its module list included core_attn, which ours excludes. The gate is
   now confirmation of a likely-benign path, not a fishing expedition.
3. Offload arms at 131k, one variable at a time: moe_act → +combine →
   +proj; matched-step/plateau memory reads; throughput A/B per arm vs
   rung-1 baseline. NOTE: arm 1 (selective + moe_act offload) is the
   FIRST configuration of the package that runs at mission length at
   all — treat its first clean step as a milestone, not an increment.
4. Parity: NOISE-RELATIVE bars only — the base path is intrinsically
   nondeterministic (per-token floor 3.7–5.5); never import 1e-6/1e-3.
   DSA tests green + canonical env block before anchors.
5. Headline anchor — a MATCHED PAIR, not a comparison against history
   (rescoped 2026-08-20, banach): TWO d16 runs on the IDENTICAL tree —
   full-recompute baseline and the phase-1 configuration — and the
   **RATIO is the headline deliverable**. Rationale: the old "record-
   comparable number" quietly depended on TF32 (PR 995 — backlogged by
   Jack, no owner on this stack) and Q2 being on the tree, plus
   archaeology on which historical figures included TF32. Matched-pair
   makes TF32/B-F/wheel/box differences cancel; absolute tok/s/GPU
   figures are reported as context only, not load-bearing. (d16 not d2
   because d2's pipeline bubble is 33% of the step vs 6%, understating
   the win; memory is identical.) **SETTLED BY JACK 2026-08-20: "TF32.
   I don't want that." The TF32 port is CANCELLED, not deferred; the
   absolute record-comparable number is NOT a deliverable of this
   workstream — never produce one, never compare our figures to the 984
   or 1089–1103 historical bands. The whole ladder runs one consistent
   no-TF32 tree.** Side benefit: rung 1's d2 baseline + this d16
   baseline give an M-scaling check on one tree.

Cautions: BT_SAVE_STATE_SYNC=1 is a NO-OP until queue item Q1 merges —
avoid /save_state until then. Per Jack (2026-08-20): NO per-PR subagent
reviews for this stack — build it, measure it.

## Pointers

- Evidence notebook: `analysis/notebooks/recompute_cost_analysis.ipynb`
- Campaign benches: `results/PCIE_OFFLOAD_BW_BENCH.md`,
  `runs/overnight_20260813_overlap_campaign/MEMORY_LEG_DECISION.md`
- Queue/ledger: `pr_shaping/MORNING_MERGE_QUEUE.md` (Q1 save-toggle, Q2
  B/F caches precede or ride alongside this stack)
- Trace: `~/perf_profiles/lps-1062/pp2cp8ep8/fe127_d16_rank0.pt.trace.json`
