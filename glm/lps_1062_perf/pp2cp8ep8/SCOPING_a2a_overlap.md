# SCOPING — selective recompute + EP a2a overlap (gibbs, 2026-08-12)

**2026-08-12 22:5x UPDATE — DIAGNOSIS OVERTURNED AGAIN (maxwell trace
analysis + gibbs rank-8 aggregates):** the dominant cost is NOT EP a2a
(7-8.5s/stage, near-healthy) but **mutual PP p2p parking** (stage-0 36.2s,
stage-1 27.1s of 69s) with EQUAL compute per stage (~18.5s). Mechanism
(code-confirmed): non-interleaved 1F1B uses FUSED batched p2p
(`send_forward_recv_backward` schedules.py:2404, `send_backward_recv_forward`
:2434) which chains each stage's forward feed to the other's backward drain —
strict alternation, convoy rate m-independent (explains flat d2→d4).
`overlap_p2p_comm` is interleaved-only (:2165). **VPP2 is the structural
fix** (interleaved schedule posts p2p eagerly, fills waits with chunk work).
E1 (selective recompute) OOM'd at d1 (258.8 GiB, stage 1) and is SUPERSEDED
as the next run; block+K is the memory-safe partial if evidence wants it.

Target: attack the wait wall measured on w56lorq (stage of 59.6s with only
~18.5s compute per stage).
Baseline to beat: **589 tok/s/GPU** (Run C, tf32+shipenv, d4/524k).

## Why the naive config-only version doesn't exist

1. `overlap_moe_expert_parallel_comm=True` requires (mcore transformer_config.py
   :2612-2642, bridge comm_overlap.py:500):
   - PP>1 ⇒ **VPP mandatory** (`virtual_pipeline_model_parallel_size` must be set);
   - `recompute_granularity != 'full'` + method/num_layers None + **"moe" not in
     recompute_modules**;
   - EP>1 ✓, dispatcher alltoall/flex ✓, bf16 ✓;
   - `moe_shared_expert_overlap` must be OFF — GLM-5.2 provider default has it ON;
     the clearing hunk exists in the preserved 0e0b65a6 patch
     (`lps1062_pp2/pre_checkout_local_changes_0e0b65a6.patch`, first hunk of
     megatron_controller.py) and must be ported into `_configure_moe_provider`
     (megatron_config.py) for the alltoall+overlap combination.
2. ~~Selective recompute (`core_attn`) does not cover DSAttention~~ **WRONG —
   corrected after deeper read**: the GLM-5.2 attention path is
   `GlmAbsorbedMLASelfAttention(AbsorbedMLASelfAttention(Attention))` with
   `core_attention = DSAttention`. `AbsorbedMLASelfAttention` has its OWN
   `_checkpointed_attention_forward` (absorbed_mla.py:742) used at :843 when
   the inherited `checkpoint_core_attention` flag is set (attention.py:379:
   selective + "core_attn"). It wraps the full DSAttention call (indexer +
   sparse attention) in `tensor_parallel.checkpoint`, passing all DSA tensors
   (q_absorbed, k_compressed, x, qr, up_v_weight) as proper tensor args —
   the base-class version even comments "such as DSA's x/qr inputs"
   (attention.py:449-451). Upstream built this deliberately. mcore validate
   defaults `recompute_modules=None` → `["core_attn"]`
   (transformer_config.py:1727). **E1 is CONFIG-ONLY.**

## Experiment ladder

### E1 — selective recompute, attention-only, no overlap (no VPP) — CONFIG-ONLY
- Config: `recompute: {"granularity": "selective"}` (modules None → core_attn).
- Code: NONE — DSA core-attn checkpointing already exists upstream (see §2
  correction above).
- Effect: MoE refwd pass disappears → **-3 of 9 a2a per layer (-33% sync
  points)**; attention still recomputed (memory bounded).
- Memory: MoE intermediates now stored (~0.5-1 GB/layer/microbatch ×38 ×2
  in-flight at PP2 ≈ +40-76 GB vs full recompute). Headroom 108 GiB from the
  167/275 peak. **d1-first ramp (abort >255 GiB reserved), then d2, then d4**
  (bounded probes). maxwell EV: ~800-850 tok/s/GPU if the sync-point model
  holds; well under ⇒ model wrong, reassess before E2.
- Numerics: same math recomputed — d2 canary must sit in the SAME 12.2-12.4
  band with matching grad norms; drift = bug, stop and diagnose.

### E2a — VPP2 ALONE (interleaved schedule, full recompute) — the clean structural test
- The `recompute_granularity != 'full'` constraint binds only the overlap
  FLAG (transformer_config.py:2633), NOT the interleaved schedule — VPP2 with
  full recompute is legal and memory-known-good (169 GiB peak measured).
- Layout [18,20,20,20] (chunk starts 1/19/39/59 verified, NOTEBOOK 20:5x);
  needs the `virtual_pipeline_parallel_size` config field + layout-table
  keying (layers, pp, vpp).
- Breaks the convoy: interleaved 1F1B posts p2p per chunk and gives each
  stage other chunks' work during waits.

### E2b — E2a + `overlap_moe_expert_parallel_comm=True` (the full lever)
- VPP required ⇒ PP2/VPP2 layout **[18, 20, 20, 20]** (flat chunk list
  [pp0v0, pp1v0, pp0v1, pp1v1]; layer ranges 1-18 / 19-38 / 39-58 / 59-78;
  chunk starts 1, 19, 39, 59 all satisfy the DSA topk rule ((n-3)%4==0) ✓;
  embedding on first, loss on last). `PipelineParallelLayerLayout`
  auto-detects VPP=2 from the 4-chunk list (transformer_config.py:1933) —
  satisfies the overlap assert with no separate vpp plumbing.
- Layout table keying: `_GLM52_DSA_PIPELINE_LAYOUTS` is keyed (layers, pp) —
  a VPP2 entry collides with the non-VPP (78,2). Add a
  `virtual_pipeline_parallel_size` field to TrainerControllerConfig (default 1),
  key the table on (layers, pp, vpp), set
  `provider.virtual_pipeline_model_parallel_size` explicitly when >1.
- Port the `moe_shared_expert_overlap=False` clearing for alltoall+overlap.
- Schedule becomes interleaved 1F1B — THD per-partition call-count invariant
  still holds (deterministic packer, DP=1); variable_seq_lengths handles shapes.
- In-flight microbatches rise under VPP (more chunk-checkpoints alive) —
  folded into the same d1 ramp.

## Implementation — DSA core-attn checkpoint (the gating change)

NOT NEEDED — exists upstream (see §2 correction). E1 is config-only.
E2's code changes are limited to: (a) `virtual_pipeline_parallel_size` field
in TrainerControllerConfig + layout-table keying + provider assignment;
(b) port the `moe_shared_expert_overlap=False` clearing hunk into
`_configure_moe_provider` for the alltoall+overlap combination.

## Verification protocol (both E's)

Virgin-topology discipline: d1 memory ramp (watch peak vs 275 GiB) → d2
canary (loss 12.2-12.4, finite gn) → d4 headline. 5-10 min bounded probes;
hang signature = all-enqueued-none-completed a2a / flatlined GPU util →
stop_trainer.sh, read ALL ranks' logs, diagnose, relaunch.

## M=N pivot (2026-08-12 23:5x, gibbs paper analysis; maxwell approved)

The convoy's root cause is the runner's serialized M=1-per-partition-call
architecture, not p2p fusion (the fused steady-loop ops never execute at
M=1). **Fix: one schedule call per op with num_microbatches=len(partitions)**
— non-interleaved 1F1B finally pipelines (stage-1 bwd(k) overlaps stage-0
fwd(k+1)). No VPP/layout/schedule change; in-flight = min(M,PP)=2 checkpoint
sets (169 GiB-known-good); variable_seq_lengths handles per-partition shapes;
forward_data_store is already per-microbatch. Owner: VOLTA (runner/accounting
domain). Also reopens E2b later: M=4 satisfies the interleaved
microbatch-group constraint (VPP2 becomes stackable).

### M=N validation ladder (gibbs, per maxwell spec)

1. **d1** (degenerate M=1): sanity — single partition must behave exactly as
   today's per-partition call.
2. **d2 canary**: loss in the SAME 12.2-12.4 band + grad norms finite and
   comparable. NOTE: bitwise drift is EXPECTED (backward order changes under
   real 1F1B) — tolerance judgment; volta's parity driver is the arbiter if
   the canary is ambiguous.
3. **Memory watch**: confirm the min(M,PP)=2 in-flight checkpoint-set model
   holds (peak should stay near 169 GiB, not scale with M).
4. **Traced d4** with BT_PROFILE_RANKS=0,8: verify the mutual p2p parks
   (36.2s/27.1s) mostly vanish; maxwell's ceiling estimate ~900-1100
   tok/s/GPU before in-convoy costs.
5. Baseline for the win measurement: 589 tok/s/GPU (Run C), modulo the
   telemetry A/B result (gate-off run may shift the baseline slightly).

## Open questions

- Exact DSAttention checkpoint seam (which sub-call(s) to wrap: indexer-only
  vs whole-attention) — decide when writing the patch; whole-attention is the
  memory-correct granularity.
- Whether mcore's a2a-overlap impl (per-chunk double-buffering) interacts
  with THD variable shapes under VPP — read the overlap path if E1 lands.
- E2 numerics: shared-expert overlap OFF changes gradient add order (mcore
  docstring notes minor numerical differences) — acceptable for perf probes;
  flag for correctness sign-off before any real training use.
