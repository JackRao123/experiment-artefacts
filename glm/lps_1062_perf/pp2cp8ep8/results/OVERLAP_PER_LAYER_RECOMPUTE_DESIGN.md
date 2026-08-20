# OVERLAP + PER-LAYER RECOMPUTE — upstream design sketch

Author: serre (K3), 2026-08-13 early CDT, for Jack's morning read. Status:
design only — no code written, no GPUs touched. Every line citation was
verified against the vendored mcore at the box's exact submodule pins
(megatron-bridge 20fcf2ea, Megatron-LM 57efae08b; local copy confirmed
byte-identical at the cited lines). Sibling doc:
`EXECUTOR_CONTRACT_SCOPING.md` — the trainer-side schedule-plan contract
(prerequisite exposed by the L0 probe; wrapper-level, ~0.5-2 days).

**One-paragraph version:** the EP all-to-all overlap flag
(`overlap_moe_expert_parallel_comm`) is memory-illegal at 131k today because it
forbids whole-layer recompute outright, and mcore offers no per-layer recompute
dial. The missing piece is small: let the fine-grained schedule carry a mixed
layer stack — the first K layers of each stage run as **opaque checkpointed
layers** (whole-layer `CheckpointFunction`, no overlap, minimal memory) and the
remaining L−K layers run the existing 5-node overlapped decomposition. The
schedule machinery already tolerates heterogeneous layer shapes (dense layers
run with no-op comm nodes today) and already pairs unequal forward/backward
layer counts (`overlapped_layers = min(f, b)`, leftovers unpaired). The patch
is one new config field, one new callable builder, one branch in the plan
builder — not a rewrite. At d16 the memory math lands at K≈21 of 40, i.e.
overlap on ~47% of layers, for an estimated ~8-12% step-time prize (plus a
recompute-tax refund of the same order vs tonight's full-recompute config).

## 1. Why the flag forbids full recompute (the mechanism, verified)

`overlap_moe_expert_parallel_comm` routes the schedule to the combined-1F1B
executor (schedules.py:743 no-pipeline, :1465 interleaved), which drives each
layer as **five explicit ScheduleNodes** — attn / post-attn / MoE-dispatch /
experts / MoE-combine — built per layer by `build_transformer_layer_callables`
(fine_grained_callables.py:471; the five-way split documented at :477-482) and
assembled by `TransformerLayerSchedulePlan._build_callable_nodes`
(model_chunk_schedule_plan.py:109-174). Two structural facts follow:

1. **Backward is driven per-node, explicitly.** Each node detaches and stashes
   its own tensors (`TransformerLayerNode.detach`, fine_grained_callables.py
   :312-318) and its backward is an explicit `torch.autograd.backward` on those
   stashed tensors (`backward_impl`, :324-331), interleaved with the paired
   microbatch's forward nodes (`TransformerLayerSchedulePlan.run`,
   model_chunk_schedule_plan.py:229-297). Weight-grad compute is *placed* by
   the scheduler: `mlp.backward_dw()` at :271, `attn.backward_dw()` deferred to
   :294-295 / the final :577-580, via the `_BackwardDWWrapper`
   (fine_grained_callables.py:408-453).
2. **A whole-layer checkpoint is opaque to all of that.** Under full
   recompute, `TransformerBlock.forward` calls `checkpointed_forward`
   (transformer_block.py:624-637), which wraps layers in
   `tensor_parallel.checkpoint` → `CheckpointFunction`
   (tensor_parallel/random.py:555). The layer's backward then exists only as
   that Function's monolithic recompute-and-backward; there are no per-node
   detached tensors for the scheduler to drive, and no dw hook to defer.

That is the real reason for the asserts — not a conservative oversight. The
current validation (transformer_config.py, `if self.overlap_moe_expert_parallel_comm:`
block opening at :2612) hard-rejects every whole-layer recompute knob:

- :2632-2633 `recompute_granularity != 'full'`
- :2635-2636 `recompute_method is None`
- :2638-2639 `recompute_num_layers is None`
- :2641-2642 `"moe" not in recompute_modules`

(Correction to the E2 verdict's citations, for the record: the
selective+num_layers rejection lives at transformer_config.py:1713-1718 — same
line numbers borel cited, but the file is transformer_config.py, not
transformer_block.py. The block-method dispatch citation
(transformer_block.py:624) is right.)

What *is* legal today: `granularity='selective'` with per-module-type
recompute (core_attn/layernorm/moe_act — the E1 family). It OOM'd at 258.8 GiB
with one microbatch in flight (E2_REVISED_SPEC §1) because the MoE expert
intermediates — the bulk of the 131k activation bill — stay saved. The missing
dial is per-layer-**index** whole-layer checkpointing.

## 2. The coexistence precedent that makes this a moderate patch

The schedule plan already runs heterogeneous layers:

- **Dense layers** (our layers 1-3) carry `NoopScheduleNode()` for
  dispatch/combine (model_chunk_schedule_plan.py:162-167) — a layer with
  missing nodes is a first-class citizen.
- **Unequal forward/backward plans** are handled by the chunk scheduler:
  `overlapped_layers = min(f_num_layers, b_num_layers)` (:522), leftovers run
  unpaired in the plain loops (:542-552 backward, :554-560 forward). (borel's
  anchor (b), verified.)
- The plan builder is already per-layer: `_build_layer_schedule_plan`
  (:403-420) loops layers and calls `build_layer_callables(layer)` (:122),
  which itself dispatches on layer type (TransformerLayer vs MTP,
  fine_grained_callables.py:805-822).

So: a layer whose nodes are {one opaque checkpointed callable + four no-ops}
plugs into the existing pairing loop with **zero scheduler changes**. When it
is the forward layer, only its `attn`-slot node does work; when it is the
backward layer, the no-ops pass `b_grad` through untouched until
`attn.backward(b_grad)` fires the monolithic checkpoint backward. The
deferred-dw calls are safe no-ops if the node's `delay_wgrad_compute` is False
and its dw map is empty (`backward_dw()` early-returns,
fine_grained_callables.py:347-350; the trailing `b_layer.attn.backward_dw()` at
model_chunk_schedule_plan.py:577-580 included).

## 3. Proposed design

### 3.1 Config surface (new field; do NOT relax the existing asserts)

Add `moe_ep_overlap_checkpoint_num_layers: Optional[int] = None` to
`TransformerConfig` (name bikesheddable; "first-K per stage" semantics).

- Validation (new, additive): may be set only when
  `overlap_moe_expert_parallel_comm` is on; must be ≤ layers-per-stage;
  default None = today's behavior. **The four existing asserts
  (:2632-2642) stay exactly as they are** — the hybrid path does not use the
  block-level recompute machinery at all, so `recompute_granularity` et al.
  remain None and keep protecting the stock path.
- Rejected alternative: relax :2632-2642 to admit granularity='full' +
  method='block' + num_layers=K and reinterpret those fields as plan metadata.
  It reuses knobs but overloads their meaning (the block forward at
  transformer_block.py:624 never executes under the combined executor) and
  loosens guards on the stock path. Not worth the confusion.
- Mirror plumbing, same shape as the L1 hunk: bridge-side duplicate validator
  (comm_overlap.py:470+) learns the field; trainer `CommOverlapConfig`
  (models/src/loops_models/control.py:135, extra="forbid") learns it; the
  provider is set directly in `_build_config` (megatron_config.py pattern) —
  not through the bridge dataclass splat.

### 3.2 The dial's plug point: per-layer construction

In `TransformerLayerSchedulePlan._build_callable_nodes`
(model_chunk_schedule_plan.py:122), branch on the layer's stage-local index:

```python
if layer_local_index < config.moe_ep_overlap_checkpoint_num_layers (and is MoE):
    fwd_callables, bwd_dw_callable_map = build_checkpointed_layer_callables(layer)
else:
    fwd_callables, bwd_dw_callable_map = build_layer_callables(layer)   # today
```

`build_checkpointed_layer_callables(layer)` (new, ~100 lines in
fine_grained_callables.py) returns:

- one real callable for the `attn` slot: pull `attention_mask /
  rotary_pos_emb* / packed_seq_params / sequence_len_offset` from
  `node.chunk_state` exactly as the stock attn callable does
  (fine_grained_callables.py:583-588), then run the **whole layer** under the
  same checkpoint primitive the block path uses — `te_checkpoint` under
  fp8/fp4 config, else `tensor_parallel.checkpoint`
  (mirror recompute.py:112-126) — with the input registered via
  `node.detach()` so `backward_impl` returns the input gradient downstream
  (:324-331 protocol);
- `raise_not_implemented`-style stubs are NOT needed — return the stock
  no-op-friendly structure and let `_build_callable_nodes` install
  `NoopScheduleNode()` for dispatch/combine (existing dense-layer branch,
  :162-167) and treat `mlp` the same way for opaque layers (one-line
  extension of that branch);
- an **empty** `backward_dw` map and `delay_wgrad_compute=False` in the node's
  extra_args (:133) so every dw-deferral call site no-ops;
- the final-layernorm tail for the stage's last layer, applied *outside* the
  checkpoint (the stock path applies it in the combine node,
  fine_grained_callables.py:691-694; the block-recompute path likewise never
  checkpoints it).

**LoRA correctness trap (load-bearing for us, likely novel upstream):** the
fine-grained executor bypasses `TransformerBlock.forward`, so the bridge's
PEFT+Recompute patch (megatron-bridge peft/recompute.py:96-105) never fires on
this path. `CheckpointFunction` is a plain autograd.Function — if no input
requires grad, backward never runs and **adapter gradients are silently zero**.
The opaque callable must therefore `detach().requires_grad_(True)` the incoming
hidden states when the layer is adapter-only (same maneuver as the block-level
patch, localized to the node). Eager overlapped layers don't need it (their
backward is driven explicitly and params carry their own grad edges). The d2
canary (loss band + grad-norm comparability) is the safety net, but get this
right by construction.

### 3.3 What does NOT change

- The chunk scheduler (`TransformerModelChunkSchedulePlan.run`,
  :465-596) — untouched; pairing, leftovers, p2p postamble all as-is.
- The five-node decomposition for overlapped layers — untouched.
- Schedule-level invariants: same layer count in f/b plans (same model), so
  `overlapped_layers = min(f, b)` never misfires; `is_first/is_last_layer`
  flags flow through extra_args as today (:408-411).
- Numerics: checkpointed layers recompute with the stock primitive
  (bitwise-same as tonight's full recompute); eager layers are the stock
  overlap path. The boundary moves no math.

## 4. Memory math (131k, PP2, CP8, M=N)

Constants (measured on this program, cited where measured):

- **S_eager ≈ 2.25 GiB / layer / in-flight microbatch** — the stored-activation
  cost of one eager MoE layer at 131k. Derivation: E1 (selective core_attn,
  everything else eager) OOM'd at 258.8 GiB allocated vs the ~169 GiB
  full-recompute peak, a ~90 GiB delta over 40 stage-1 layers
  (E2_REVISED_SPEC §1). Treat as an **upper bound**: the combined executor's
  eager input-freeing (`should_free_input`, fine_grained_callables.py:46-103)
  and `ep_overlap_early_attn_memory_release` reduce it by an unmeasured amount
  (this was the honest 5% tail on borel's L0 OOM prediction).
- **S_ckpt ≈ 0.19 GiB / layer / in-flight microbatch** — checkpointed layer
  stores only its input: 16,384 tokens/rank (131072/CP8) × 6144 × bf16.
- **I = 2** in-flight microbatches per stage (M=N, PP2: min(M,PP)).
- Headroom for stored activations at d16 ≈ 96 GiB (275 GiB budget − (194 GiB
  d16 peak − ~15 GiB that full recompute's stored inputs contribute)); at d4
  the free margin is ~80-100 GiB (HANDOFF).

Constraint: `I × (K·S_ckpt + (L−K)·S_eager) ≤ headroom`, L=40 (stage-1;
stage-0 has 38).

- **d16:** 2×(0.19K + 2.25(40−K)) ≤ 96 → **K ≥ ~21** → overlap on 19/40 ≈
  **47%** of layers.
- **d4:** same I, ~80-100 GiB headroom → K ≈ 17-21 → overlap on ~48-57%.

If the eager-free machinery shaves S_eager even 25%, K drops to ~13-15 at d16
(overlap ~65%). The first prototype boot should log the true per-layer stored
delta (memory snapshot per layer index) so K is set from measurement, not this
bound.

## 5. Estimated prize

- a2a exposure today: ~25% of the d16 wall (~2 s/mb of all_to_allv at ~0%
  compute overlap — HANDOFF). The hybrid overlaps (L−K)/L of layers → ceiling
  ≈ 25% × 47% ≈ **12% of step time at d16** if pairing hides all exposed a2a
  on overlapped layers; at a realistic 60-80% hide efficiency, **~7-9%**.
- **Recompute-tax refund** (vs tonight's full-recompute config, often
  forgotten): tonight every layer pays a re-forward; hybrid pays it on K/L
  only. At ~4f-vs-3f per layer (fwd + bwd≈2×fwd + recompute fwd), dropping
  recompute on 47% of layers is worth ~12% of layer compute — same order as
  the overlap prize itself. (This refund is *also* available to the L0b
  offload path; not additive with it.)
- Net: **~10-15% step-time at d16** as the honest combined ceiling, pending
  the L3 trace decomposition (E2_REVISED_SPEC §3b) confirming the a2a exposure
  share. Even at half that, it clears the "big win" bar.
- Position vs L0b (fine-grained activation offload): L0b can overlap ~100% of
  layers but pays PCIe D2H/H2D and TE-fuser disengagement, and is
  LoRA-unvalidated upstream; this design pays extra compute on K layers
  (already paid today — free) and needs no new runtime machinery. They are
  composable (offload the eager layers, checkpoint the rest) and should be
  presented to upstream as such.

## 6. Risks (ranked)

1. **Silent-zero-adapter-grads** via the requires_grad trap (§3.2). Highest
   severity because it is silent in a short canary's loss curve at this scale
   of LR; mitigated by construction (force requires_grad) + grad-norm canary.
2. **Stream/RNG semantics of a checkpoint backward inside the schedule.** The
   recompute re-runs forward at backward time on the comp stream; RNG tracker
   fork/restore must match the stock path (use `tensor_parallel.checkpoint`'s
   own machinery — it saves/restores all RNG states, random.py:574-575). The
   schedule's event wiring already covers arbitrary per-node backward; the
   opaque node never runs on the comm stream. Medium risk, testable at 1 GPU.
3. **Memory-model surprise in the eager layers' true cost** (S_eager smaller
   than the 2.25 bound → K mis-set → OOM at ramp). Mitigated by the d1 ramp +
   per-layer memory logging on first prototype boot.
4. **MTP layer on the last stage** has its own callables
   (build_mtp_layer_callables, :719-802) and the flag already constrains
   `mtp_num_layers` to ≤1 (transformer_config.py:2652-2654). The dial must exclude the MTP slot
   (or no-op there). Low risk; one assert.
5. **Upstream acceptance risk**: new config field + new callable builder is a
   moderate API surface; the "no behavior change when unset" shape and the
   dense-layer/noop-node precedent are the selling points. The alternative
   (relaxing the four asserts) is worse for upstream, not just for us.
6. **FSDP reshard hooks** (set_fsdp_reshard_hooks, :176-214) assume the
   5-node structure; we don't run megatron-FSDP, but the opaque node should
   keep the hook no-ops safe for upstream users who do.

## 7. Effort estimate

- mcore: new config field + validation (~30 LoC);
  `build_checkpointed_layer_callables` (~100 LoC, mostly mirroring the stock
  attn callable + recompute.py's checkpoint invocation); plan-builder branch
  (~10 LoC); unit tests — plan structure with K=1 on a tiny MoE model, and
  grad-equivalence of the opaque layer vs stock `checkpointed_forward`
  (single-GPU, CPU-runnable except the GPU numerics).
- bridge: validator mirror (~10 LoC). trainer: config field + direct provider
  set (~10 LoC, the L1-hunk pattern).
- Validation: 1-GPU grad-match → box d1 memory ramp (log per-layer stored
  delta) → d2 canary (loss 12.2-12.4 band, grad norms) → d4 A/B vs 918.
- **~1 day for a branch-quality prototype; 2-3 days upstream-grade with tests
  and docs.** The design risk is concentrated in §6.1-6.2, both discoverable
  in the 1-GPU test before any box time.

## 8. Verification checklist for whoever picks this up

- [ ] Opaque layer's backward returns the correct input grad (unit test vs
      stock checkpoint).
- [ ] Adapter grads nonzero and matching stock full-recompute within tolerance
      (the LoRA trap test — adapter-only model, K=1).
- [ ] Per-layer stored-memory table at 131k d1; set K from measurement.
- [ ] d2 canary: loss band + grad-norm comparability vs the M=N baseline.
- [ ] Trace check: on overlapped layers, dispatch/combine kernels on the comm
      stream overlap the paired microbatch's compute; on checkpointed layers,
      comm stream idle (expected, not a bug).
