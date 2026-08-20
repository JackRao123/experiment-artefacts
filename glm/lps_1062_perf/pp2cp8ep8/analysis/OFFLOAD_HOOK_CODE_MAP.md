# Offload Hook Code Map — LPS-1062 Activation Placement, Deliverable 1

Author: carnot (Kimi-K3), 2026-08-20, for banach (orchestrator). Read-only
code archaeology; no code was written or modified.

**Status 2026-08-20 (later same day):** Deliverable 2 (pool/NUMA/valve/NVTE
patch + SPEC) and Deliverable 3 (the two hooks + engagement-probe extension +
hooks SPEC) are in `../patches/`. §c carries a post-script with the TE-source
finding that the combine window captures ~0 bytes on the current path (the
census row was shape-derived). §d's proj-in hook is implemented; qkv-in stays
deferred (the third-placement design is recorded in §d and deliberately left).

**Source of truth:** vendored Megatron-Core at
`server-interface/vendor/megatron-bridge/3rdparty/Megatron-LM`, submodule pin
`57efae08b` ("fix(te): pass pad_between_seqs explicitly for tail-padded THD
under CP (#25)"). This matches the campaign-documented box pin (serre's
DEEP_EP_CONFIG_AUDIT: "mcore 57efae08b"). Direct hash-check against the
box's installed copy was attempted twice; the training container's
site-packages is not reachable from the box's host shell in a quick probe.
Residual: the freeze-exception process's hash-before step (gauss precedent)
will close this at Deliverable 2 time. All line numbers below are the pin
above.

Terminology used throughout: **offload** = copying a GPU tensor to pinned
(page-locked) CPU memory during forward and streaming it back during
backward. **Hook site** = a place in model code where the offload machinery
is wrapped around a computation so that everything autograd saves inside it
is captured. **Group** = the set of tensors captured between one group-start
mark and one group-commit mark, offloaded/reloaded as a unit.
**Force-release** = manually zeroing a tensor's GPU storage after its D2H
copy is enqueued, for tensors Python still references.

---

## (a) Pinned pool: the MoE-groups-off-pool hardcode

**Location confirmed with a small correction to the plan's citation:**
`megatron/core/pipeline_parallel/fine_grained_activation_offload.py`,
lines **362–368** (not 362–366), in `OffloadTensorGroup.__init__`:

```python
        # Using memory pool is for the compatibility with cuda graph.
        # Shapes of tensors for MoE activation offload groups are not known in advance,
        # so we do not use CPU pool for them.
        if name in ("expert_fc1", "moe_act", "fused_group_mlp"):
            self.use_cpu_pool = False
        else:
            self.use_cpu_pool = True
```

**What it does in plain English.** Every offload group carries a
`use_cpu_pool` flag. When a group is committed, `ChunkOffloadHandler.offload`
(lines 831–847) either draws the destination CPU buffer from the shared
`OffloadTensorPool` (a shape/dtype-keyed free-list of pinned buffers,
allocated once and reused across layers and steps) or — for the three MoE
names — calls `torch.empty(shape, device="cpu", pin_memory=True)` **fresh,
for every tensor, of every layer, of every microbatch**. Pinned allocation is
a `cudaHostAlloc` under the hood: slow, and synchronizing. The one 32k
offload trial lost ~30% of its step to exactly this (jacobi's decomposition
in MEMORY_LEG_DECISION §1: ~0.6 s of copies vs ~2.1 s of alloc+sync residual
at 32k). The pool's reuse path is what the other (attention) groups get.

**Why NVIDIA wrote the hardcode:** the comment gives two reasons — CUDA-graph
compatibility, and "shapes not known in advance." Neither binds us: we run
no full-iteration CUDA graphs (jacobi, MEMORY_LEG_DECISION §1), and see the
shape analysis below.

**Is the pad-to-max claim true?** Precisely stated: MoE tensor shapes on our
config are **not inherently constant** — they vary with routing — and the
patch must *make* them constant. Evidence:

- The mission runs `MoEFlexTokenDispatcher` with the DeepEP backend
  (`moe_token_dispatcher_type="flex"`, `moe_flex_dispatcher_backend="deepep"`;
  bridge mapping in `megatron_config.py:43-56`, audited in
  `results/DEEP_EP_CONFIG_AUDIT.md`). No `moe_expert_capacity` is set in any
  mission config JSON → no drop-and-pad → the permuted expert-input token
  count is routing-dependent per layer per microbatch.
- Training is bf16 (the bridge *dequantizes* the FP8 checkpoint;
  `glm5_bridge.py:327-335`), so `config.fp8` is false and the
  quantization-padding path in `experts.py` is inactive — no alignment
  rounding either.
- The plan's own census proves variability: `moe_act 0.75–1.15 GiB/layer/mb`
  and `combine ~0.20–0.25` are *ranges*, i.e. shape jitter across
  layers/microbatches.

The pool itself tolerates multiple shapes (keys are `(shape, dtype)`), but
with routing jitter it would accumulate a long tail of pinned buffers over
thousands of steps (per-iteration `reset()` marks all free but never
shrinks). So the Deliverable-2 patch should not merely flip the flag; it
should **pad allocations to a constant maximum shape** (hard upper bound
exists: every token in the EP group routing to this rank's experts) or to a
coarse bucket, so each group sees one pool key.

**Gotcha for the patch author (Deliverable 2):** `OffloadTensorPool.free`
(line 226) validates membership by *identity* (`any(tensor is t for t in
pool['all'])`, line 252) and keys by `tensor.shape`. A narrowed view of a
padded buffer is a new Python object and fails the identity check. The patch
must therefore pool the *padded* tensor and carry the real shape in the
offload `state` tuple (or key pools by padded shape), and `reload` must copy
only the real region or H2D bandwidth is wasted on padding.

---

## (b) The `off_interface` API and the moe_act pattern end to end

`FineGrainedActivationOffloadingInterface`, aliased `off_interface`
(`fine_grained_activation_offload.py:1356`). The usage contract, exactly as
the working MoE hooks in `megatron/core/transformer/moe/experts.py` do it
(lines 701–712 for `expert_fc1`, 793–812 for `moe_act`):

```python
mgr = off_interface(flag, tensor, name)   # 1. group-START identity op applied to tensor
with mgr as tensor:                       # 2. saved-tensor hooks ACTIVE inside this block
    ... compute ...
out = mgr.group_offload(out, forced_released_tensors=[...])  # 3. group COMMIT
```

Mechanics:

1. **Registration.** Entering the `with` enters the singleton
   `PipelineOffloadManager`'s `saved_tensors_hooks` context
   (`on_save_for_backward` / `on_get_saved_tensor`, lines 807–822). Every
   tensor autograd saves in the dynamic extent of the block is swapped for a
   `(group_index, position)` tag and recorded in the current
   `OffloadTensorGroup`. Filters: not a `Parameter`, not a fake/functional
   tensor, must be on CUDA, `numel() >= min_offloaded_tensor_size` (default
   1M elements), and neither `_TE_do_not_offload` nor `_do_not_offload`
   (`tensor_need_offloading_checker`, lines 986–1000; `mark_not_offload` is
   the explicit opt-out, line 772).
2. **Commit.** `group_offload` inserts an identity autograd Function whose
   forward calls `on_group_commit_forward`: the D2H stream waits on the
   compute stream, then every registered tensor that passes the checker is
   copied out (`bulk_offload_group`, lines 1002–1026) and an event recorded.
3. **Force-release discipline.** Tensors listed in `forced_released_tensors`
   get `record_stream` + `untyped_storage().resize_(0)` right after the D2H
   enqueue (lines 1099–1106) — this is what actually frees GPU memory when
   the tensor is still referenced elsewhere (e.g. a Python variable in the
   forward scope). The docstring at line 1253: "specify the tensors only when
   they are not automatically released by torch gc."
4. **Reload.** In backward, the commit node's backward
   (`on_group_commit_backward`) makes the compute stream wait on the group's
   reload event; the start node's backward (`on_group_start_backward`)
   triggers the *next* group's H2D (pipelined one ahead, plus
   `pre_reload_last_layer`). The unpack hook (`tensor_pop`, lines 972–984) is
   the correctness floor: if a group's bulk reload hasn't replaced the state
   tuple yet, it reloads **synchronously on the compute stream** — correct,
   but an H2D copy on the critical path of whatever triggered it.
5. **Per-microbatch driver.** `GPTModel.preprocess_for_fine_grained_offloading`
   (`gpt_model.py:482`) calls `init_chunk_handler` at each microbatch forward
   and marks all parameters non-offloadable. Iteration boundary:
   `PipelineOffloadManager.reset()` resets the pool and chunk state.
6. **Warmup and policy.** The first iteration is discovery: all groups
   offload, byte counts are collected, then `post_warmup_callback`
   (lines 566–650) applies policy — see item (h) below — and prints a
   per-rank per-group MB table (`print_offload_summary_table`). **This table
   is the runtime verification that a new hook captured what we think it
   captured.**

**moe_act end to end** (the proven pattern): group-start on `fc1_output`;
the `with` block wraps the activation function; commit is *delayed until
after* `linear_fc2` with `forced_released_tensors=[fc1_output]`. When
`moe_act` *recompute* is also on, the `CheckpointWithoutOutput` sits **inside**
the offload window: the checkpoint's `save_for_backward(fc1_output)` fires
the pack hook, so the checkpoint's input is what gets offloaded, and the
comment at experts.py:805-807 documents the ordering requirement — the commit
must be downstream of the checkpoint node so the reload-wait fires before the
recompute reads the input. `CheckpointWithoutOutputFunction.backward`
(random.py:683-697) caches the unpacked inputs (`ctx.inputs`) explicitly "to
avoid double-reloading the inputs in CPU offloading scenario."

---

## (c) Hook site 1 — dispatcher combine

**Live path for our config (EP8, ETP1, flex+deepep):** `MoELayer.forward` →
`routed_experts_compute` (`moe_layer.py:530-556`): `dispatch_postprocess` →
`experts(...)` → `combine_preprocess(expert_output)`; then `combine()` →
`token_combine`; then `postprocess` → `combine_postprocess`.

On the flex/deepep dispatcher (`MoEFlexTokenDispatcher`,
`token_dispatcher.py:1666`):

- `combine_preprocess` (line 1835) → `_DeepepManager.get_restored_hidden_states_by_experts`
  (line ~1396) → **`unpermute(...)`** (`moe_utils.py:432`; with
  `moe_permute_fusion=True` — set on our provider — this is TE's
  `fused_unpermute`).
- `token_combine` (line 1844) → `_DeepepManager.combine` → **`fused_combine`**
  (`fused_a2a.py:165`).
- `combine_postprocess` (line 1868): shared-expert add + reshape.

**Correction to the plan's mental model:** `FusedCombine.forward` saves
**nothing** for backward — only the DeepEP handle and flags
(`fused_a2a.py:169-191`); its backward re-dispatches the gradient. So a hook
wrapped around the combine *communication* would capture nothing and appear
to work while moving zero bytes. The combine-side saved state lives in the
**unpermute** (the permutation-reversal autograd op; with merging probs it
must save its input/probs operands for backward). The census's
"combine ~0.20–0.25 GiB/layer/mb" is the expert output held at this seam.

**Exact wrap point:** `MoEFlexTokenDispatcher.combine_preprocess` — group
start on the incoming `hidden_states` (= the routed experts' output), `with`
block around `get_restored_hidden_states_by_experts`, commit on the restored
output. One `off_interface` wrap, as the plan says — but at the *unpermute*,
not at the a2a. The group rides the new `"moe_combine"` name (Deliverable-2
hunk 5 legalizes it in `transformer_config.py`; jacobi's `b6894e56` mirrors
trainer-side).

**⚠ Post-script (Deliverable 3, TE-source-verified): the window is expected
to capture ~0 bytes on the current path.** TE's `moe_unpermute`
(`permutation.py`, baseten TE checkout 79d44a2f) saves by variant:
mask-map WITH merging probs → `inp, row_id_map, merging_probs, pad_offsets`;
mask-map NO probs (**ours**: probs are applied inside the experts, so the
combine unpermute is called with none) → `row_id_map, pad_offsets` only;
index-map → always saves `inp`. Our saves are sub-megabyte index tensors,
below `min_offloaded_tensor_size`. The census's "combine ~0.20–0.25
GiB/layer/mb" was *shape-derived* (notebook cell 8), not measured against
this dispatcher/TE path — it models a save this path does not make. The
dispatcher seam's big tensors (dispatched tokens, expert output, ~1.6
GiB/layer/mb at 131k) are transient: nothing saves them for backward, so
they are freed in forward and never resident. Consequence: the phase-1
offload total is moe_act + proj-in ≈ **0.95–1.35 GiB/layer/mb, not 1.6** —
the fit arithmetic needs a re-derivation. The hook ships anyway: it is ~free
when empty, makes the emptiness loud at the 32k bring-up (warmup table row +
engagement probe), and auto-captures the combine input if a future
path/TE variant saves it. No `forced_released_tensors` (the caller's
reference dies at return; force-release would be unsafe for anything not
actually offloaded).

**Live-path confirmation:** flex requires `tp_size * ep_size > 1` (1×8 = 8 ✓);
the dispatcher is per-MoE-layer, so it fires on every sparse layer and
correctly does NOT fire on the 3 dense layers (see note in §g-0).

---

## (d) Hook site 2 — attention projection inputs on AbsorbedMLA

GLM-5.2's attention is `GlmAbsorbedMLASelfAttention`
(`experimental_attention_variant/glm_absorbed_mla.py`), a thin LoRA-folding
subclass of `AbsorbedMLASelfAttention` (`absorbed_mla.py:125`). Its `forward`
(line 796) does **not** pass through `attention.py`'s forward, so the
existing `qkv_linear` (attention.py:1347) and `attn_proj` (attention.py:1577,
also `multi_latent_attention.py:461`) hook sites are inert on this model —
the plan's blocking fact, confirmed.

Forward structure (absorbed_mla.py):

1. `get_query_key_value_tensors(hidden_states, ...)` (line 832): q/kv
   down-projections, norms, then `qkv_up_proj_and_rope_apply` (itself
   optionally checkpointed when `"mla_up_proj"` is in `recompute_modules`).
2. Core attention: `self.core_attention(q_absorbed, kv_compressed, None, ...,
   x=hidden_states, qr=q_compressed, up_v_weight=v_up_weight, ...)` — the DSA
   sparse-attention module (`DSAttention`, dsa.py:1530), which takes the raw
   layer input `hidden_states` and the low-rank query `q_compressed` as extra
   inputs for its indexer. When `checkpoint_core_attention` (selective
   recompute with `core_attn`, our plan) this whole call is wrapped in
   `tensor_parallel.checkpoint` (`_checkpointed_attention_forward`,
   lines 742–794) whose saved inputs are: `q_absorbed, kv_compressed,
   hidden_states, q_compressed, attention_mask(None for us), up_v_weight,
   attn_mask_type-as-int-tensor`.
3. `_apply_absorbed_v_up_projection(...)` — **a no-op on our path**:
   `DSAttention.consumes_absorbed_v_up_projection = True` (dsa.py:1539), so
   the V up-projection is fused into the kernel and this helper returns its
   input unchanged (lines 75–108). No extra saved tensor here.
4. `output, bias = self.linear_proj(core_attn_out)` (line 908) — the
   out-projection.

**Out-projection input (proj-in, 0.20 GiB/layer/mb): CLEAN SITE.**
`core_attn_out` is a single, unshared tensor consumed only by
`linear_proj` (and, when `mla_up_proj` recompute is on, carrying a grad hook
— data-independent). Under core-attn recompute, nothing in the checkpoint's
backward needs the original `core_attn_out` (recompute regenerates its own).
A hook exactly mirroring attention.py:1577-1580 —
`off_interface(flag, core_attn_out, "attn_proj")` around the `linear_proj`
call, commit on `output` with `forced_released_tensors=[core_attn_out]` — is
safe. One wrinkle: the config guard at `transformer_config.py:1830-1836`
forbids `"attn_proj"` without `"core_attn"` in `offload_modules` (its
rationale — attn_proj's input is needed by core_attn.backward — does not
hold when core_attn is *recomputed*). Either relax the guard (frozen-tree
touch) or register under a new name (also a frozen-tree touch, via
`allowed_modules`). Same exception bundle as the pool patch.

**QKV-projection input (qkv-in, 0.20 GiB/layer/mb): NOT CLEAN — recommend
deferring.** The analog tensor is `hidden_states` entering
`get_query_key_value_tensors`, but it is multiply-saved:

- by the `linear_q_down_proj` and `linear_kv_down_proj` LoRA adapters
  (attention projections are LoRA'd — `lora_targets.py` dsa branch — and the
  adapter's input saves fire for adapter wgrad),
- and by the core-attention checkpoint as a saved input (the DSA indexer
  needs `x=hidden_states` at recompute time).

What the machinery does when a tensor is both offload-registered and saved by
an enclosing checkpoint (this was banach's direct question):

- **No dedup.** Each save inside an active offload window gets its own tag →
  its own D2H copy and its own reload. Same tensor saved twice in-window =
  two copies. Correct, duplicated traffic.
- **A save outside the window pins the tensor on GPU.** The attention.py
  placement (qkv context closes before the checkpoint call) would make D2H
  copies of `hidden_states` while the checkpoint keeps it resident: bandwidth
  spent, **zero memory freed** — the 0.20 GiB qkv-in saving is illusory
  there.
- **Extending the window over the checkpoint call** captures the checkpoint's
  save too, but then the group's commit (on the qkv output) is *upstream* of
  the checkpoint node in backward order, so the reload-wait fires too late
  and the recompute's read of `hidden_states` falls to the synchronous
  pop-time reload — an H2D copy parked on the critical path of the one bucket
  (core attention) we deliberately kept PCIe-free.
- A third placement (one window spanning `get_query_key_value_tensors`
  through the core-attn checkpoint, commit on `core_attn_out`, with
  `mark_not_offload` on the big checkpoint inputs `q_absorbed` /
  `up_v_weight` / rotary) satisfies the commit-downstream rule and moves only
  hidden_states-class bytes — but it offloads part of the checkpoint input
  set, needs a maintained mark-list, and eats ~2× D2H duplication on
  `hidden_states` (adapter saves + checkpoint save). Real design work, sharp
  edges.

**Recommendation (matches banach's flagged fallback):** phase 1 hooks =
proj-in only (0.2 GiB). qkv-in is a phase-1.5 design item, not a hook.

---

## (e) expert_fc1 is inert under LoRA — confirmed at both levels

1. **Train-side:** GLM-5.2's LoRA target list (`lora_targets.py`, `dsa`
   branch) covers attention projections, dense first-k MLP, shared experts,
   LM head — and deliberately **excludes** routed `*.mlp.experts.linear_fc{1,2}`.
   Routed-expert weights have `requires_grad=False`.
2. **TE-side:** with frozen weights, TE's grouped linear saves no input
   (wgrad never fires; dgrad needs only the weight) — jacobi source-verified
   against TE v2.16 (MEMORY_LEG_DECISION §2), and consistent with the
   observed 0.00 GB for the `expert_fc1` group in the 32k trial.
3. Note the residual effect: `experts.py:271-274` calls
   `set_save_original_input(self.linear_fc1)` when `offload_expert_fc1` —
   harmless when nothing is saved, but another reason to simply drop
   `expert_fc1` from `offload_modules` as the plan says.

---

## (f) Selective-recompute × offload interaction

**What the one working trial ran:** `configs/trainer_pp2cp8ep8_32k_selective_offload_novpp.json`
— `recompute.granularity = "selective"` (default `recompute_modules =
["core_attn"]`) **plus** `activation_offload.modules = ["core_attn",
"attn_proj", "expert_fc1", "moe_act"]`. So offload × selective-core_attn **is
the exercised combination** — but only `moe_act` actually fired (attn hooks
inert on AbsorbedMLA per §d; `expert_fc1` empty per §e). Both offload boots
ran tree `6d8b22da`, **pre-valve** (MEMORY_LEG_DECISION §1): the
`max_inflight_offloads` backpressure valve has zero hardware evidence —
the plan's gate #4 stands.

**Nesting rules** (from code + config guards):

- **Offload-inside-checkpoint = broken.** If a region containing
  `off_interface` calls is itself checkpointed (e.g. `"moe"` in
  `recompute_modules`), the original forward runs under `no_grad` (hooks
  don't fire, groups stay empty) and the backward recompute re-runs the
  `off_interface` calls — group indices double-increment and commit/backward
  nodes get created at recompute time. mcore's own guard
  (`transformer_config.py:1838-1847`) rejects exactly this with "redundant
  and will cause errors."
- **Checkpoint-inside-offload-window = supported**, with the
  commit-downstream ordering rule (§b, moe_act pattern).
- **Our plan's config is clean:** `recompute_modules = ["core_attn"]` only.
  The core-attn checkpoint contains no offload calls (its custom_forward is
  just `self.core_attention(...)`), and the offload groups (moe_act, the new
  moe_combine, attn_proj) contain no checkpoints. The regions are disjoint —
  no tensor is both offloaded and recomputed. The only place that changes is
  if qkv-in is ever attempted (§d).

---

## (g) Shared-indexer index lifetime (the gating item) — SAVE-BY-REFERENCE, never recompute

GLM-5.2's DSA sparse attention computes its top-k token selection only on
"leader" layers (`dsa_indexer_topk_freq = 4` from the checkpoint;
`glm5_bridge.py:140-145`); the next three "shared" layers reuse the leader's
indices. `is_dsa_skip_topk_layer` / `source_dsa_compute_layer`
(dsa.py:37-56) assign roles at init.

Findings, answering banach's (a)/(b)/(c):

- **(a) A shared layer's core-attn checkpoint saves the usual input list
  (§d.2) — the indices are NOT in it.** The indices arrive via a Python-side
  "holder" dict keyed by layer number, attached to a per-microbatch "carrier"
  object — `packed_seq_params` on our THD path
  (`_get_index_share_carrier`/`_get_index_share_topk_holder`,
  dsa.py:1593-1612; carrier built fresh per microbatch in
  `packer.py:158`). The leader writes `topk_holder[self.layer_number] =
  topk_indices` during forward (dsa.py:2213); the shared layer reads
  `topk_holder[self.source_layer]` (dsa.py:1962-1977). The reference chain is
  `checkpoint closure → packed_seq_params → holder → indices tensor`, alive
  until the microbatch's backward completes.
- **(b) A shared layer CANNOT recompute locally — `self.indexer = None` on
  shared layers (dsa.py:1581-1585).** There is no local-recompute branch and
  no unplanned cost. A missing holder entry raises a loud `RuntimeError`
  naming layer, source layer, and freq (the cross-PP-straddle guard — the
  code reason the 38/40 split must land on group boundaries). Failure is a
  crash, never silent mis-attention.
- **(c) Today's mechanism under full recompute is exactly this holder**, and
  it is granularity-agnostic: entries are written in forward (leader before
  any shared layer) and never popped anywhere in the tree; reverse-order
  backward (S3,S2,S1 before L) is satisfied because the entry outlives the
  whole microbatch. The leader's own backward recompute re-writes the same
  key after all shared layers are done — self-consistent. Selective
  recompute retains *more*, not less; the indices' lifetime is untouched
  either way. **banach's prior confirmed mechanistically.**
- **Offload-immunity:** the indexer runs under `torch.no_grad()` on detached
  `x`/`qr` (dsa.py:1888-1889) inside the core-attn checkpoint, so saved-tensor
  offload hooks can never register or force-release the indices.
- **Size:** `[1, 16384 q-rows/CP-rank, 2048]` int64 ≈ 268 MB per leader layer
  per in-flight microbatch (int32 kernels halve it); ~9–10 leaders/stage ×
  I=2 in flight ≈ **2.4–5.4 GB per rank, already paid today** under full
  recompute. Megabytes per layer, low-single-digit GiB aggregate, not new.
- Adjacent non-risk: `dsa_indexer_loss_coeff = 0.001` enables the indexer
  loss on **leader layers only**, and under recompute it materializes during
  the backward recompute (it requires grad-enabled). Unchanged by this plan.

**Consequence for validation:** rung 2's numerical gate is
belt-and-suspenders for this specific risk; no Mac-level precondition is
needed from the index-lifetime fork.

### g-0. Engagement note (from banach's architecture fact)

`mlp_layer_types = ['dense']×3 + ['sparse']×75`; with the 38/40 split, rank 0
holds 3 dense + 35 MoE layers, rank 8 holds 40 MoE. MoE-group and
dispatcher-combine hooks **must not** fire on layers 0–2 (no expert block, no
dispatcher) — 3 non-engaging layers on rank 0 is correct behavior. Validate
engagement on rank 8 (uniform MoE).

---

## (h) Tail-layer exclusion — existing machinery inventory

Three distinct mechanisms in `fine_grained_activation_offload.py`:

1. **Dynamic imminent-backward exclusion** (`should_bulk_offload`,
   lines 1066–1087): when the chunk being forwarded is also the next chunk in
   the backward queue, the last group of each *name* stays on GPU.
   Stage-aware by construction: fires on rank 8's forward→backward seam every
   microbatch; does not fire on rank 0 mid-forward-phase. **K=1 per name,
   zero code.**
2. **Static post-warmup margin** (`post_warmup_callback`, lines 579–596):
   per group name, one group (first microbatch's last) is marked
   non-offloadable so reload can't block the main stream. Static, global,
   one per name.
3. **`activation_offload_fraction`** (lines 608–627; config field plumbed by
   `f2407a10`'s ActivationOffloadConfig): per chunk, disables the last
   `int(eligible × (1 − fraction))` groups in reverse forward order — later
   layers first. This *is* a tail-exclusion dial, but quantized in groups and
   uniform across ranks. With 3 groups/MoE-layer (moe_act, moe_combine,
   attn_proj): rank 8 has 120 eligible groups → K=2 layers ⇒ fraction =
   **0.95**; rank 0 has 105 → exact K=2 would be 0.943. One global value
   can't be exactly K=2 on both ranks; 0.95 errs conservative on rank 0.

**Answer:** hilbert's K=2 needs no new code for the case that matters (the
dynamic rule covers the imminent-backward seam); a static K≈2 is
`activation_offload_fraction = 0.95`. Exact per-rank K=2-in-layers would be a
small frozen-tree knob — only worth it if an A/B shows the approximation
costing memory.

**Valve semantics (acknowledged, matches code):** `max_inflight_offloads`
*blocks, never skips* — `_drain_offload_pending` (lines 1110–1121) makes the
main stream wait on the oldest per-name D2H event past the cap. Per-name
queues; 0 = full serialization. An undersized cap costs throughput, never
memory.

---

## Frozen-tree touchpoints for Deliverables 2–3 (all in one freeze-exception bundle)

| Change | File:lines | Why |
|---|---|---|
| Pool hardcode + pad-to-max | `fine_grained_activation_offload.py:362-368` (+ `offload`/`reload`/`free` shape handling) | (a) |
| New group name in `allowed_modules` | `transformer_config.py:1820-1828` | (c) `moe_combine` |
| `attn_proj`-without-`core_attn` guard | `transformer_config.py:1830-1836` | (d) relax, or sidestep via new name |
| Combine hook | `token_dispatcher.py:1835` `combine_preprocess` | (c) |
| proj-in hook | `absorbed_mla.py:908` region | (d) |

Open runtime verifications: box hash of the pin (freeze-exception
hash-before step); exact byte capture of the new groups via the warmup
summary table; TE `fused_unpermute`'s precise save set (its input +
merging-probs operands — confirm at hook time via the same table).
