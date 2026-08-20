# DIAL GRAD-EQUIVALENCE + LoRA-TRAP TEST SPEC (bohr, 2026-08-13)

Validation lane for the per-layer recompute dial (P1). Build owner: jacobi.
**Interface FROZEN + pushed (2026-08-13 ~21:3x PDT)** — branches at all three
levels: trainers `jackrao/lps-1062-recompute-dial` @ `6b4dabc6` (off c30afc3e)
/ megatron-bridge @ `acbbf06a` (off 20fcf2ea) / mcore @ `06393114b` (off
57efae08b; commits e5f8643a0 dial + b907b6153 review fixes + 06393114b probe
rank field). Verified against the pushed mcore branch (this doc's contract
statements are code-verified, not transcribed):

- config field `moe_ep_overlap_checkpoint_num_layers` (Optional[int], None
  default, **0 = legal no-op**); semantics = stage-local layer index < K,
  **MoE layers only** (dense + MTP excluded; a missing `layer_local_index`
  disables the dial for that layer rather than risking a wrong assignment).
  The four existing transformer_config.py:2632-2642 asserts untouched.
- builder `build_checkpointed_layer_callables(layer)` in
  fine_grained_callables.py → returns `([ckpt_fn, None, None, None, None],
  {})`; the plan builder installs `NoopScheduleNode` for mlp / moe_dispatch /
  moe_combine (mlp MUST be a no-op node — the schedule calls
  `mlp.backward_dw()` unconditionally; `NoopScheduleNode` gained a
  `backward_dw` no-op) and forces `delay_wgrad_compute=False` for dial layers.
- the attn-slot callable runs the whole layer under the stock checkpoint
  primitive (`te_checkpoint` under fp8/fp4, else `tensor_parallel.checkpoint`
  — mirrors recompute.py); final-layernorm tail applied OUTSIDE the checkpoint
  on the stage's last layer (`node.is_last_layer`), mirroring the stock
  combine-node site.
- **LoRA-trap fix, as built:** the callable mutates the framework's
  already-detached input leaf in place (`hidden_states.requires_grad_(True)`)
  — NOT a re-`detach()` — so the input gradient keeps flowing to
  `ScheduleNode.get_grad` and upstream (jacobi's docstring; the T2 monkeypatch
  target is exactly this line).
- **PreProcessNode grad-root landmine (pre-existing, blocks ANY overlap+LoRA
  boot):** frozen embedding ⇒ `decoder_input` arrives `requires_grad=False` ⇒
  the executor's closing `pre_process.backward` raises RuntimeError at the
  first stage-0 backward. Fixed on-branch (`b907b6153`, same maneuver in
  `PreProcessNode`). **Every test/boot in this spec requires the branch tip
  (or that hunk verified present).**
- mem probe (pinned for the d1-ramp parser): env `BT_DIAL_MEM_PROBE=1` (read
  per call), one `logger.warning` per opaque layer per microbatch-forward per
  rank: `[dial_mem] rank=<r> layer=<global 1-78> kind=ckpt alloc_before_mib=<f>
  alloc_after_mib=<f> delta_mib=<f>` (delta = saved input + live output +
  last-layer tail). Eager layers NOT sampled by design — S_eager comes from
  doppler's W1b block+K sweep (zero-touch).

## 0. What is being proven

The dial runs the first-K MoE layers of each stage as **opaque whole-layer
checkpoints** inside the combined-1F1B fine-grained executor, and the remaining
L−K as the stock 5-node overlapped decomposition. The executor drives backward
per-node on detached, stashed tensors (fine_grained_callables.py:312-331); the
opaque node instead relies on `CheckpointFunction`'s monolithic
recompute-and-backward. Two things must be proven equivalent to the stock
block-checkpoint path (`TransformerBlock.forward` → `checkpointed_forward` →
`tensor_parallel.checkpoint`, transformer_block.py:624-637):

1. **Gradients are correct** — adapter grads, and the input grad returned
   downstream (serre §8 checkbox 1).
2. **The LoRA trap is closed by construction** — `CheckpointFunction` is a
   plain autograd.Function: if no input requires grad, backward NEVER fires and
   adapter grads are **silently zero**. The fine-grained executor bypasses
   `TransformerBlock.forward`, so the bridge's block-level PEFT patch
   (megatron-bridge peft/recompute.py:96-105) never fires on this path. The
   opaque callable must `detach().requires_grad_(True)` the incoming hidden
   states on adapter-only layers. This test exists to catch that class — with
   a negative control that proves the test CAN catch it (the Aug-9 inert-patch
   lesson: a test that cannot fail is not a test).

Shim-independence: this spec drives mcore directly (model + schedule plan
constructed in-test, PP1 no-pipeline executor, schedules.py:743 path). The
trainer-side contract shim is NOT exercised here — that is W2's domain
(lebesgue). No trainer, no box fleet needed: 1 GPU.

## 1. Test fixture (all tests)

Tiny MoE model, constructed twice from the same seed (test path / stock
reference path), identical inputs:

- hidden 256, ffn 512, 2 attention heads, 2 MoE layers (4 experts, topk 2),
  4-layer stack total (2 dense + 2 MoE mirrors the mission's dense-head
  structure), seq 128 tokens × batch 2, bf16.
- LoRA r=8 on attention projections ONLY; all base weights frozen
  (`requires_grad=False`), incl. routed expert fc1/fc2 — mirrors the GLM-5.2
  adapter-only regime (lora_targets.py:143-174 excludes routed experts; TE
  saves no fc1 input when the weight is frozen — MEMORY_LEG_DECISION §2).
- Config: `overlap_moe_expert_parallel_comm=True`,
  `moe_ep_overlap_checkpoint_num_layers=K` per test; `recompute_granularity /
  method / num_layers` all None (the dial path does not use the block-level
  machinery — the existing asserts must NOT fire; a fire = the patch wired the
  wrong path).
- Reference: same model, flag OFF, `recompute_granularity='full'` (stock
  block checkpoint), identical seed + inputs.
- Determinism: same seed, same data, `torch.manual_seed` per construction;
  single GPU; no dropout except T3.

## 2. Test matrix (all 1-GPU, on-box or Linux-GPU CI; runtime bound < 5 min total)

Two tiers, per jacobi's harness note (2026-08-13): **T1–T3 drive the callable
STANDALONE** (no executor — mock-node recipe in §4); **T4–T7 construct the real
`TransformerLayerSchedulePlan`** at 1 GPU (PP1, no trainer; crib
`tests/unit_tests/a2a_overlap/utils.py` + `test_schedule_chunk_1f1b.py`).

| # | name | construction | PASS bars |
|---|---|---|---|
| T1 | **LoRA-trap grad equivalence (PRIMARY)** | K=1 (layer-2 = first MoE layer opaque; layer-3 eager-overlapped), adapter-only per fixture; **branch tip ≥ b907b6153 required** (the frozen embedding triggers the PreProcessNode grad-root landmine on pre-fix trees — see §0) | (a) adapter grads **NONZERO** on both paths (the trap check); (b) adapter grads **bitwise-equal** to the stock reference (max abs diff == 0; if any nondeterminism is identified and named, ≤1e-8 rel); (c) input grad at the opaque layer's boundary equal (same tol); (d) frozen expert weights have NO grad entries on either path |
| T2 | **negative control (trap detector), two arms** | arm A: T1 with the opaque callable's `hidden_states.requires_grad_(True)` line monkeypatched out. arm B (jacobi, second trap site): T1 against a tree WITHOUT b907b6153 | arm A: opaque layer's adapter grads MUST come out zero/absent. arm B: the closing `pre_process.backward` MUST raise RuntimeError (the pre-fix trap is fail-loud). **If arm A shows nonzero grads, the TEST is broken (the trap is not load-bearing in this construction) — STOP, fix the test, do not proceed to box.** The Aug-9 v1-inertness lesson encoded |
| T3 | **RNG fork/restore equivalence** (risk §6.2) | T1 + dropout p=0.1 inside the MoE layers | forward outputs AND adapter grads bitwise-equal across the two paths (the opaque node's checkpoint owns RNG save/restore via `tensor_parallel.checkpoint`, random.py:574-575 — any divergence = stream/RNG semantics bug) |
| T4 | **eager-side regression** | K=0 (all overlapped — the legal no-op) vs flag-ON-no-dial reference | grads bitwise-equal; structural assert: opaque builder never invoked |
| T5 | **degenerate anchor K=L** | all MoE layers opaque | grads bitwise-equal vs stock full-recompute reference; structural assert: no comm-stream work on any layer |
| T6 | **node-contract structural check** (per the frozen build) | K=1, inspect the plan's nodes | opaque layer: attn = real node on comp stream; mlp/moe_dispatch/moe_combine = `NoopScheduleNode` (mlp MUST be — the schedule calls `mlp.backward_dw()` unconditionally; NoopScheduleNode's `backward_dw` no-op exists); attn node `delay_wgrad_compute=False` (plan-builder-forced), dw map empty; final-LN applied outside the checkpoint only when `is_last_layer` |
| T7 | **plan-structure coverage** (assigned to this harness by the branch docstring) | K=1 tiny MoE plan construction | dense layers (1-2) keep their stock structure (dial skips non-MoE); MTP slot excluded if present; a layer with missing `layer_local_index` in extra_args is left eager (safe fallback); eager MoE layers keep the 5-node decomposition |

## 3. Failure triage (pre-registered)

- T1(a) adapter grads zero on the DIAL path, nonzero on reference → the trap
  fired: the requires_grad maneuver is missing/misplaced. Blocking; back to
  jacobi with the repro.
- T1(b/c) mismatch within nonzero grads → ordering/RNG/stream semantics:
  capture both grad tensors, diff histogram, check T3 to isolate RNG.
- T3 fails with T1 passing → RNG fork/restore placement; blocking for
  correctness (silent numerical drift class).
- T4 fails → the branch regressed the stock overlap path; blocking.
- Any config assert from transformer_config.py:2632-2642 fires → the patch
  touched the stock-path guards (forbidden by design §3.1); blocking.

## 4. Mechanics — driving the callable standalone (jacobi's recipe, folded in)

The fiddly part is running `submodule_checkpointed_layer_forward` without the
full executor. Per jacobi (code-verified against the pushed branch):

- **Mock node** needs only:
  `(a)` `.chunk_state = SimpleNamespace(attention_mask=None,
  rotary_pos_emb=<per model spec>, rotary_pos_cos=None, rotary_pos_sin=None,
  packed_seq_params=None, sequence_len_offset=None, padding_mask=None,
  model=SimpleNamespace(decoder=SimpleNamespace(final_layernorm=None)))`;
  `(b)` `.is_last_layer = False`.
- **Mimic the framework's input handling EXACTLY** (`ScheduleNode._forward`,
  pipeline_parallel/utils.py:205-213): `hidden = base.detach();
  hidden.requires_grad = base.requires_grad`. The callable mutates THAT leaf's
  `requires_grad` in place; the harness reads the input grad back off the same
  object (that is the framework's `get_grad` path).
- **Reference arm:** stock `tensor_parallel.checkpoint(custom_forward, False,
  hidden)` over the same layer + same weights + same input. Bars: forward
  output **bitwise**, adapter grads **allclose** (rtol 1e-5 / atol 1e-8, max
  diff reported; a systematic nonzero diff = investigate), input grad **exists**
  and matches.
- **T2 negative control, sharpened (jacobi):** with the `requires_grad_` flip
  patched out AND input `requires_grad=False`, the checkpoint **output** must
  come out `requires_grad=False` (plain `autograd.Function` semantics) →
  backward never fires → adapter grads stay None/zero. **If they don't, the
  TEST is broken, not the patch.**
- Crib patterns: `tests/unit_tests/a2a_overlap/utils.py` +
  `test_schedule_chunk_1f1b.py` (existing tiny-MoE builders).

## 5. Harness placement + execution

- Test file lives on jacobi's dial branch in the vendored Megatron-LM fork
  (`tests/unit_tests/a2a_overlap/test_recompute_dial_grad_equiv.py` alongside
  jacobi's config-surface suite — mcore-side code, mcore-side tests, per the
  campaign's repo-layering rule). Mac-side stub harness NOT planned (mcore does
  not import on Darwin; the volta/poincare stub pattern covers trainer-side
  code only).
- Execution: on-box 1 GPU (`pytest
  tests/unit_tests/a2a_overlap/test_recompute_dial_grad_equiv.py`, no torchrun
  at PP1) in any box window ≥ 10 min, or Linux-GPU CI.
- The test file is written ahead against the frozen interface; first run is
  ladder rung 1 (DIAL_VALIDATION_LADDER.md).
- Evidence: test output + the T1/T2 grad-dump tensors archived to the campaign
  run folder (`runs/overnight_20260813_overlap_campaign/`) on first execution.
