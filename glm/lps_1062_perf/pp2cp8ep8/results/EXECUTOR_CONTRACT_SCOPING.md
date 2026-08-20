# Executor-contract scoping — what adopting the combined-1F1B schedule-plan protocol takes in OUR trainer

Author: serre (K3), 2026-08-13 early CDT. Companion to
`OVERLAP_PER_LAYER_RECOMPUTE_DESIGN.md` (the upstream memory dial). This doc
scopes the *trainer-side* prerequisite that the L0 probe just exposed. No code
written, no GPUs touched; citations verified against the box's exact pins
(trainer @ 73c24b00 read on-box where the local checkout predates the M=N
merge; vendored mcore 57efae08b line-verified locally).

## 1. The failure being scoped (L0 record, NOTEBOOK ~04:4x)

L0 boot 2 (BT_SKIP_WARMUP=1) reached READY, then the d2 probe failed
instantly:

```
TypeError: _sum_over_microbatches.wrapped() got an unexpected keyword
argument 'return_schedule_plan'
```

Mechanism (source-verified): with `overlap_moe_expert_parallel_comm` on, the
schedule dispatches to the combined-1F1B executor (schedules.py:743
no-pipelining, :1465 interleaved — entered only `and not forward_only`), whose
step calls `forward_step_func(data_iterator, unwrapped_model,
return_schedule_plan=True)` (combined_1f1b.py:390-391) and requires an
`AbstractSchedulePlan` back (:393-395). The trainer's forward-step closures
(`_build_forward_step` family, loss.py:1027-1050, wrapped by volta's
`_sum_over_microbatches`, loss.py:1053-1075) take exactly
`(data_iterator, model)` and return `(output_tensor, loss_func)` — no
schedule-plan protocol exists. A kwargs passthrough in the M=N wrapper only
moves the failure one level down (borel, NOTEBOOK): the closures themselves
must learn the protocol.

**Why this matters beyond L0:** BOTH big-win paths run on this executor — the
offload probe (L0b with the flag) and the per-layer recompute dial. The
contract is the shared prerequisite. (L2/VPP-p2p is unaffected: plain
interleaved schedules carry no schedule-plan references.)

## 2. The contract (what the executor expects), verified

Reference implementation: mcore's `pretrain_gpt.py:333-340`:

```python
if return_schedule_plan:
    schedule_plan = model.build_schedule_plan(
        tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask)
    return schedule_plan, partial(loss_func, loss_mask, model=model)
```

So the forward step must return `(schedule_plan, loss_func)` where:

1. **The plan carries the batch.** `GPTModel.build_schedule_plan`
   (gpt_model.py:787-860) accepts input_ids, position_ids, attention_mask,
   decoder_input, labels, packed_seq_params (:795), padding_mask (:800), and
   `output_processor` + `output_processor_context` (:802-803) — a user hook
   run "instead of the default logits/loss path" in the postprocess node.
   Everything lands in the plan's `ModelChunkState`
   (model_chunk_schedule_plan.py:365-380).
2. **The loss still runs as a node after the chunk.** The executor wraps the
   returned `loss_func` in a `ScheduleNode` on the compute stream and drives
   it through `forward_step_calc_loss` (combined_1f1b.py:478-491) — the *same*
   loss-scaling seam the conventional path uses (schedules.py:514). At
   backward, `torch.autograd.backward(loss)` seeds through the loss node and
   `loss_node.get_grad()` (utils.py:270-276) hands the plan's output gradient
   to the chunk backward (combined_1f1b.py:441-446).
3. **Metrics flow is unchanged.** `forward_step_calc_loss` appends
   `loss_reduced` to `forward_data_store` per microbatch in forward order
   (schedules.py:342) — exactly the convention the M=N runner reads
   (`partition_metrics[i]["loss"]/["logprobs"]`, box training_runner.py
   :436-460).
4. **Grad finalize stays once per schedule call.** The combined branch returns
   into the schedule wrapper whose tail finalize is unchanged
   (schedules.py:828-836 no-pipelining, :2470-2481 interleaved) — the M=N
   runner's load-bearing invariant (box training_runner.py:383-397 comment)
   holds without changes.
5. **Grad scaling wrapper unchanged.** `_sum_over_microbatches` cancels the
   `/num_microbatches` inside `forward_step_calc_loss` (:335/:341); the
   combined path calls the same function, so volta's wrapper applies verbatim
   to the plan-world loss_func.

## 3. Verdict: wrapper-level adaptation, NOT a forward-step re-architecture

The trainer's loss functions compute the loss **from hidden states** (chunked
LM head: `_ce_loss_from_hidden`, loss.py:125-184; the RL/DPO family shares the
pattern, loss.py:645, :1011). That maps onto the contract without touching the
loss math at all:

**Option B (recommended first step — zero loss-code change):** the plan
returns post-norm hidden states; the existing loss_func runs inside the loss
node untouched.

- forward-step shim: under `return_schedule_plan=True`, pull the batch as
  today, then
  `return model.build_schedule_plan(input_ids=b.input_ids, position_ids=b.position_ids, attention_mask=None, packed_seq_params=b.packed_seq_params, padding_mask=b.padding_mask, output_processor=<passthrough>), <existing loss partial>`.
- The passthrough `output_processor` returns `hidden_states` unchanged
  (3 lines) so the plan's postprocess does NOT materialize full-vocab logits
  (the whole point of the chunked head; the default labels path would build
  [16384, 154880] logits per microbatch).
- The loss node then runs `_ce_loss_from_hidden` (chunked head + metrics)
  exactly as today; its backward seeds the chunk backward through
  `loss_node.get_grad()`.
- Known cost: the postprocess node upcasts its output with
  `float16_to_fp32` (fine_grained_callables.py:240) — the boundary hidden
  states go bf16→fp32, +192 MB transient per microbatch at 131k/CP8 (16384 ×
  6144 × 4 B). Acceptable against the ~96 GiB d16 headroom; if it ever
  matters, Option A removes it.

**Option A (memory-tight follow-up):** compute the CE loss *inside* the plan
via the `output_processor` hook (gpt_model.py:708-723 passes hidden_states,
labels, packed_seq_params, and the user context). The plan boundary carries
only the loss scalar (the fp32 upcast is then free); metrics (incl. the
logprobs tensor) stash into `output_processor_context` and the returned
loss_func reads them back into the mcore `(loss, metrics)` tuple. Slightly
more new code (hook adapter + stash protocol), same loss math.

Both options are wrapper-level: no changes to `_ce_loss_from_hidden`'s
numerics, to the RL family, to the runner, or to the schedule.

## 4. Seam-by-seam checklist (all verified)

| seam | status under the contract |
|---|---|
| THD per-partition data | unchanged — `_schedule_data_iterator` already replicates per-VPP-chunk iterators (box training_runner.py:245-248); each executor step pulls one packed microbatch as today |
| packed_seq_params / padding_mask | first-class `build_schedule_plan` params (:795/:800); THD shapes already validated on this machinery (E2b verdict: packed_seq_params threads through combined_1f1b; dynamic p2p shape exchange, p2p_communication.py:322) |
| attention_mask | stays None under THD (loss.py:207-208) |
| model unwrap | the executor unwraps to GPTModel itself (combined_1f1b.py:381-389, asserts GPTModel); GLM-5.2 maps to GPTModel (glm5_bridge.py:55) ✓; `language_model_for_chunked_output` unwraps internally (chunked_lm_head.py:123) so the loss closure is wrap-agnostic ✓ |
| labels | not passed to the plan (loss is computed from hidden states); they close over in the loss partial as today |
| metrics / logprobs | forward_data_store convention preserved (§2.3); runner's per-partition reads unchanged |
| grad scaling | `_sum_over_microbatches` applies verbatim (§2.5) |
| grad finalize | once per schedule call, preserved (§2.4) |
| reporting CP all-reduce | `_loss_report` reduces a **detached** reporting tensor on the CP group (loss.py:79-81) — off the autograd graph; under the executor it runs inside the loss node on the compute stream. Rank-symmetric schedule → no desync. Watch item, not a blocker |
| forward_only ops (parity probes) | the combined path is gated `and not forward_only` (schedules.py:743/:1465) — probes take the conventional path even with the flag on ✓ |
| router replay (R3) | global replay hooks fire inside the layer forward regardless of schedule; the fine-grained path calls the same submodules ✓ |
| VLM | `build_schedule_plan` has no pixel_values/image_grid_thw params — VLM × overlap-flag is unsupported; guard with a clear trainer-side error (text-only GLM-5.2 unaffected) |
| startup warmup | M=1 is illegal under the interleaved executor (schedules.py:1141-1148; L0 boot-1 died here). Options: warmup builds ≥2 partitions under VPP (poincare's morning item) or keep BT_SKIP_WARMUP=1 as the hatch. Independent of this contract work but on the same boot path |

## 5. Effort and risks

**Effort: ~100-150 trainer-side LoC + tests. Half a day to a running
prototype; 1-2 days hardened.**

- `_with_schedule_plan_protocol(forward_step)` wrapper (or a kwarg grown on
  each closure — one shared wrapper is cleaner) in loss.py, applied in
  `_build_forward_step` (:1027-1050): ~60-100 LoC.
- Passthrough `output_processor` (+ Option-A hook later if wanted): ~10-30
  LoC.
- Runner: **no changes** (the M=N call site already passes everything the
  executor needs).
- Tests: a protocol unit test (closure accepts the kwarg, returns an
  AbstractSchedulePlan + callable; CPU-mock the plan), then the standard
  ladder: d2 canary (loss 12.2-12.4 band, grad-norm comparability) → parity
  leg re-run under the flag at small seqlen → memory ramp.

**Risks, ranked:**

1. *Silent contract drift* — the shim returns a plan that is subtly wrong
   (wrong microbatch's batch, stale chunk_state). Mitigation: the parity
   harness at small seqlen is exactly the tool that catches this (per-token
   logprob bars); plus the unit test.
2. *Loss-node stream placement* of the reporting CP all-reduce (§4) — expect
   fine, watch the first trace for a serialization blip.
3. *Warmup M=1 illegality* (§4) — separate morning item (poincare); until
   then BT_SKIP_WARMUP=1 is load-bearing on flag-on boots.
4. *fp32 boundary upcast* under Option B (+192 MB/mb transient) — bounded,
   measured on the first boot; Option A is the escape.
5. *VPP-only coverage at PP>1*: the combined executor exists for
   no-pipelining (PP1) and interleaved (VPP) schedules only — plain PP2
   non-VPP never enters it (verified: no combined branch in
   `forward_backward_pipelining_without_interleaving`, schedules.py:2127+).
   So the flag must always ship with VPP at PP>1 — a config-validation note,
   not code.

## 6. Where this sits in the three-leg map

(a) **This contract** — our trainer, prerequisite for everything below.
(b) **Offload memory path** (L0b) — needs (a); the `f2407a10` offload hunk
    does not touch the contract (borel, NOTEBOOK ~04:4x).
(c) **Per-layer recompute dial** (upstream,
    `OVERLAP_PER_LAYER_RECOMPUTE_DESIGN.md`) — needs (a) on our side to run at
    all, plus the upstream dial to make memory fit at 131k.

(a) is independently valuable: it is the difference between "the overlap
executor is unreachable from our trainer" and "we can A/B every current and
future overlap feature upstream ships."

## 7. Addendum — the dispatcher question: is DeepEP the designed enabler of
## the overlap prize? (serre, 2026-08-13, cauchy-ordered)

**Short answer: no.** The overlap prize (the 25-28 s/step schedule-hideable
ceiling, gauss's A2A_EXPOSURE_DECOMPOSITION) is achievable on the stock
alltoall dispatcher; DeepEP/flex is a compatible transport, not the enabler.

Verified from source:

1. **The executor admits alltoall by name.** The overlap flag's dispatcher
   assert allows both `'alltoall'` and `'flex'`
   (transformer_config.py:2627-2629). The fine-grained callables carry
   dedicated flex-backend branches (enable_deepep/hybridep/ncclep,
   fine_grained_callables.py:499-510; the token_probs detachment at
   :608-611/:628-631) AND first-class alltoall handling (the default path;
   `should_free_input`'s alltoall rules at :82-100).
2. **The hiding mechanism is dispatcher-agnostic.** The dispatch/combine
   schedule nodes run on the comm stream and pair against the other
   microbatch's compute (model_chunk_schedule_plan.py:229-297); the a2a is
   hidden by stream placement regardless of which dispatcher issued it. The
   25-28s ceiling was measured on the NCCL alltoallv stack — no DeepEP in
   that measurement.
3. **Flex's async hooks DO auto-engage under the executor** — but they are
   host-side sharpening, not the prize. `MoEFlexTokenDispatcher.
   token_dispatch/token_combine` default `async_finish=True,
   allocate_on_comm_stream=True` (token_dispatcher.py:1785-1786/:1847-1848),
   so the combined executor's `layer.mlp.dispatch/combine` calls engage
   DeepEP's EventOverlap machinery automatically (no CPU wait on the GPU
   signal — cf. the blocking note at fused_a2a.py:106). What that buys is
   less host serialization around the comm — relevant to the CPU-blocked
   residual the L3 trace flagged, unmeasured tonight. It does not change the
   wire time (L5 measured DeepEP kernels ≈ the NCCL a2a they replace:
   8.40s vs 7-8s/step) or the imbalance-wait mass (~70% of exposure,
   dispatcher-agnostic).
4. **Historical framing, honestly:** the flag was built around flex-style
   async dispatch (the name says "expert_parallel_comm"; the flex branches
   are the special-cased ones), but the shipped executor supports alltoall
   explicitly — and our 131k stack runs alltoall. The prize does not route
   through DeepEP.

**Consequence for the overlap-program EV:** the path to the 25-28s ceiling
is **contract shim (0.5-2d, ours) + memory path (offload L0b OR the
per-layer dial)** — no DeepEP dependency. L5's answered-negative verdict
stands as a standalone-lever statement; "flex as overlap transport" becomes
an optional A/B inside the overlap program (after the shim), not a
prerequisite. If a future trace ever shows the alltoall dispatcher's
host-side dispatch_preprocess serialization (its DtoH sync for
tokens_per_expert, token_dispatcher.py:600-610 region) blocking the comm
stream under the executor, that — not wire time — is what would make flex
load-bearing, and it is checkable in one traced overlap boot.
