# M=N SCHEDULE MEMO — one schedule call per optimizer step on the THD path (volta, 2026-08-12)

Re-scoped from the VPP-grouping option-1 per gibbs's paper-trace: the convoy is the
runner's M=1-per-partition-call convention (the schedule never receives a second
microbatch to overlap). The fix is **M=N**: one `forward_backward` schedule call per op
with `num_microbatches = len(partitions)`. Plain non-interleaved 1F1B then actually
pipelines at PP2 (stage-1 bwd(k) overlaps stage-0 fwd(k+1)). No VPP, no layout change,
no schedule surgery. Bonus: M=N with N≥2 satisfies the interleaved constraint
(M ≥ PP), so VPP2 + `overlap_p2p_comm` (E2b) becomes a *later stacking lever* on top of
this, not a prerequisite.

## (a) Call-site restructure (feasibility: yes — small)

The change is contained in `MegatronTrainingRunner._run_forward_backward`
(`training_runner.py`, THD branch, currently the per-partition `for microbatch in
microbatches` loop at :343-399).

Today (per partition): `stack.forward_backward(data_iterator=iter([mb]),
num_microbatches=1, seq_length=mb.thd_global_seq_len, micro_batch_size=1)`, finalize
suppressed across the loop and called once after.

After (one call):

```python
num_microbatches = len(microbatches)
forward_step = _sum_over_microbatches(forward_step, num_microbatches)  # see (c)
metrics = stack.forward_backward(
    forward_step_func=forward_step,
    data_iterator=iter(microbatches),
    model=stack.model_list,
    num_microbatches=num_microbatches,
    seq_length=max(mb.thd_global_seq_len for mb in microbatches),  # cosmetic, see below
    micro_batch_size=1,
    forward_only=forward_only,
)
```

- `seq_length` is consumed only by `get_tensor_shapes` (schedules.py:2291/2299) and is
  ignored under `variable_seq_lengths=True` (set for all CP>1 runs) — per-partition
  length variation inside one call is already handled by on-the-wire shape exchange
  (PACKING_MEMO Q2). Under the PP>1 pad gate every partition is 131072 anyway.
- **Finalize**: stop suppressing; the schedule's own tail calls
  `finalize_model_grads_func([model], None, pg_collection, force_all_reduce=False)`
  exactly once per call (schedules.py:2466-2481 non-interleaved, :832-838
  no-pipelining) — identical semantics to today's explicit post-loop
  `saved_finalize(model_list, None)`. `_suppress_schedule_finalize` becomes dead code,
  removed. This also aligns THD with the BSHD path, which has always run M=N with
  in-schedule finalize (training_runner.py:418-428).
- **Empty DP slice** (0 partitions — legal at DP>1): skip the schedule call but still
  call the config's `finalize_model_grads_func(model_list, None)` explicitly. Today's
  code does the same via the post-loop `saved_finalize`; a rank with no partitions must
  still join the DP grad all-reduce its peers are waiting on. (Preserved behavior,
  now explicit.)
- **Per-microbatch → per-partition accounting**: `forward_data_store` on the last stage
  is already per-microbatch in forward order (schedules.py:320-340; appended in
  microbatch order on both the no-pipelining and 1F1B paths). The runner walks
  `packed.microbatches` (all ranks, for the datum-offset check) and indexes the
  returned list by position (last stage) — `partition_metric = metrics[i]` corresponds
  to partition i by construction. `is_thd_output_leader` logprob splitting per
  partition is unchanged (each microbatch carries its own `packed_seq_params`; the CP
  stitch stays inside the loss func).
- **Router replay (R3)**: safe under M=N. `set_target_indices` appends to a per-router
  FIFO (`replay_backward_list`) and `REPLAY_BACKWARD` pops in order
  (router_replay.py:114-117, 175); forward(k) arms per-microbatch inside the forward
  step, backward(k)'s recompute pops FIFO order which matches forward order on both
  schedules. The runner-level global set/clear now wraps the single call instead of
  each partition. (R3 stays config-rejected under PP>1 regardless; this keeps PP=1 R3
  correct.)

## (b) Blast radius

- **Files**: `training_runner.py` only (THD branch restructure + delete the suppression
  helper). No packing/THD/loss/config changes. `pack_thd_cp_microbatch` and the
  pad-to-131k gate are untouched (M=N composes with them: padding makes per-call shapes
  uniform; M=N makes the count of microbatches per call N).
- **Per-partition metric/loss accounting**: unchanged content — the same per-partition
  dicts arrive as a list instead of being appended in a loop; aggregation
  (`aggregate_microbatch_metrics`, `_aggregate_reported_sum_count`, CP-only
  `_loss_report` inside loss funcs, post-loop `_dp_reduce_sum`, PP broadcast) is
  untouched.
- **Odd partition counts / d1**: M=1 degenerates to exactly today's shape (one
  microbatch, no pipelining, `_sum_over_microbatches` is a no-op at N≤1). d1 ramps
  unaffected. Odd M is fine — 1F1B imposes no divisibility constraint.
- **Warmup**: unchanged path — 1 partition → M=1 call (with the PP2 pad gate, still the
  full-size 131k probe from the packing branch).
- **DP>1 with unequal partition counts across ranks**: safe. Each DP rank makes one
  call with its own M; all in-call collectives (CP loss reduce, CP logprob stitch, EP
  all-to-all, PP p2p) are within groups that share a DP coordinate and therefore share
  M. The only cross-DP collectives are the finalize and the runner's post-call
  `_dp_reduce_sum`, both once per op, reached by every rank regardless of M.
- **R3 replay**: covered in (a) — FIFO semantics make it safe; still config-gated off
  under PP>1.

## (c) Numerical-parity argument

- **Grad scale**: the schedule's legacy 2-tuple path scales each microbatch loss by
  `cp_size / num_microbatches` (schedules.py:336-339). At M=1 today that's ×cp. Under
  M=N we apply the BSHD path's existing `_sum_over_microbatches` wrapper
  (loss.py:1053-1075), which multiplies the loss by N before the schedule divides —
  net per-microbatch scale identical to today (×cp), so accumulated grads equal today's
  sum (not the mean — no N× under-scaling). `grad_scale_func` is identity for bf16
  (`optimizer.scale_loss = get_loss_scale() * loss`, loss_scale=1).
- **Accumulation order**: backward completes microbatches in forward order
  (FIFO pop in both schedules), so grad-buffer addition order across partitions is
  unchanged (0,1,…). Residual drift sources: the ×N/N fp round-trip and different
  kernel interleaving under a live pipeline — tolerance-level, not systematic.
- **Validation instrument**: the parity driver (tools/parity_driver.py) is exactly the
  right check, re-scoped: same datum set, pre-M=N vs post-M=N build, loss rel ≤1e-6 /
  per-token logprobs abs ≤1e-3 / loss-token counts exact. (Its DP-topology invariance
  argument was already cleared with maxwell.)

## (d) Memory estimate (flag i)

In-flight activation sets per stage go from 1 (today, serialized) to at most
min(M, PP) = 2 at PP2 (stage 0: 1 warmup + 1 steady; stage 1: 1; the count caps at PP
regardless of M — d4 does not hold 4). One set at 131k/CP8 with full recompute ≈
per-layer checkpointed inputs only: 38 stage-0 layers × 16,384 local tokens × hidden ×
2 B ≈ **8-9 GiB** (hidden 6-7k), plus one microbatch's transient recompute
intermediates which exist under both schemes. First-light showed 164/275 GiB at d2, so
the delta is ~8% of the ~110 GiB headroom — expected safe; the d1→d2 ramp measures it
directly. (At PP1 the no-pipelining schedule interleaves fwd/bwd per microbatch and
holds one set at a time — no PP1 memory change, so CP32 golden is not re-profiled.)

## (e) Effort

Hours, not days: the diff is ~60-80 lines in one function plus test updates (the
dispatch tests' `fake_schedule` pattern already asserts per-call num_microbatches, so
they'll catch the restructure precisely). Mac-verifiable logic; on-box validation =
parity driver + d1/d2 canary ladder gibbs is prepping.

## (f) Degenerate fallback

Not needed as a special case: M=1 (d1 ramp, warmup) is already the degenerate path and
behaves identically to today. If M=N itself misbehaves on-box, the revert is one
commit; no runtime flag is warranted (the behavior change is the point of the
experiment, and a flag would leave two live schedule conventions to validate).
