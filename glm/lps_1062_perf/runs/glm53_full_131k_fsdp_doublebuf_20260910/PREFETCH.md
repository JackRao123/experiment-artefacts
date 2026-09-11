# CP8EP1 FSDP: expert prefetch deadlines versus temporal overlap

## Conclusion

The earlier 73% compute-overlap statistic does **not** imply the other 27%
delays the step. In this saved profiled step, all steady-state expert gathers
finish before the preceding block's GPU work ends. This supports successful
latency hiding of bulk expert-weight transfers under the observed schedule.
It does not prove zero overhead from FSDP, prove no resource contention, or
establish how the schedule changes without communication.

All eight ranks were checked:

| Expert gathers | Ready before preceding block ends, per rank | Smallest observed lead across all ranks |
|---|---:|---:|
| Forward | 75 / 75 | 6.359 ms |
| Backward, after initial refill | 74 / 74 | 33.917 ms |

Rank 0's median lead is 8.240 ms forward and 34.388 ms backward. The initial
backward expert refill is separate: it takes 30.106 ms on rank 0 with no
concurrent non-NCCL kernel. There is no preceding transformer backward block
against which to apply the same deadline check.

## Why 73% and on-time prefetch are compatible

Communication can execute during CPU launch gaps, CP communication, or other
pauses and still finish before its target layer needs the weights. Such time
counts as not overlapping compute, but is not necessarily a communication
stall on the model's dependency path.

For example, rank 0's inferred layer-5 weight gather lasts 48.282 ms, with
24.006 ms having no concurrent compute, yet finishes **10.417 ms before
layer 4's MLP GPU work ends**. That gather is not late for layer 5.

Of rank 0's 1.202 s of expert-gather time without non-NCCL kernels, 0.381 s
is concurrent with CP collectives. The remainder includes many smaller gaps
and the initial backward refill. This observation does not classify every
remaining gap's cause; the gap-duration statistics are retained in JSON.

The distinct initial non-expert gather has large rank-arrival skew (roughly
1.11 s on rank 0 versus 1.15 ms on rank 7). See `OVERLAP.md`. Neither that
boundary effect nor non-expert weight-gather behavior is erased by the
steady-state expert conclusion here.

## Method and validation

- `fsdp_prefetch_timing.py` uses all-rank intervals already extracted by
  `fsdp_overlap.py`, plus GPU annotation envelopes for layer subparts.
- There are 150 expert gathers per rank, each with the already-verified
  EXPERT_DATA_PARALLEL_GROUP ownership. The first 75 belong to forward and
  complete before the last forward MLP ends; the remaining 75 start later.
- Target blocks are inferred from this model's sequential order: layers
  3..77 forward and 77..3 backward. This is not a generic mapping for arbitrary
  FSDP schedules, pipeline parallelism, or multiple microbatches.
- Forward deadlines use the preceding layer's `forward/mlp` GPU annotation
  endpoint. Backward deadlines use the preceding executed block's backward
  GPU annotation endpoint. These are proxies for the earliest next-block
  start, not reconstructed CUDA event dependency graphs.
- GPU `block_N/forward` annotations are absent even though CPU scopes exist;
  child GPU annotations provide the forward boundary instead. Backward GPU
  annotation labels contain `including_recompute`, but their GPU spans need
  not cover nested recompute annotations; only their ending boundary is used.
- The checked Core worktree matches the run's source SHA256 values for both
  `megatron_fsdp.py` and `param_and_grad_buffer.py`, at equivalent commit
  `e6c86ea3b75674dfff3769c7c92141629864dc0c`.
- Source uses layer-sized FSDP units, current/next-unit double buffering,
  asynchronous all-gathers, and current-weight readiness waits. Forward
  lookahead is suppressed during activation recomputation; backward hooks
  request backward-order prefetch.
- No training, tests, original driver/MFU edits, or new performance claims
  from unprofiled controls were introduced in this analysis.

Per-rank results: `cp8ep1/result/analysis/overlap/rankN-prefetch.json`.

Reproduce for a rank from this run directory:

```bash
python3 fsdp_prefetch_timing.py cp8ep1/result --rank 0
```
