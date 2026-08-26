# Full 131k DeepEP memory and correctness analysis

## Verdict

- DeepEP does **not** materially lower the fleet-wide live-memory ceiling. Maximum PyTorch-allocated memory is 153.063 GiB with DeepEP versus 153.079 GiB with all-to-all, only 16.4 MiB lower. The ceiling is on pipeline stage 1, where the dispatcher does not determine the model-wide peak.
- DeepEP materially lowers dispatcher-path live memory. The per-rank maximum attributed to the dispatcher falls from 11.360 GiB mean (9.129-14.267 GiB) to 6.166 GiB (5.262-7.221 GiB), a 5.194 GiB/rank or 45.7% reduction. The union specifically visible through `fused_a2a.py` and `deep_ep/buffer.py` peaks at 1.683 GiB/rank mean.
- DeepEP's true maximum reserved memory is **higher**, not lower: 160.061 GiB versus 157.264 GiB, a 2.797 GiB increase. `result.json` gives the opposite impression because `aggregates.peak_gpu_memory_bytes` is rank 0's final reserved value, not the maximum across ranks.
- No PyTorch-managed Flex leak is present. All 16 ranks in each run have two consecutive snapshot markers, one control forward apart, with byte-identical live memory. No DeepEP/Flex-attributed allocation remains live at the end on any rank, and there is no OOM event.
- Operational correctness passes: both jobs finish five steps with finite losses and gradient norms, identical token accounting, and no CUDA/NCCL failure, traceback, OOM, or dropped-token report. Numerical behavior is close but not proven equivalent: maximum loss delta is 0.001223 (0.00993%) and maximum gradient-norm delta is 4.67% in warmup and 1.56% after warmup.
- DeepEP 16 SM is slower at full 131k despite its smaller dispatcher live set: control forward/backward mean is 46.258 s versus 42.590 s (+8.61%), and throughput is 708.4 versus 769.4 tokens/s/GPU (-7.93%). Memory reduction on stage 0 does not translate into a full-run performance win.

## Scope and method

Compared:

- `artifacts/full-131k-alltoall`: 16 snapshots, `memory.rank0.pickle` through `memory.rank15.pickle`, plus `result.json` and `trainer_srun.log`.
- `artifacts/full-131k-deepep-sms16`: the corresponding 16 snapshots, result, and log.
- Configs are identical except for `moe_token_dispatcher=alltoall` versus `moe_token_dispatcher=flex`, `moe_flex_dispatcher_backend=deepep`, and `moe_flex_dispatcher_num_sms=16`.

Both runs use GLM-5.2-FP8, sequence length 131,072, four datums and 524,288 tokens per step, 16 GPUs, TP=1, PP=2, EP=8, CP=8, ETP=1, DP=1, LoRA rank 32, and seed 1234. Ranks 0-7 and 8-15 are treated as pipeline stages 0 and 1, respectively; the stage split is also directly visible in their 99.1 versus 110.6 GiB steady live sets.

All values below are binary GiB. "Live" is PyTorch `memory_allocated` semantics (`active_allocated` blocks). "Allocator-active" additionally includes blocks awaiting asynchronous free. "Reserved" is mapped caching-allocator memory. Dispatcher attribution is the union of allocations whose Python stack passes through `token_dispatcher.py`; explicit DeepEP attribution is the union passing through `fused_a2a.py` or the `deep_ep` package. Attribution maxima and model-wide maxima can occur at different times and are not additive.

All snapshots use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Every rank's history reached the configured 1,000,000-event cap. The retained ring covers the final step (about 61-62 s for all-to-all and 65-67 s for DeepEP), includes two consecutive snapshot markers, and reconstructs the logged peak exactly. It does not preserve every allocation event from all five steps; full-run leak conclusions therefore also use the repeated snapshot markers and the five per-step log records.

## Fleet summary

| Metric | All-to-all | DeepEP 16 SM | DeepEP minus all-to-all |
|---|---:|---:|---:|
| Live peak, all-rank mean | 146.415 | 144.303 | -2.112 |
| Live peak, fleet maximum | **153.079** | **153.063** | **-0.016** |
| Allocator-active peak, all-rank mean | 146.415 | 145.559 | -0.856 |
| Final live, all-rank mean | 104.884 | 104.868 | -0.017 |
| Final reserved, all-rank mean | 150.609 | 151.202 | +0.593 |
| Reserved peak, fleet maximum | **157.264** | **160.061** | **+2.797** |
| Inactive reserve at end, mean | 45.725 (30.36%) | 46.334 (30.64%) | +0.609 |
| Inactive reserve at live peak, mean | 4.194 | 6.894 | +2.700 |
| Dispatcher-path live maximum, mean | 11.360 | 6.166 | -5.194 (-45.7%) |
| Dispatcher-path rank spread | 5.138 | 1.959 | -3.179 (-61.9%) |
| Permutation-path live maximum, mean | 7.875 | 4.661 | -3.214 (-40.8%) |
| Explicit all-to-all / DeepEP live maximum, mean | 4.671 | 1.683 | -2.988 |
| Final dispatcher-path allocation | 0 on 15 ranks; 8.125 MiB on rank 0 | 0 on every rank | No DeepEP retention |

The 8.125 MiB all-to-all rank-0 allocation has `_linear_forward` as its top frame and merely has `token_dispatcher.py` deeper in its stack. It is not evidence of a retained communication buffer. The remaining 16.387 MiB/rank final-live difference is in blocks without a Python stack and cannot be assigned to Flex. Both differences are stable across consecutive snapshots.

## Pipeline-stage balance

| Stage / metric | All-to-all mean (range) | DeepEP mean (range) | Change |
|---|---:|---:|---:|
| PP0 live peak, ranks 0-7 | 139.790 (135.813-144.083) | 135.582 (132.472-138.996) | -4.207 mean |
| PP0 live-peak spread | 8.270 | 6.524 | -1.746 (-21.1%) |
| PP0 allocator-active peak | 139.790 (135.813-144.083) | 138.093 (134.583-141.776) | -1.696 mean |
| PP0 reserved peak/end | 144.067 (140.367-148.588) | 142.544 (138.965-146.055) | -1.523 mean |
| PP1 live peak, ranks 8-15 | 153.040 (153.034-153.079) | 153.024 (153.018-153.063) | -0.016 mean |
| PP1 live-peak spread | 0.045 | 0.045 | unchanged |
| PP1 reserved peak/end | 157.151 (156.893-157.264) | 159.861 (159.670-160.061) | +2.709 mean |

DeepEP helps the stage where routed-token imbalance is visible: stage-0 live peak falls by 3.00-5.09 GiB depending on rank, and its spread narrows. It does not move the stage-1 live peak, which remains the fleet bottleneck. DeepEP also changes the allocation shape: on PP0, allocator-active peak averages 2.511 GiB above user-live peak and reaches a 2.847 GiB gap on the worst rank. All-to-all has no measurable pending-free gap. These are stream-ordered DeepEP allocations awaiting completion, not leaked live tensors; allocator-active and live memory are identical again at snapshot end.

The reserved-memory movement is stage-dependent. DeepEP reserves 1.523 GiB/rank less on PP0 but 2.709 GiB/rank more on PP1. Consequently the fleet maximum and overall mean reserve both increase even though rank 0 decreases.

## Per-rank comparison

`Reserve` is final and retained-window peak because no segment was unmapped in the retained final-step window. `Dispatch peak` is backend-path-attributed live memory and can peak at a different instant from total live memory.

| Rank | PP | A2A live peak | DeepEP live peak | Live delta | A2A reserve | DeepEP reserve | Reserve delta | A2A dispatch peak | DeepEP dispatch peak |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 139.705 | 135.184 | -4.521 | 143.434 | 142.227 | -1.207 | 11.144 | 6.142 |
| 1 | 0 | 138.491 | 135.030 | -3.461 | 142.182 | 141.152 | -1.030 | 9.129 | 5.262 |
| 2 | 0 | 136.014 | 132.472 | -3.542 | 140.424 | 139.082 | -1.342 | 10.050 | 5.609 |
| 3 | 0 | 142.023 | 137.244 | -4.779 | 146.947 | 144.785 | -2.162 | 11.563 | 6.307 |
| 4 | 0 | 140.601 | 135.724 | -4.877 | 144.584 | 143.750 | -0.834 | 11.941 | 6.441 |
| 5 | 0 | 144.083 | 138.996 | -5.087 | 148.588 | 146.055 | -2.533 | 12.104 | 6.487 |
| 6 | 0 | 141.589 | 137.199 | -4.390 | 146.012 | 144.336 | -1.676 | 10.992 | 6.099 |
| 7 | 0 | 135.813 | 132.809 | -3.004 | 140.367 | 138.965 | -1.402 | 10.156 | 5.543 |
| 8 | 1 | 153.079 | 153.063 | -0.016 | 157.205 | 160.061 | +2.856 | 11.786 | 6.349 |
| 9 | 1 | 153.034 | 153.018 | -0.016 | 157.244 | 160.041 | +2.797 | 14.267 | 7.221 |
| 10 | 1 | 153.034 | 153.018 | -0.016 | 157.225 | 159.924 | +2.699 | 10.095 | 5.709 |
| 11 | 1 | 153.034 | 153.018 | -0.016 | 157.146 | 159.768 | +2.622 | 9.252 | 5.300 |
| 12 | 1 | 153.034 | 153.018 | -0.016 | 157.010 | 159.670 | +2.660 | 12.532 | 6.614 |
| 13 | 1 | 153.034 | 153.018 | -0.016 | 157.225 | 159.865 | +2.640 | 11.019 | 6.128 |
| 14 | 1 | 153.034 | 153.018 | -0.016 | 157.264 | 159.885 | +2.621 | 14.259 | 7.176 |
| 15 | 1 | 153.034 | 153.018 | -0.016 | 156.893 | 159.672 | +2.779 | 11.471 | 6.272 |

Final live memory is strongly stage-balanced: 99.157 GiB on all-to-all PP0 ranks (99.173 on rank 0) versus 99.141 GiB with DeepEP (99.149 on rank 0), and exactly 110.610 versus 110.594 GiB on every PP1 rank. The approximately 11.453 GiB PP-stage baseline split is expected topology/model state, not expert-routing imbalance.

## Flex and DeepEP allocation ownership

The snapshot stacks prove that the intended backends actually executed: all-to-all ranks contain `mappings.py:all_to_all`, while all DeepEP ranks contain `fused_a2a.py` and `deep_ep/buffer.py` allocations. This is stronger effective-backend evidence than the logs, which do not print the dispatcher backend or SM count. The artifacts independently prove DeepEP execution, but the effective 16-SM setting remains config-only evidence.

Explicit DeepEP live ownership is bounded and balanced:

| Stack source | Mean source-live max | Rank range | Interpretation |
|---|---:|---:|---|
| `fused_a2a.py:236 fused_dispatch` | 1.496 | 1.367-1.513 | Dispatch-visible tensor union |
| `deep_ep/buffer.py:407 dispatch` | 1.496 | 1.367-1.513 | Same nested dispatch allocations; do not add to row above |
| `fused_a2a.py:199 backward` | 1.484 | 1.356-1.501 | Dispatch backward payload |
| `deep_ep/buffer.py:396 dispatch` | 1.484 | 1.356-1.501 | Same nested backward allocations; do not add to row above |
| `deep_ep/buffer.py:462 combine` | 0.188 | exactly 0.188 | Combine output/work tensor |

The union of all explicit DeepEP stacks peaks at 1.683 GiB/rank mean with only 0.146 GiB rank spread. All are transient; explicit DeepEP live bytes are zero at the end on every rank.

The snapshots do not expose a long-lived allocation attributed to process-global DeepEP `Buffer(...)` construction. DeepEP can allocate NVLink/RDMA or CUDA memory outside PyTorch's caching allocator, which these pickles cannot see. The analysis therefore rules out a retained **PyTorch-managed** Flex/DeepEP buffer but cannot quantify externally managed communication memory.

## Reserve, cache, and fragmentation

- Final inactive reserve averages 45.725 GiB for all-to-all and 46.334 GiB for DeepEP. This is high-water cache, not live memory.
- At the instant of live peak, DeepEP has 6.894 GiB/rank mean inactive reserve versus 4.194 GiB for all-to-all. DeepEP's allocation shape leaves 2.700 GiB/rank more cache headroom at that instant.
- No retained final-step trace contains a segment unmap. Reserved memory reaches its high-water mark and remains mapped through the end of the recorded window.
- The allocator has only about 203 inactive blocks/rank for all-to-all and 201 for DeepEP, with very large available blocks: largest inactive block averages 30.714 GiB for all-to-all and 28.931 GiB for DeepEP. There is no OOM and no evidence that small-hole fragmentation prevented a large allocation.
- The device has 267.69 GiB total memory. Even the 160.061 GiB maximum reserve leaves about 107.63 GiB outside the PyTorch cache, so neither run is close to capacity pressure.

## Leak analysis

- Every one of the 32 snapshots has exactly 1,000,000 retained events and zero OOM events.
- Every rank has two snapshot markers bracketing one final control forward. The marker's live-byte value is identical before and after that forward on all 16 ranks in both runs; all 32 per-rank deltas are exactly zero bytes.
- `result.json` and the logs show five completed optimizer steps in both runs. Peak allocated and reserved values plateau by step 2-3 rather than growing each step.
- Final live memory is lower, not higher, with DeepEP by 16.4 MiB on normal ranks and 24.5 MiB on rank 0.
- DeepEP's pending-free memory is transient: allocator-active exceeds user-live memory during PP0 work, but there are no `active_awaiting_free` blocks at the end.
- No explicit DeepEP allocation remains live at the end. There is no monotonic PyTorch-managed Flex growth.

These checks are sufficient to reject a step-over-step PyTorch allocator leak over this five-step run. They do not test an hours-long steady-state run and cannot see memory allocated outside PyTorch.

## Memory reporting correctness

`result.json` reports:

| Source | All-to-all | DeepEP | Apparent change |
|---|---:|---:|---:|
| `aggregates.peak_gpu_memory_bytes` | 143.434 GiB | 142.227 GiB | -1.207 GiB |

Those values exactly equal rank 0's final reserved segment total in the pickles. They are not the all-rank peak despite the field name. The per-step log's reduced `peak_reserved_bytes` and `peak_allocated_bytes` agree exactly with the maxima reconstructed from all pickles:

| Correct all-rank high-water metric | All-to-all | DeepEP | Actual change |
|---|---:|---:|---:|
| Peak allocated/live | 153.079 GiB | 153.063 GiB | -0.016 GiB |
| Peak reserved | 157.264 GiB | 160.061 GiB | +2.797 GiB |

Therefore `result.json` must not be used to claim a DeepEP memory reduction for this PP2 run. It samples the lower-memory pipeline stage and reverses the conclusion about reserved memory.

## Numerical and operational correctness

| Window | Loss, all-to-all | Loss, DeepEP | Loss delta | Grad norm, all-to-all | Grad norm, DeepEP | Grad relative delta |
|---|---:|---:|---:|---:|---:|---:|
| warmup 0 | 12.316959892 | 12.317238367 | +0.000278475 (+0.00226%) | 0.427006662 | 0.407057852 | -4.6718% |
| traced 0 | 12.316998039 | 12.315963295 | -0.001034745 (-0.00840%) | 0.462981462 | 0.455741793 | -1.5637% |
| control 0 | 12.319063714 | 12.318967392 | -0.000096322 (-0.00078%) | 0.352792561 | 0.353146493 | +0.1003% |
| control 1 | 12.309552647 | 12.309829215 | +0.000276568 (+0.00225%) | 0.354250073 | 0.358770669 | +1.2761% |
| control 2 | 12.313551434 | 12.312328814 | -0.001222620 (-0.00993%) | 0.404852450 | 0.398569375 | -1.5519% |

The losses track closely and do not drift or diverge. Gradient norms are finite and in the same range, but their differences are substantially larger than the short EP8 debug comparison, especially the 4.67% warmup difference. FP8 arithmetic, long-context reduction order, and backend-specific ordering can plausibly amplify gradient-norm differences, but aggregate scalar metrics cannot prove that no token was misrouted. A strict numerical-equivalence claim would require identical frozen inputs/weights and tensor- or update-level comparisons, which are not present in these artifacts.

Operational evidence is clean:

- Both runs complete step 5.
- Every step reports `num_tokens=524288` and `num_loss_tokens=524284`. The fixed four-token difference is one non-loss/masked token per datum, not a MoE dropped-token count.
- No log contains a traceback, OOM, CUDA error, NCCL warning/error, NaN/Inf report, or dropped-token report.
- Both logs contain the same nonfatal profiler error, `External init callback must run in same thread as registerClient`; both still produce a valid trace.
- Both logs emit `AccumulateGrad` stream-mismatch warnings across ranks. This can add synchronization and affect performance, but it is common to both backends and does not indicate wrong outputs.
- `final_status.last_loss` and `final_status.grad_norm` are null despite complete window results. This is a status-reporting limitation, not a failed step.

## Performance context

| Metric | All-to-all | DeepEP 16 SM | DeepEP change |
|---|---:|---:|---:|
| Control FB mean | 42.590 s | 46.258 s | +8.61% slower |
| Control TPS/GPU | 769.381 | 708.368 | -7.93% |
| Control FB population CV | 0.69% | 0.11% | More stable, but slower |
| Traced FB | 44.285 s | 48.295 s | +9.05% |
| Kineto overhead | 3.98% | 4.40% | +0.42 percentage points |
| Optimizer mean | 1.036 s | 0.659 s | -36.4%, noisy/outlier-sensitive |

DeepEP is steadier across the three controls, but the slowdown is much larger than that variance. For this full 131k PP2/CP8 shape, smaller dispatcher live memory is not a sufficient reason to prefer DeepEP 16 SM: it leaves the actual live-memory bottleneck unchanged, increases maximum reserve, and reduces throughput.
