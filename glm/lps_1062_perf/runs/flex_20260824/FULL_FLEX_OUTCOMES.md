# Full Flex 131k outcomes

## Verdict

- **HybridEP sms16 is the clear steady-state winner.** It reaches 837.550 tokens/s/GPU, 8.86% above full alltoall, 13.97% above DeepEP sms20, and 18.24% above DeepEP sms16. Its mean control forward/backward (FB) time is 39.124 s versus 42.590 s for alltoall.
- **DeepEP sms20 improves on DeepEP sms16 but does not recover alltoall parity.** It is 3.74% faster than sms16, yet remains 4.49% below alltoall. DeepEP sms16 is 7.93% below alltoall.
- **HybridEP also has the best PyTorch allocator shape.** Its all-rank mean reserve is 147.653 GiB, 2.956 GiB below alltoall and about 3.6 GiB below either DeepEP run. Its PP0 live-peak spread is 5.470 GiB versus 8.270 GiB for alltoall.
- **None of the backends moves the fleet live-memory ceiling.** PP1 still peaks at about 153.0 GiB in every run. HybridEP lowers PP0 peaks and reserve, but PP1 remains the maximum-memory stage.
- **There is no observed PyTorch-managed step leak.** Every one of the 64 rank snapshots has two steady-state snapshot markers with an exact zero-byte live-memory delta. Per-step peak values plateau by step 2 or 3. This does not cover hours-long operation or memory owned outside PyTorch's caching allocator.
- **Loss and gradient behavior is operationally healthy.** All values are finite and follow the same phase pattern. The largest aligned loss spread is 0.001989, or 0.016% of the loss. The largest aligned gradient-norm spread is 0.02146, about 5.3%, so these artifacts support stability but not bitwise numerical equivalence.
- **HybridEP pays the largest first-step cost.** Its warmup FB is 128.281 s, 3.28x its control mean and 89.157 s above steady state. Its launch-to-benchmark proxy is otherwise normal at 681.8 s. DeepEP sms16 has the worst startup reliability and latency evidence: one prior attempt stalled before server startup, and the successful run took 1467.8 s from launcher timestamp to benchmark start.

## Scope

Only the four full Flex 131k runs are compared:

- `artifacts/full-131k-alltoall`
- `artifacts/full-131k-deepep-sms16`
- `artifacts/full-131k-deepep-sms20`
- `artifacts/full-131k-hybridep-sms16`

The debug runs are excluded. The separate `artifacts/full-131k-deepep-sms16-stall.log` is used only for startup reliability evidence.

All completed runs use GLM-5.2-FP8, sequence length 131,072, four datums and 524,288 tokens per step, 16 GPUs, TP=1, PP=2, EP=8, CP=8, ETP=1, DP=1, and five optimizer steps. Ranks 0-7 are PP0 and ranks 8-15 are PP1.

Sources used for every configuration:

- The complete `result.json`.
- The complete `trainer_srun.log`.
- All 16 `memory.rank<N>.pickle` files, for 64 rank snapshots total.

Memory units are binary GiB. "Live" means `active_allocated` PyTorch blocks. "Allocator-active" also includes allocations awaiting asynchronous free. "Reserved" means mapped PyTorch caching-allocator memory. External allocations made directly by NCCL, NVSHMEM, CUDA, or a communication backend are outside these pickle measurements.

## Throughput and timing

The aggregate TPS from `result.json` is computed from mean control FB time and is the primary throughput metric. Cluster TPS is the per-GPU value multiplied by 16. Standard deviations below are sample standard deviations across only three control windows, so ranges are also shown.

| Backend | Control TPS/GPU | Cluster TPS | Delta vs alltoall | Control FB mean +/- SD | FB CV | FB range | Server-step mean +/- SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full alltoall | 769.381 | 12,310.1 | baseline | 42.590 +/- 0.356 s | 0.837% | 42.374-43.001 s | 40.454 +/- 1.672 s |
| DeepEP sms16 | 708.368 | 11,333.9 | -61.013 (-7.93%) | 46.258 +/- 0.065 s | **0.140%** | 46.186-46.310 s | 43.499 +/- 0.061 s |
| DeepEP sms20 | 734.860 | 11,757.8 | -34.521 (-4.49%) | 44.591 +/- 0.495 s | 1.110% | 44.299-45.163 s | 41.878 +/- 0.490 s |
| HybridEP sms16 | **837.550** | **13,400.8** | **+68.169 (+8.86%)** | **39.124 +/- 0.237 s** | 0.607% | 38.975-39.397 s | **36.429 +/- 0.236 s** |

Pairwise TPS outcomes:

| Comparison | TPS change |
|---|---:|
| DeepEP sms20 vs DeepEP sms16 | +3.74% |
| HybridEP sms16 vs DeepEP sms16 | +18.24% |
| HybridEP sms16 vs DeepEP sms20 | +13.97% |
| HybridEP sms16 vs alltoall | +8.86% |

DeepEP sms16 is the most repeatable over these three controls, but it is repeatably slow. The backend gaps are much larger than each run's internal window variation. DeepEP sms20 trades some repeatability for speed and still does not reach alltoall. HybridEP is both faster than alltoall and reasonably stable.

### Result aggregates

| Metric | Alltoall | DeepEP 16 | DeepEP 20 | HybridEP 16 |
|---|---:|---:|---:|---:|
| Control FB mean | 42.590 s | 46.258 s | 44.591 s | **39.124 s** |
| Control TPS/GPU | 769.381 | 708.368 | 734.860 | **837.550** |
| MFU3x | 7.004% | 6.448% | 6.690% | **7.624%** |
| HFU | 10.165% | 9.359% | 9.709% | **11.066%** |
| Traced FB | 44.285 s | 48.295 s | 48.042 s | **39.795 s** |
| Reported Kineto overhead | 3.98% | 4.40% | 7.74% | **1.72%** |
| Control optimizer mean | 1.036 s | 0.659 s | 0.659 s | **0.085 s** |

Optimizer timings are noisy and phase-dependent. Alltoall ranges from 0.118 to 2.245 s over controls, DeepEP sms16 from 0.046 to 0.967 s, and DeepEP sms20 from 0.126 to 0.968 s. HybridEP is the only run with consistently small control optimizer calls, 0.047-0.120 s. The FB/TPS result is the more reliable backend decision metric.

## Startup and one-time costs

"Launch to benchmark" is the interval from the first `torch.distributed.run` timestamp in the trainer log to `result.json.started`. The server-ready log lines do not carry timestamps, so this is a readiness proxy rather than a pure model-initialization measurement; it includes any health-poll/client gap.

| Backend | Launch to benchmark | Warmup FB | Warmup / control FB | Warmup FB excess | Warmup optimizer | Profile start + stop |
|---|---:|---:|---:|---:|---:|---:|
| Alltoall | 730.5 s (12:10.5) | 91.869 s | 2.16x | +49.279 s | 9.330 s | 12.27 s |
| DeepEP sms16 | **1467.8 s (24:27.8)** | 93.235 s | 2.02x | +46.976 s | 7.035 s | 11.59 s |
| DeepEP sms20 | **643.4 s (10:43.4)** | 110.328 s | 2.47x | +65.737 s | 10.583 s | 12.34 s |
| HybridEP sms16 | 681.8 s (11:21.8) | **128.281 s** | **3.28x** | **+89.157 s** | **5.936 s** | **10.55 s** |

Startup conclusions:

- HybridEP has the largest first-FB compilation/warmup tax, but then the lowest steady-state FB time. The log emits `torch._dynamo` graph breaks around `HybridEP` metadata setup before application startup, first on PP0 ranks at 09:04:48 and then PP1 ranks at 09:05:24. These are startup events, not steady-state failures.
- DeepEP sms20 starts fastest by the launch proxy, but its first FB is 17.094 s slower than DeepEP sms16 and 18.459 s slower than alltoall.
- DeepEP sms16 has a separate abandoned startup attempt beginning at 07:27:00. It reaches model loading and checkpoint-manager setup but never logs `Started server process` or `Application startup complete`; there is no traceback or OOM explaining the stop. The successful attempt begins at 07:38:53 and still has the longest readiness proxy at 24:27.8. This is real reliability/latency evidence, but the artifacts do not identify the root cause.
- Runtime-profiler setup and flush cost is similar, about 10.6-12.3 s, and does not explain steady-state control TPS because controls are unprofiled.

## Loss and gradient behavior

Each cell is `loss / grad_norm`.

| Window | Alltoall | DeepEP 16 | DeepEP 20 | HybridEP 16 |
|---|---:|---:|---:|---:|
| Warmup | 12.316959892 / 0.427006662 | 12.317238367 / 0.407057852 | 12.316255121 / 0.405797958 | 12.315897491 / 0.406140327 |
| Traced | 12.316998039 / 0.462981462 | 12.315963295 / 0.455741793 | 12.316884551 / 0.457037151 | 12.317191637 / 0.456513375 |
| Control 0 | 12.319063714 / 0.352792561 | 12.318967392 / 0.353146493 | 12.318316027 / 0.355182618 | 12.318688917 / 0.355384290 |
| Control 1 | 12.309552647 / 0.354250073 | 12.309829215 / 0.358770669 | 12.308676214 / 0.354469061 | 12.308987114 / 0.353778124 |
| Control 2 | 12.313551434 / 0.404852450 | 12.312328814 / 0.398569375 | 12.311562054 / 0.383395374 | 12.313093667 / 0.401886284 |

| Backend | Control loss mean +/- SD | Control loss range | Control grad mean +/- SD | Control grad range |
|---|---:|---:|---:|---:|
| Alltoall | 12.314055932 +/- 0.004775561 | 0.009511067 | 0.370631695 +/- 0.029645002 | 0.052059889 |
| DeepEP sms16 | 12.313708474 +/- 0.004722729 | 0.009138177 | 0.370162179 +/- 0.024761552 | 0.045422882 |
| DeepEP sms20 | 12.312851432 +/- 0.004947562 | 0.009639814 | 0.364349018 +/- 0.016498486 | 0.028926313 |
| HybridEP sms16 | 12.313589899 +/- 0.004869900 | 0.009701803 | 0.370349566 +/- 0.027323404 | 0.048108160 |

The phase-to-phase pattern is shared: traced grad norms are about 0.456-0.463, the first two controls are about 0.353-0.359, and the final control rises to 0.383-0.405. There is no exploding, vanishing, NaN, or monotonic backend-specific drift.

The numerical differences are nevertheless larger than floating-point noise:

- Maximum aligned loss spread: 0.001989 at control 2, about 0.016% of the loss.
- Maximum aligned grad-norm spread: 0.021457 at control 2, where DeepEP sms20 is 0.383395 versus alltoall at 0.404852, about 5.3% lower.
- Traced grad norms are much closer, with a maximum spread of 0.007240.

FP8 arithmetic and backend-specific reduction/routing order can plausibly produce these scalar differences, but aggregate loss and grad norms cannot prove tensor-level equivalence or perfect token routing. They do prove that every variant completes the short run with bounded, finite training signals.

## All-rank memory summary

The memory histories use `expandable_segments:True`. Every rank reaches the 1,000,000-event cap, so the early warmup history is truncated. Peaks below are reconstructed over the retained final-step ring and agree with the trainer log's reduced all-rank high-water values. Final live and reserve come directly from the complete snapshot segment state.

| Metric | Alltoall | DeepEP 16 | DeepEP 20 | HybridEP 16 |
|---|---:|---:|---:|---:|
| Live peak, all-rank mean | 146.415 | 144.303 | 144.306 | **143.457** |
| Live peak, fleet maximum | **153.079** | 153.063 | 153.063 | 153.063 |
| Allocator-active peak, all-rank mean | 146.415 | 145.559 | 145.559 | **143.457** |
| Allocator-active peak, fleet maximum | **153.079** | 153.063 | 153.063 | 153.063 |
| Final live, all-rank mean | 104.884 | 104.868 | 104.868 | 104.868 |
| Final live, fleet maximum | 110.610 | 110.594 | 110.594 | 110.594 |
| Final/peak reserved, all-rank mean | 150.609 | 151.202 | 151.299 | **147.653** |
| Reserved, fleet maximum | 157.264 | 160.061 | **160.178** | **156.498** |
| Inactive reserve at end, mean | 45.725 | 46.334 | 46.431 | **42.785** |
| Inactive reserve fraction | 30.36% | 30.64% | 30.69% | **28.98%** |
| Snapshot-marker live delta | 0 B on all ranks | 0 B on all ranks | 0 B on all ranks | 0 B on all ranks |

Memory deltas versus alltoall:

| Metric | DeepEP 16 | DeepEP 20 | HybridEP 16 |
|---|---:|---:|---:|
| Mean live peak | -2.112 GiB | -2.109 GiB | **-2.958 GiB** |
| Fleet max live peak | -0.016 GiB | -0.016 GiB | -0.016 GiB |
| Mean reserve | +0.593 GiB | +0.690 GiB | **-2.956 GiB** |
| Fleet max reserve | +2.797 GiB | +2.914 GiB | **-0.766 GiB** |
| Mean end cache | +0.609 GiB | +0.706 GiB | **-2.940 GiB** |

The memory verdict is stage-dependent. DeepEP and HybridEP materially reduce PP0 live peaks, but PP1 has the same approximately 153.0 GiB live ceiling for every backend. DeepEP then reserves more cache on PP1, producing the highest fleet reserve. HybridEP reduces reserve on both stages and is the only backend that improves both mean and maximum reserve versus alltoall.

DeepEP also has transient pending frees on PP0: allocator-active peak is about 2.5 GiB above user-live peak. DeepEP sms16 and sms20 are effectively identical in this respect. The gap disappears by snapshot end and is not a leak. Alltoall and HybridEP have no measurable pending-free peak gap.

### Pipeline-stage balance

| Stage / metric | Alltoall | DeepEP 16 | DeepEP 20 | HybridEP 16 |
|---|---:|---:|---:|---:|
| PP0 live-peak mean | 139.790 | 135.582 | 135.589 | **133.890** |
| PP0 live-peak range | 135.813-144.083 | 132.472-138.996 | 132.496-138.998 | **131.653-137.122** |
| PP0 live-peak spread | 8.270 | 6.524 | 6.503 | **5.470** |
| PP0 allocator-active mean | 139.790 | 138.093 | 138.095 | **133.890** |
| PP0 reserve mean | 144.067 | 142.544 | 142.795 | **139.060** |
| PP1 live-peak mean | 153.040 | 153.024 | 153.024 | 153.024 |
| PP1 live-peak spread | 0.045 | 0.045 | 0.045 | 0.045 |
| PP1 reserve mean | 157.151 | 159.861 | 159.802 | **156.247** |

HybridEP gives the best PP0 peak balance: its peak spread is 33.9% smaller than alltoall. DeepEP narrows the spread by about 21%. Increasing DeepEP from 16 to 20 SMs has no meaningful memory effect. PP1 is already tightly balanced and insensitive to dispatcher choice.

Final live memory is also tightly balanced within each stage. Alltoall ends at 99.157 GiB on normal PP0 ranks, 99.173 GiB on rank 0, and 110.610 GiB on every PP1 rank. All three alternative backends end at 99.141 GiB on normal PP0 ranks, 99.149 GiB on rank 0, and 110.594 GiB on every PP1 rank. The roughly 11.45 GiB stage split is model/pipeline state, not an EP imbalance.

### Per-rank live / peak / reserve

Each backend cell is `final live / retained live peak / final reserve`, in GiB.

| Rank | PP | Alltoall | DeepEP 16 | DeepEP 20 | HybridEP 16 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 99.173 / 139.705 / 143.434 | 99.149 / 135.184 / 142.227 | 99.149 / 135.222 / 142.031 | 99.149 / 133.240 / 138.898 |
| 1 | 0 | 99.157 / 138.491 / 142.182 | 99.141 / 135.030 / 141.152 | 99.141 / 135.008 / 140.898 | 99.141 / 133.165 / 137.727 |
| 2 | 0 | 99.157 / 136.014 / 140.424 | 99.141 / 132.472 / 139.082 | 99.141 / 132.496 / 140.664 | 99.141 / 131.653 / 136.066 |
| 3 | 0 | 99.157 / 142.023 / 146.947 | 99.141 / 137.244 / 144.785 | 99.141 / 137.236 / 144.746 | 99.141 / 135.285 / 140.383 |
| 4 | 0 | 99.157 / 140.601 / 144.584 | 99.141 / 135.724 / 143.750 | 99.141 / 135.729 / 143.730 | 99.141 / 133.808 / 141.066 |
| 5 | 0 | 99.157 / 144.083 / 148.588 | 99.141 / 138.996 / 146.055 | 99.141 / 138.998 / 146.895 | 99.141 / 137.122 / 142.492 |
| 6 | 0 | 99.157 / 141.589 / 146.012 | 99.141 / 137.199 / 144.336 | 99.141 / 137.201 / 144.199 | 99.141 / 135.196 / 140.812 |
| 7 | 0 | 99.157 / 135.813 / 140.367 | 99.141 / 132.809 / 138.965 | 99.141 / 132.822 / 139.199 | 99.141 / 131.653 / 135.031 |
| 8 | 1 | 110.610 / 153.079 / 157.205 | 110.594 / 153.063 / 160.061 | 110.594 / 153.063 / 159.865 | 110.594 / 153.063 / 156.283 |
| 9 | 1 | 110.610 / 153.034 / 157.244 | 110.594 / 153.018 / 160.041 | 110.594 / 153.018 / 159.787 | 110.594 / 153.018 / 156.244 |
| 10 | 1 | 110.610 / 153.034 / 157.225 | 110.594 / 153.018 / 159.924 | 110.594 / 153.018 / 159.768 | 110.594 / 153.018 / 156.127 |
| 11 | 1 | 110.610 / 153.034 / 157.146 | 110.594 / 153.018 / 159.768 | 110.594 / 153.018 / 159.592 | 110.594 / 153.018 / 156.322 |
| 12 | 1 | 110.610 / 153.034 / 157.010 | 110.594 / 153.018 / 159.670 | 110.594 / 153.018 / 160.178 | 110.594 / 153.018 / 156.186 |
| 13 | 1 | 110.610 / 153.034 / 157.225 | 110.594 / 153.018 / 159.865 | 110.594 / 153.018 / 159.885 | 110.594 / 153.018 / 156.322 |
| 14 | 1 | 110.610 / 153.034 / 157.264 | 110.594 / 153.018 / 159.885 | 110.594 / 153.018 / 159.475 | 110.594 / 153.018 / 156.498 |
| 15 | 1 | 110.610 / 153.034 / 156.893 | 110.594 / 153.018 / 159.672 | 110.594 / 153.018 / 159.867 | 110.594 / 153.018 / 155.990 |

## Leak and capacity analysis

- All 64 histories contain exactly 1,000,000 retained events and no OOM event.
- Each rank contains two snapshot markers bracketing the final control FB. Marker-to-marker live bytes are exactly identical on every rank in every run. The marker spans are about 43.24-43.33 s for alltoall, 47.25-47.33 s for DeepEP sms16, 45.31-45.41 s for DeepEP sms20, and 39.16-39.22 s for HybridEP sms16.
- Peak values plateau in the trainer logs: alltoall and both DeepEP runs stabilize by step 2-3, and HybridEP stabilizes by step 3. There is no monotonic step-over-step allocated or reserved growth through step 5.
- No run has an active-awaiting-free block at snapshot end. DeepEP's larger allocator-active peak is transient stream-ordered memory.
- The device has 267.69 GiB. The worst reserve is 160.178 GiB on DeepEP sms20, leaving about 107.51 GiB outside the PyTorch cache. None of these runs is close to a capacity OOM.
- All snapshots use expandable segments and end with tens of GiB in inactive reserve. That reserve is high-water cache, not live memory. There is no OOM or evidence that fragmentation blocked an allocation.

This is strong evidence against a PyTorch-managed leak over these five steps. It is not evidence about multi-hour retention, and it cannot see memory allocated outside PyTorch's caching allocator.

## Memory reporting caveat

`result.json.aggregates.peak_gpu_memory_bytes` is misleading for this PP2 comparison. It equals rank 0's final reserved total, not the maximum allocated or reserved memory across all ranks.

| Source / metric | Alltoall | DeepEP 16 | DeepEP 20 | HybridEP 16 |
|---|---:|---:|---:|---:|
| `result.json` reported "peak" | 143.434 | 142.227 | 142.031 | 138.898 |
| Correct fleet max live | 153.079 | 153.063 | 153.063 | 153.063 |
| Correct fleet max reserve | 157.264 | 160.061 | 160.178 | 156.498 |

Using only the JSON field would incorrectly claim that DeepEP lowers reserve and would overstate HybridEP's fleet maximum reduction. All-rank snapshots show the actual result: DeepEP increases fleet reserve, HybridEP lowers it modestly, and all variants leave the live ceiling essentially unchanged.

## Trainer-log health

Operationally common behavior:

- Every completed run reaches step 5.
- Every step reports 524,288 tokens and 524,284 loss tokens. The fixed four-token difference is consistent across backends and is not evidence of backend-specific token loss.
- No completed log contains a traceback, OOM, CUDA failure, NCCL failure, NaN/Inf report, or dropped-token report.
- Every completed log contains the same nonfatal profiler error, `External init callback must run in same thread as registerClient`; every run still emits a trace and completes.
- The logs emit `AccumulateGrad` stream-mismatch warnings across ranks. These warnings can introduce synchronization and affect absolute performance, but they are common to all four runs and do not explain HybridEP's relative win.
- `final_status.last_loss` and `final_status.grad_norm` are null in every JSON even though all five window records and optimizer logs contain valid values. This is a status-reporting limitation, not failed training.

Backend-specific behavior:

- HybridEP emits startup-time Dynamo graph-break warnings around `padded_num_tokens = int(max_num_tokens_across_ep.item())`. The run still initializes and its control windows are stable.
- DeepEP sms16 has the separate unexplained pre-server stall described above. The successful log has no CUDA/NCCL/runtime failure.
- DeepEP sms16 and sms20 produce nearly identical memory behavior. Raising the SM count improves steady TPS but does not fix DeepEP's PP1 reserve increase.

## Decision

Use **HybridEP sms16** for this full 131k Flex shape.

It is the only tested backend that simultaneously:

- Improves control TPS over alltoall, by 8.86%.
- Reduces mean and maximum PyTorch reserve.
- Produces the lowest PP0 live peak and narrowest PP0 rank spread.
- Preserves finite, bounded loss and gradient behavior.
- Shows zero steady marker-to-marker live growth on every rank.

The cost is a large one-time warmup FB, 128.281 s, and startup-time Dynamo graph breaks. For a long-running trainer, the 3.466 s steady FB saving versus alltoall repays HybridEP's additional 39.878 s warmup FB cost after roughly 12 steady steps. For very short jobs, that startup amortization matters; for sustained training, HybridEP sms16 is the correct choice from these artifacts.

DeepEP sms20 is preferable to sms16 if DeepEP must be used, but neither DeepEP configuration is justified over alltoall here: both are slower, both raise fleet reserve, and sms20 does not improve memory balance over sms16.
