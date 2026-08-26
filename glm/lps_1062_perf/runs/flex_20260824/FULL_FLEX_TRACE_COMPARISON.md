# GLM-5.2 131K full Flex trace comparison

## Executive conclusion

HybridEP at 16 SMs is the clear winner for this GLM-5.2 PP2/EP8/CP8, 131,072-token configuration:

| Backend | Control FB | Control tok/s/GPU | Relative to all-to-all |
|---|---:|---:|---:|
| HybridEP, 16 SMs | **39.124 s** | **837.550** | **+8.86% throughput** |
| all-to-all | 42.590 s | 769.381 | baseline |
| DeepEP, 20 SMs | 44.591 s | 734.860 | -4.49% throughput |
| DeepEP, 16 SMs | 46.258 s | 708.368 | -7.93% throughput |

The backend ordering comes directly from serialized dispatcher and pipeline-readiness costs:

| Backend | Dispatcher residency | Dispatcher overlap | Exposed dispatcher | PP `SendRecv` wait |
|---|---:|---:|---:|---:|
| HybridEP, 16 SMs | **7.008 s** | 0.000 s | **7.008 s** | **9.692 s** |
| all-to-all | 11.556 s | **0.735 s** | 10.822 s | 10.305 s |
| DeepEP, 20 SMs | 12.375 s | 0.000 s | 12.375 s | 11.437 s |
| DeepEP, 16 SMs | 13.044 s | 0.000 s | 13.044 s | 12.079 s |

HybridEP wins despite spending 4.725 s in `device_sync_kernel` polling. Its useful dispatch/combine work is only 1.548 s and its permutation/scan/update work is only 0.735 s. The complete HybridEP dispatcher is therefore 3.814 s shorter on the exposed rank-local critical path than all-to-all, 5.367 s shorter than DeepEP 20, and 6.036 s shorter than DeepEP 16.

The local advantage propagates into PP2 readiness. HybridEP has the shortest total pipeline-peer wait. Against all-to-all, the 3.814 s dispatcher saving plus the 0.612 s PP-wait saving accounts for 4.426 s of the 4.490 s traced forward/backward gap, or 98.6%. Against DeepEP 16, dispatcher plus PP savings account for 8.424 s of the 8.500 s traced gap, or 99.1%.

Further DeepEP SM tuning is not worthwhile as a production optimization while HybridEP is available. Raising DeepEP from 16 to 20 SMs improves payload kernels by 18.3% and control throughput by 3.74%, but notification/polling grows by 18.9% and absorbs most of the payload gain. At 20 SMs, DeepEP has 7.287 s of non-payload dispatcher work; even a physically impossible zero-time payload would leave it slightly slower than HybridEP's entire 7.008 s dispatcher. One 24-SM run is defensible only as a bounded diagnostic or HybridEP fallback test, not as the likely path to the best backend.

## Scope and method

Compared artifacts:

- `artifacts/full-131k-alltoall/result.json`
- `artifacts/full-131k-alltoall/trace/b300-1-izksekdp-0001_157551.1787642315821630974.pt.trace.json`
- `artifacts/full-131k-deepep-sms16/result.json`
- `artifacts/full-131k-deepep-sms16/trace/b300-1-izksekdp-0001_167311.1787645159221749551.pt.trace.json`
- `artifacts/full-131k-deepep-sms20/result.json`
- `artifacts/full-131k-deepep-sms20/trace/b300-1-izksekdp-0001_178292.1787647563426841854.pt.trace.json`
- `artifacts/full-131k-hybridep-sms16/result.json`
- `artifacts/full-131k-hybridep-sms16/trace/b300-1-izksekdp-0001_184807.1787648976440949488.pt.trace.json`

All runs use the same model and parallelism:

| Setting | Value |
|---|---:|
| Model | `zai-org/GLM-5.2-FP8` |
| Sequence length | 131,072 |
| GPUs | 16 |
| Tokens per step | 524,288 |
| Tensor parallel | 1 |
| Pipeline parallel | 2 |
| Expert parallel | 8 |
| Context parallel | 8 |
| Data parallel | 1 |

The only intended differences are dispatcher backend and DeepEP/HybridEP SM count. Each trace records one full profiled rank-0 step. All SQL was run with Perfetto `trace_processor` v57.2. Exact overlap uses Perfetto interval intersection followed by a per-dispatcher-call interval union, so simultaneous kernels on multiple streams are not double-counted.

Perfetto reported 6,234 / 6,284 / 6,237 / 5,765 overlapping-complete-event spills for all-to-all / DeepEP 16 / DeepEP 20 / HybridEP 16. Perfetto moved those events to overflow tracks and reports no data loss.

The three unprofiled control windows are the throughput source of truth. Profiler overhead differs materially across runs:

| Backend | Traced FB | Kineto overhead |
|---|---:|---:|
| HybridEP, 16 SMs | 39.795 s | 1.72% |
| all-to-all | 44.285 s | 3.98% |
| DeepEP, 16 SMs | 48.295 s | 4.40% |
| DeepEP, 20 SMs | 48.042 s | 7.74% |

This is why traced wall time is used to explain paths within a trace, not to replace the control-window ranking. In particular, the DeepEP 20 trace is disproportionately inflated relative to its controls.

## Dispatcher accounting

### Common decomposition

The full rank-local dispatcher paths are:

| Component | all-to-all | DeepEP 16 | DeepEP 20 | HybridEP 16 |
|---|---:|---:|---:|---:|
| Payload or transport | 10.437 s | 6.229 s | 5.087 s | **1.548 s** |
| Explicit notify/sync polling | included above | 5.120 s | 6.090 s | **4.725 s** |
| Permutation/sort/layout/scan | 1.119 s | 1.694 s | 1.197 s | **0.735 s** |
| **Raw dispatcher total** | 11.556 s | 13.044 s | 12.375 s | **7.008 s** |
| Overlap with other GPU kernels | **0.735 s** | 0.000 s | 0.000 s | 0.000 s |
| **Exposed dispatcher total** | 10.822 s | 13.044 s | 12.375 s | **7.008 s** |

All-to-all's NCCL `SendRecv` combines transfer, peer readiness, and network wait in one kernel, so its 10.437 s cannot be split into payload and barrier buckets. The DeepEP and HybridEP implementations expose their polling kernels separately.

### All-to-all

| Kernel class | Calls | Total | Mean | p50 | p90 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| EP NCCL `SendRecv` | 1,260 | 10.437 s | 8.284 ms | 4.520 ms | 12.022 ms | 21.003 ms | 2.127 s |
| `_sort_chunks_by_map_kernel` | 840 | 0.397 s | 0.473 ms | | | | 0.981 ms |
| `_permute_kernel` | 420 | 0.346 s | 0.823 ms | | | | 1.104 ms |
| `_unpermute_kernel` | 420 | 0.274 s | 0.652 ms | | | | 0.803 ms |
| `_make_chunk_sort_map_kernel` | 560 | 0.102 s | 0.182 ms | | | | 0.429 ms |

The 1,260 EP calls are three NCCL operations for each of 420 dispatch/combine cycles. Eleven calls over 50 ms contribute 3.982 s, including nine calls over 100 ms contributing 3.808 s. This is a readiness/transfer long tail, not uniform 8.3-ms communication.

All-to-all is the only backend that overlaps any dispatcher work with other GPU kernels. The exact overlap union is 0.735 s, or 6.36% of dispatcher residency. That reduces 11.556 s of raw dispatcher work to 10.822 s exposed on the critical path.

### DeepEP 16 versus 20 SMs

| DeepEP component | 16 SMs | 20 SMs | 20-SM change |
|---|---:|---:|---:|
| Dispatch payload | 2.405 s | **1.968 s** | -0.437 s |
| Combine payload | 3.825 s | **3.120 s** | -0.705 s |
| **Payload subtotal** | **6.229 s** | **5.087 s** | **-1.142 s (-18.3%)** |
| `cached_notify_combine` | **4.926 s** | 5.876 s | +0.950 s |
| Other notify kernels | 0.194 s | 0.214 s | +0.020 s |
| **Notify subtotal** | **5.120 s** | **6.090 s** | **+0.970 s (+18.9%)** |
| Permute | 0.550 s | 0.555 s | +0.004 s |
| Unpermute | 1.107 s | **0.605 s** | -0.502 s |
| Layout | 0.037 s | 0.037 s | unchanged |
| **Full dispatcher** | 13.044 s | **12.375 s** | **-0.669 s (-5.1%)** |

The 16-to-20 SM payload response is real and close to ideal bandwidth scaling. A 25% larger SM budget reduces payload time by 18.3%; perfect inverse scaling would reduce it by 20%. The end-to-end improvement is much smaller because polling moves in the wrong direction.

`cached_notify_combine` is the central DeepEP problem:

| Statistic | DeepEP 16 | DeepEP 20 |
|---|---:|---:|
| p50 | 5.461 ms | 6.131 ms |
| p90 | 12.964 ms | 12.666 ms |
| p99 | 103.126 ms | **288.658 ms** |
| Max | **1.171 s** | 0.817 s |
| Calls over 50 ms | 8 | 11 |
| Time in calls over 50 ms | 2.531 s | **3.334 s** |
| Calls over 100 ms | 6 | 9 |
| Time in calls over 100 ms | 2.371 s | **3.162 s** |

Twenty SMs shortens regular payload kernels but increases the number and total cost of long rank-readiness waits. This is why more SMs do not translate cleanly into the full dispatcher path.

DeepEP has zero measured overlap at either SM count. Its communication kernels run on a separate track, but Flex dependencies leave no temporal intersection with other GPU kernels. Reducing SMs would therefore not recover hidden compute overlap in these traces; it would only slow an already serialized payload. Increasing SMs can continue to shorten payload, but cannot directly fix notification waits, layout/permutation, or scheduling.

### HybridEP 16 SMs

| HybridEP component | Calls | Total | Mean | p50 | p90 | p99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dispatch payload | 420 | 0.747 s | 1.778 ms | 1.796 ms | 2.059 ms | 2.105 ms | 2.109 ms |
| Combine payload | 420 | 0.801 s | 1.908 ms | 1.911 ms | 2.086 ms | 2.191 ms | 2.210 ms |
| `device_sync_kernel<8>` | 1,680 | 4.725 s | 2.812 ms | 0.098 ms | 7.224 ms | 13.854 ms | 0.980 s |
| Permute | 420 | 0.332 s | 0.791 ms | | | | 1.328 ms |
| Unpermute | 420 | 0.388 s | 0.923 ms | | | | 1.592 ms |
| Scan/update | 1,120 | 0.015 s | | | | | 0.054 ms |

HybridEP's payload is regular and bounded. Its problem is explicitly the 4.725 s device-side polling bucket. Six calls over 50 ms contribute 1.961 s, including five calls over 100 ms contributing 1.901 s. The 0.980-s maximum confirms significant rank-arrival skew.

That wait is still cheaper than the alternatives in aggregate. HybridEP payload is 69.6% shorter than DeepEP 20 payload and its token permutation/scan path is 38.6% shorter. Compared with all-to-all, the entire HybridEP dispatcher is 35.2% shorter than all-to-all's exposed dispatcher.

HybridEP dispatcher kernels all run on the primary GPU track and have zero overlap with other kernels. The 7.008 s is therefore fully exposed. This is a remaining optimization opportunity, not a reason to reject HybridEP: its serialized path is already much shorter than every alternative.

## Host-visible Flex waits

Host synchronization spans mirror the GPU-side dispatcher waits and are not additive to the kernel totals:

| Backend | Flex-specific host wait | Calls | Total | Mean | Max |
|---|---|---:|---:|---:|---:|
| all-to-all | `cudaEventSynchronize` | 280 | 2.499 s | 8.924 ms | 96.460 ms |
| DeepEP 16 | `FusedDispatch` | 280 | 2.845 s | 10.162 ms | 80.989 ms |
| DeepEP 20 | `FusedDispatch` | 280 | 2.931 s | 10.468 ms | 135.489 ms |
| HybridEP 16 | `HybridEPDispatch` | 280 | **0.158 s** | **0.565 ms** | **1.916 ms** |

DeepEP `FusedCombine` is only 43 / 42 ms total at 16 / 20 SMs. HybridEP `HybridEPCombine` is 30 ms total. The important host-visible difference is dispatch: HybridEP does not force the host to sit inside multi-millisecond dispatch spans the way all-to-all and DeepEP do.

The traces also contain 14K-16K broad `cudaStreamSynchronize` calls. Their inclusive totals overlap the GPU work and cover more than the dispatcher, so adding them to dispatcher residency would double count the same critical path.

## Pipeline serialization

Each trace has a second NCCL `SendRecv` track with exactly ten calls. This track is PP P2P, not EP traffic. Its duration closely matches `cudaDeviceSynchronize` on the host and has zero overlap with any other GPU kernel in all four traces. These are serialized pipeline-peer readiness waits, not evidence that a pipeline tensor requires 5-7 seconds to transfer.

| Backend | PP total | Four largest calls | Top-four share |
|---|---:|---|---:|
| HybridEP, 16 SMs | **9.692 s** | 5.652, 2.848, 0.846, 0.311 s | 99.6% |
| all-to-all | 10.305 s | 6.554, 1.753, 1.122, 0.842 s | 99.7% |
| DeepEP, 20 SMs | 11.437 s | 7.019, 2.179, 1.276, 0.930 s | 99.7% |
| DeepEP, 16 SMs | 12.079 s | 7.161, 2.575, 1.251, 1.058 s | 99.7% |

The other six calls are roughly 0-8 ms each. Four pipeline bubbles determine effectively the entire PP wait budget.

Relative to HybridEP, extra serialized PP wait is:

| Backend | Extra PP wait versus HybridEP |
|---|---:|
| all-to-all | +0.612 s |
| DeepEP, 20 SMs | +1.745 s |
| DeepEP, 16 SMs | +2.387 s |

The PP message shape did not change. The backend changes when ranks and pipeline stages reach those messages. Faster, more regular local MoE work lets the peer become ready sooner; slow or long-tailed dispatcher polling appears downstream as longer PP `SendRecv` residency.

## MoE checkpoint localization

Each trace contains 140 MoE checkpoint calls and 12 non-MoE checkpoint calls in both forward/recompute and backward. A checkpoint is classified as MoE when it contains `_AllToAll`, `FusedDispatch`, or `HybridEPDispatch` for its backend.

| Checkpoint group | all-to-all | DeepEP 16 | DeepEP 20 | HybridEP 16 |
|---|---:|---:|---:|---:|
| MoE forward/recompute, mean | 54.522 ms | 64.391 ms | 63.290 ms | **49.920 ms** |
| MoE backward, mean | 152.611 ms | 160.060 ms | 160.709 ms | **129.772 ms** |
| Non-MoE backward, mean | 96.475 ms | 97.085 ms | 95.985 ms | 97.197 ms |

Non-MoE backward is stable within 1.2 ms. The meaningful differences are localized to MoE checkpoints. Relative to all-to-all, HybridEP saves 0.644 s across MoE forward/recompute checkpoints and 3.197 s across MoE backward checkpoints, 3.842 s total. That agrees with the independently measured 3.814-s exposed dispatcher saving.

## Why the ranking is 838, 769, and 735 tok/s/GPU

### Why HybridEP reaches 838

HybridEP turns the complete exposed dispatcher from 10.8-13.0 s into 7.0 s. Its dispatch/combine payload is only 1.55 s, permutation and metadata are 0.74 s, and the remaining 4.72 s is synchronization polling. It then reaches PP communication earlier and has the lowest PP wait total at 9.69 s. The local dispatcher saving and downstream PP saving explain essentially all of its traced advantage over all-to-all and DeepEP 16.

### Why all-to-all reaches 769

All-to-all spends 10.44 s in EP `SendRecv` plus 1.12 s in sorting/permutation. It recovers 0.735 s through actual overlap, the only backend to do so, and its PP bubbles total 10.30 s. That is enough to beat DeepEP even though DeepEP's bulk payload kernels are faster in isolation.

### Why the best DeepEP only reaches 735

DeepEP 20 reduces payload to 5.09 s, but then spends 6.09 s in notification/polling and another 1.20 s in permutation/layout. Nothing overlaps, so all 12.37 s is exposed. Its PP wait is also 1.13 s longer than all-to-all and 1.75 s longer than HybridEP. The payload kernel improvement is real, but it attacks less than half of DeepEP's dispatcher and worsens the long-tail readiness bucket.

## Is further DeepEP SM tuning worthwhile?

Not as the production path while HybridEP is available.

The positive evidence for more SMs is narrow but real:

- DeepEP payload falls from 6.229 s at 16 SMs to 5.087 s at 20 SMs, an 18.3% reduction.
- Full dispatcher falls by 0.669 s, or 5.1%.
- PP wait falls by 0.642 s, or 5.3%.
- Control FB falls by 1.668 s and throughput rises from 708.4 to 734.9 tok/s/GPU, a 3.74% gain.

The stopping evidence is stronger:

- Notify/polling rises by 0.970 s and `cached_notify_combine` p99 grows from 103 to 289 ms.
- DeepEP 20 remains 4.49% behind all-to-all and 13.97% behind HybridEP in control throughput.
- DeepEP 20 has 6.090 s of notification plus 1.197 s of permutation/layout. This 7.287-s non-payload floor is already above HybridEP's complete 7.008-s dispatcher.
- DeepEP has zero overlap, so the SM knob cannot reveal hidden overlap; it can only shorten the 5.087-s payload bucket.
- PP readiness remains worse than both all-to-all and HybridEP.

An optimistic 24-SM bound assumes perfect inverse scaling from the measured 20-SM payload and no regression elsewhere:

```text
20-SM payload                         5.087 s
Ideal 24-SM payload = 5.087 * 20/24  4.240 s
20-SM non-payload                     7.287 s
Ideal 24-SM dispatcher               11.527 s
All-to-all exposed dispatcher        10.822 s
Remaining local gap                  +0.705 s
```

Even this favorable bound does not reach all-to-all locally, and it assumes the notification tail stops getting worse. It cannot approach HybridEP. A single 24-SM run is useful only if DeepEP must remain supported and the goal is to map the curve. Stop if it does not reduce both full dispatcher time and PP wait; do not continue a broad SM sweep based only on payload-kernel speedup.

The higher-value work is to reduce readiness polling and restore overlap. For HybridEP specifically, the 4.725-s `device_sync_kernel` bucket and four PP bubbles are now the largest backend-local opportunities. HybridEP is already fastest without solving them.

## Limitations

- Each trace is one step from rank 0, not all ranks. Polling and PP `SendRecv` durations reveal rank-arrival skew but cannot identify which peer or rank caused it.
- The control means come from three windows. They are consistent enough to establish the backend ordering, but a longer production run would tighten confidence intervals.
- NCCL all-to-all `SendRecv` does not separate useful transfer from waiting. Comparisons use full exposed kernel residency, which is the relevant critical-path cost but not a pure transport benchmark.
- Cross-run traced FB deltas involving DeepEP 20 are distorted by its 7.74% Kineto overhead. The control result, not traced FB, establishes that 20 SMs is better than 16 SMs.

## Recommendation

Use HybridEP with 16 SMs for this GLM-5.2 131K PP2/EP8/CP8 configuration.

Do not invest in a broad DeepEP SM sweep. If DeepEP fallback performance matters, run one matched 24-SM full control plus trace and require full dispatcher residency and PP wait to improve, not just dispatch/combine payload. Otherwise, spend the profiling budget on multi-rank HybridEP traces to localize the six long `device_sync_kernel` outliers and the four PP readiness bubbles.
