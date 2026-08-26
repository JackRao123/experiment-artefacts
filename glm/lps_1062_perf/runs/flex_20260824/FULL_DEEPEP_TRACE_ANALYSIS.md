# GLM-5.2 131K Flex dispatcher: all-to-all vs DeepEP (16 SMs)

## Executive conclusion

At this 16-GPU, PP2/EP8/CP8, 131,072-token configuration, DeepEP is slower because its Flex dispatcher does more serialized work and loses the small amount of EP/compute overlap present in the all-to-all run. That local delay is then amplified as longer pipeline-peer waits.

The trace-level critical-path accounting is:

| Contribution | DeepEP penalty |
|---|---:|
| Extra exposed Flex dispatcher time | +2.223 s |
| Extra serialized pipeline `SendRecv` wait | +1.775 s |
| Accounted total | **+3.998 s** |
| Measured traced forward/backward gap | **+4.009 s** |
| Residual | 0.012 s |

This explains 99.7% of the traced forward/backward gap without invoking unrelated model compute.

The steady-state control result is **769.381 tok/s/GPU for all-to-all versus 708.368 tok/s/GPU for DeepEP**, so DeepEP is 61.013 tok/s/GPU, or 7.93%, lower. Equivalently, its forward/backward takes 3.668 s, or 8.61%, longer.

**Recommendation:** keep the all-to-all Flex backend for this 131K configuration. A DeepEP SM sweep upward from 16 is worth running as a diagnostic, but changing SMs alone is not a convincing path to full parity. It can accelerate the 6.230 s payload-kernel portion, but it does not directly fix 5.120 s of notification/barrier work, 1.694 s of token movement/layout, the observed zero overlap, or the pipeline scheduling waits. Test 20, 24, and optionally 32 SMs; do not lower the SM count.

## Inputs and comparability

Artifacts compared:

- `artifacts/full-131k-alltoall/result.json`
- `artifacts/full-131k-alltoall/trace/b300-1-izksekdp-0001_157551.1787642315821630974.pt.trace.json`
- `artifacts/full-131k-deepep-sms16/result.json`
- `artifacts/full-131k-deepep-sms16/trace/b300-1-izksekdp-0001_167311.1787645159221749551.pt.trace.json`

Both runs use:

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

Each trace is the full recorded rank-0 step: 740.2 MB for all-to-all and 767.9 MB for DeepEP. Queries were run through Perfetto `trace_processor`. Perfetto reported 6,234 and 6,284 overlapping-complete-event spills respectively; it explicitly reports that these events were moved to overflow tracks and no data was lost.

The result JSON control windows, rather than profiler throughput, are the throughput source of truth. The trace is used to explain the delta. Kineto overhead is similar but nonzero: 3.98% for all-to-all and 4.40% for DeepEP.

## Result JSON comparison

| Metric | All-to-all | DeepEP, 16 SMs | DeepEP delta |
|---|---:|---:|---:|
| Control FB mean | 42.590 s | 46.258 s | **+3.668 s (+8.61%)** |
| Control throughput | 769.381 tok/s/GPU | 708.368 tok/s/GPU | **-61.013 (-7.93%)** |
| Traced FB | 44.285 s | 48.295 s | **+4.009 s (+9.05%)** |
| Kineto overhead | 3.98% | 4.40% | +0.42 pp |

The DeepEP controls are tightly clustered at 709.477, 707.572, and 708.058 tok/s/GPU. The all-to-all controls are 762.023, 772.932, and 773.295 tok/s/GPU. The backend gap is much larger than window noise.

## Flex dispatcher GPU cost

### All-to-all

The EP communication is a dedicated `SendRecv` stream, distinct from the pipeline `SendRecv` stream.

| Dispatcher component | Calls | Summed GPU time |
|---|---:|---:|
| EP `ncclDevKernel_SendRecv` on EP track | 1,260 | 10.437 s |
| `_sort_chunks_by_map_kernel` | 840 | 0.397 s |
| `_permute_kernel` | 420 | 0.346 s |
| `_unpermute_kernel` | 420 | 0.274 s |
| `_make_chunk_sort_map_kernel` | 560 | 0.102 s |
| **Full dispatcher total** | | **11.556 s** |

The EP `SendRecv` distribution is broad because an NCCL kernel includes both transfer and peer/network waiting:

| Statistic | Duration |
|---|---:|
| p50 | 4.521 ms |
| p90 | 12.047 ms |
| p99 | 21.673 ms |
| Mean | 8.284 ms |
| Max | 2.127 s |

### DeepEP, 16 SMs

All DeepEP intranode kernels are on one dispatcher track. The payload kernels are regular and bounded; the notification/combine wait has the long tail.

| Dispatcher component | Calls | Summed GPU time | Mean per call |
|---|---:|---:|---:|
| `deep_ep::intranode::dispatch` | 420 | 2.405 s | 5.726 ms |
| `deep_ep::intranode::combine` | 420 | 3.825 s | 9.106 ms |
| `deep_ep::intranode::cached_notify_combine` | 420 | 4.926 s | 11.729 ms |
| `deep_ep::intranode::notify_dispatch` | 280 | 0.190 s | 0.678 ms |
| `deep_ep::intranode::cached_notify_dispatch` | 140 | 0.004 s | 0.032 ms |
| `_permute_kernel` | 420 | 0.550 s | 1.310 ms |
| `_unpermute_kernel` | 420 | 1.107 s | 2.636 ms |
| `deep_ep::layout::get_dispatch_layout` | 280 | 0.037 s | 0.132 ms |
| **Full dispatcher total** | | **13.044 s** | |

The DeepEP core splits into:

- **Payload dispatch + combine:** 6.230 s.
- **Notification/barrier path:** 5.120 s.
- **Permutation/layout:** 1.694 s.

The payload kernels are stable:

| Kernel | p50 | p90 | p99 | Max |
|---|---:|---:|---:|---:|
| Dispatch | 5.717 ms | 6.137 ms | 6.338 ms | 6.395 ms |
| Combine | 9.196 ms | 9.886 ms | 9.931 ms | 9.952 ms |

`cached_notify_combine` is not stable in the same way:

| Statistic | Duration |
|---|---:|
| p50 | 5.465 ms |
| p90 | 13.195 ms |
| p99 | 194.334 ms |
| Mean | 11.729 ms |
| Max | 1.171 s |

That long tail is the signature of a barrier/peer-readiness wait, not simply an under-provisioned bulk-copy kernel. Increasing SMs may make the preceding payload arrive sooner, but it does not make this 4.926 s bucket linearly SM-scalable.

### Raw dispatcher delta

| Component | All-to-all | DeepEP | DeepEP delta |
|---|---:|---:|---:|
| EP communication / DeepEP core | 10.437 s | 11.350 s | +0.913 s |
| Permutation, sorting, and layout | 1.119 s | 1.694 s | +0.575 s |
| **Raw dispatcher wall time** | **11.556 s** | **13.044 s** | **+1.488 s** |

DeepEP's 6.230 s payload path is faster than the complete all-to-all NCCL EP path in isolation, but its 5.120 s notification/barrier path more than consumes that advantage. Its token permutation is also materially slower, especially `_unpermute_kernel` at 1.107 s versus 0.274 s.

## Waits and synchronization

### Host-side waits

All-to-all has 280 `cudaEventSynchronize` calls nested under forward/recompute and backward checkpoint functions:

- Total: 2.499 s.
- Mean: 8.924 ms.
- Maximum: 96.460 ms.

DeepEP has 280 `FusedDispatch` CPU spans:

- Total: 2.845 s.
- Mean: 10.162 ms.
- Maximum: 80.989 ms.

The DeepEP `FusedCombine` host cost is only 43 ms total. These host waits are not additive to the GPU totals above: they are the CPU waiting for the already-counted dispatcher work. Adding them would double count the same critical path.

### Pipeline-peer waits

The traces contain a second `SendRecv` track with exactly 10 calls in each run. This is pipeline P2P, not EP all-to-all. Four calls dominate its duration:

| Long pipeline wait | All-to-all | DeepEP | Delta |
|---|---:|---:|---:|
| 1 | 6.554 s | 7.161 s | +0.608 s |
| 2 | 0.842 s | 1.058 s | +0.216 s |
| 3 | 1.122 s | 1.251 s | +0.129 s |
| 4 | 1.753 s | 2.575 s | +0.822 s |
| **All 10 calls** | **10.305 s** | **12.079 s** | **+1.775 s** |

These kernels coincide with `cudaDeviceSynchronize` intervals and have zero overlap with other GPU kernels on the recorded rank. They are mostly waiting for the peer pipeline stage to become ready, not spending 6-7 seconds transferring a pipeline tensor. DeepEP makes each major bubble longer, with the last bubble accounting for 0.822 s of the increase.

This is a propagated Flex cost: slower or less-overlapped MoE work changes pipeline-stage readiness and exposes more P2P wait. It is not evidence that DeepEP changed pipeline message size or P2P bandwidth.

## Overlap and critical-path exposure

I computed interval unions rather than summing all kernel durations. A dispatcher interval is "overlapped" only when another GPU kernel is active at the same timestamp.

| Dispatcher interval metric | All-to-all | DeepEP |
|---|---:|---:|
| Dispatcher wall union | 11.556 s | 13.044 s |
| Overlap with other GPU kernels | 0.735 s | **0.000 s** |
| Overlap fraction | 6.36% | **0.00%** |
| Exposed dispatcher time | 10.822 s | 13.044 s |

For EP communication alone, all-to-all overlaps 0.734 of 10.437 s (7.03%). DeepEP's entire 11.350 s intranode path has zero overlap. Including permutation/layout does not change the DeepEP result: the full 13.044 s dispatcher remains serialized.

Therefore the exposed dispatcher penalty is:

```text
DeepEP exposed dispatcher       13.044 s
- all-to-all exposed dispatcher 10.822 s
= local exposed penalty          2.223 s
```

The raw dispatcher penalty is only 1.488 s. The remaining 0.735 s comes from losing the overlap that all-to-all achieved.

The pipeline P2P tracks are also fully exposed in both traces. Adding their difference gives:

```text
Extra exposed dispatcher time       2.223 s
Extra exposed pipeline P2P wait     1.775 s
Total accounted trace penalty       3.998 s
Measured traced FB penalty          4.009 s
Residual                            0.012 s
```

This is the central result. The 708 versus 769 tok/s/GPU outcome is not caused by one very slow DeepEP copy kernel. It is the combination of:

1. More dispatcher work, especially notification/barrier and unpermute.
2. No overlap for any DeepEP dispatcher kernel on the recorded rank.
3. Longer downstream pipeline waits after that serialized work.

## MoE-layer localization

Checkpoint functions containing `_AllToAll` or `FusedDispatch` isolate the MoE layers. There are 140 MoE checkpoint calls and 12 non-MoE calls in both forward and backward.

| Checkpoint group | All-to-all mean | DeepEP mean | DeepEP delta |
|---|---:|---:|---:|
| MoE forward/recompute | 54.522 ms | 64.391 ms | +9.869 ms per layer |
| MoE backward | 152.611 ms | 160.060 ms | +7.449 ms per layer |
| Non-MoE backward | 96.475 ms | 97.085 ms | +0.610 ms per layer |

Across the 140 MoE calls, DeepEP adds 1.382 s in forward/recompute checkpoints and 1.043 s in backward checkpoints. Non-MoE backward is effectively unchanged. This confirms that the local delta is in the Flex/MoE path rather than unrelated attention or dense compute.

## Could changing DeepEP SMs recover the gap?

### What the SM knob can affect

The 16-SM setting directly sizes DeepEP's high-throughput dispatch/combine work. Those two payload kernels total 6.230 s and are the part most likely to improve with more SMs.

The trace does not support reducing SMs to improve compute overlap. DeepEP shows zero temporal overlap with any other GPU kernel, so there is no concurrently running compute from which 16 SMs are being stolen. Fewer SMs would primarily slow serialized payload work.

The SM knob does not directly solve:

- 4.926 s in long-tailed `cached_notify_combine` barrier/readiness waits.
- Another 0.194 s of notify work.
- 1.694 s in permutation/layout kernels.
- Zero dispatcher/compute overlap caused by scheduling/dependencies.
- Pipeline-peer readiness waits, except indirectly if both stages finish their MoE work sooner.

### Idealized payload-only scaling

The following is an optimistic arithmetic bound. It assumes dispatch and combine scale perfectly with SM count and every other cost is unchanged.

| SMs | Ideal payload time | Payload saving vs 16 | DeepEP local dispatcher after saving | Local gap vs all-to-all exposed dispatcher |
|---:|---:|---:|---:|---:|
| 16 | 6.230 s | 0.000 s | 13.044 s | +2.223 s |
| 20 | 4.984 s | 1.246 s | 11.798 s | +0.976 s |
| 24 | 4.153 s | 2.077 s | 10.968 s | +0.146 s |
| 32 | 3.115 s | 3.115 s | 9.929 s | -0.893 s |

This bound is favorable to DeepEP. Real scaling will be lower because NVLink traffic, token imbalance, barriers, and fixed work do not scale linearly with SM count.

To recover the 3.668 s control gap from the 6.230 s payload kernels alone requires a 2.43x payload speedup, equivalent to roughly 39 SMs under perfect linear scaling. Even an ideal 16-to-32 doubling saves only 3.115 s if pipeline waits do not also contract.

Twenty-four SMs could plausibly get the local dispatcher close to parity if payload scaling is unusually good. Full throughput parity would additionally require the 1.775 s pipeline-wait penalty to shrink substantially. That compound outcome is possible enough to measure, but the current trace does not justify expecting it.

### Recommended sweep and acceptance criteria

Run matched control windows at:

- 20 SMs.
- 24 SMs.
- 32 SMs only if 24 continues to improve payload kernels without increasing other stalls.

For each point, require all of the following:

- `dispatch + combine` drops materially from 6.230 s.
- `cached_notify_combine` does not grow and ideally loses its p99 tail.
- Full dispatcher exposed time approaches or beats 10.822 s.
- Pipeline P2P wait approaches the all-to-all 10.305 s total.
- Three untraced control windows approach or exceed 769 tok/s/GPU.

If 24 SMs does not reduce both payload time and pipeline waits, stop tuning this knob. The remaining problem is dispatcher synchronization/scheduling and overlap, not SM allocation.

## Final recommendation

Use all-to-all for GLM-5.2 at 131K with this PP2/EP8/CP8 topology. It wins by 61 tok/s/GPU and the traces explain why:

- All-to-all exposes 10.822 s of Flex dispatcher work; DeepEP exposes 13.044 s.
- DeepEP spends 5.120 s in notification/barrier work and 1.694 s in permutation/layout.
- All-to-all overlaps 0.735 s of dispatcher work; DeepEP overlaps none.
- DeepEP adds 1.775 s of serialized pipeline-peer wait.

An upward DeepEP SM sweep is reasonable, especially because 16 is conservative, but treat it as an experiment. The robust fix would need to reduce barrier/notify wait and restore overlap, not merely allocate more SMs to already-serialized payload kernels.
