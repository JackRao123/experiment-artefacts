# HybridEP Flex dispatcher trace analysis

## Recommendation

Use **HybridEP with 16 SMs** (`moe_flex_dispatcher_num_sms: 16`) for this EP8 Flex configuration.

The 32-SM trace has the shortest *single observed* rank-local HybridEP synchronization path, but that is not enough to recommend 32 SMs:

- Useful HybridEP dispatch/combine time does not improve above 16 SMs: 6.196 ms at 16, 6.348 ms at 20, and 6.245 ms at 32.
- The apparent 32-SM win is entirely in `device_sync_kernel` polling: 16.073 ms versus 104.145-124.100 ms. Those totals are dominated by two or three rank-arrival outliers, not payload kernels, and are not monotonic from 16 to 20 SMs.
- The 32-SM trace is heavily confounded by 772.646 ms of unrelated NCCL all-reduce wait and 106% Kineto overhead. Its whole-step wall time is not a valid dispatcher comparison.
- The unprofiled control windows are stable and effectively tied at 16/20 SMs, while 32 SMs is 5.2% slower than 16 SMs in mean forward/backward time and much more variable.

In short, 16 SMs performs the same useful HybridEP work with the smallest tested SM budget. The trace does not establish that spending 32 SMs produces a repeatable end-to-end gain.

## Scope and method

I queried the six completed rank-0 Kineto traces with `trace_processor`:

| Backend | SM count | Trace |
|---|---:|---|
| all-to-all | n/a | `artifacts/debug-ep8-alltoall/trace/*.pt.trace.json` |
| DeepEP | 16 | `artifacts/debug-ep8-deepep-sms16/trace/*.pt.trace.json` |
| DeepEP | 20 | `artifacts/debug-ep8-deepep-sms20/trace/*.pt.trace.json` |
| HybridEP | 16 | `artifacts/debug-ep8-hybridep-sms16/trace/*.pt.trace.json` |
| HybridEP | 20 | `artifacts/debug-ep8-hybridep-sms20/trace/*.pt.trace.json` |
| HybridEP | 32 | `artifacts/debug-ep8-hybridep-sms32/trace/*.pt.trace.json` |

`debug-ep8-hybridep-sms16-failed` contains only a trainer log and no trace, so it is excluded. Each usable trace contains one profiled training step on one rank. The SQL grouped `slice` rows by backend kernel class, inspected individual event timelines, followed synchronization call parents, and separated EP kernels from unrelated NCCL collectives.

Perfetto reported 83-93 `slice_spill_overlapping_complete_event` import warnings per trace. Perfetto states that these events are moved to overflow tracks without data loss; the kernel slices used below remain queryable.

## HybridEP SM-count results

Each HybridEP trace contains three dispatch kernels, three combine kernels, and twelve `hybrid_ep::device_sync_kernel<8>` calls. The two `true` dispatches are forward/recompute and the one `false` dispatch is backward.

| SMs | Forward dispatch, avg | Backward dispatch | Combine total | Useful dispatch + combine | Device-sync wait total | Largest wait |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | **1.038 ms** | **0.959 ms** | 3.160 ms | **6.196 ms** | 124.100 ms | 68.068 ms |
| 20 | 1.092 ms | 1.009 ms | 3.154 ms | 6.348 ms | 104.145 ms | 86.902 ms |
| 32 | 1.058 ms | 0.981 ms | **3.148 ms** | 6.245 ms | **16.073 ms** | **9.733 ms** |

The useful kernels are flat. Relative to 16 SMs, useful payload time is 2.5% slower at 20 SMs and 0.8% slower at 32 SMs. There is no payload-throughput evidence for increasing the SM count.

The wait totals look different, but they are long-tail synchronization rather than steady transfer work:

| SMs | Dominant `device_sync_kernel` waits | Share of all device-sync time |
|---:|---|---:|
| 16 | 68.068 ms + 50.802 ms | 95.8% |
| 20 | 86.902 ms + 11.997 ms | 95.0% |
| 32 | 9.733 ms + 3.620 ms + 2.453 ms | 98.3% |

The 20-SM maximum wait is worse than the 16-SM maximum despite the larger budget. This is characteristic of rank arrival skew or dependency polling, not a clean SM-bandwidth curve. A single rank and a single step cannot show whether the 32-SM reduction is repeatable across ranks.

HybridEP's other dispatcher-local kernels do not explain the difference. `permute_kernel + unpermute_kernel` total 2.646 / 2.592 / 2.602 ms at 16 / 20 / 32 SMs, and scan/update kernels are sub-0.1 ms. Host-side synchronization directly under `HybridEPDispatch` is also small: 0.282 / 0.347 / 0.667 ms.

All HybridEP dispatch, combine, and device-sync kernels execute on the same main GPU track in these traces. They do not overlap primary-stream work, so exposed `device_sync_kernel` tails are real critical-path stalls. That makes the 32-SM observation worth retesting, but it does not make it causal or repeatable.

## Backend comparison

The following table isolates backend payload kernels and explicit backend polling/wait kernels. It excludes generic token permutation/layout kernels and excludes non-EP NCCL collectives. NCCL SendRecv does not expose transfer versus wait separately, so all-to-all's number is the combined kernel residency.

| Backend | Payload/transport kernels | Explicit notify/sync waits | Total backend kernel residency |
|---|---:|---:|---:|
| all-to-all | 36.072 ms SendRecv | not separable | 36.072 ms |
| DeepEP 16 SM | 24.282 ms | 77.256 ms | 101.538 ms |
| DeepEP 20 SM | **19.677 ms** | 87.430 ms | 107.107 ms |
| HybridEP 16 SM | **6.196 ms** | 124.100 ms | 130.296 ms |
| HybridEP 20 SM | 6.348 ms | 104.145 ms | 110.493 ms |
| HybridEP 32 SM | 6.245 ms | **16.073 ms** | **22.318 ms** |

Observations:

- HybridEP payload kernels are much shorter than DeepEP payload kernels or NCCL SendRecv, but 16/20-SM HybridEP loses that advantage to device-sync polling.
- DeepEP shows an actual payload response to more SMs: dispatch/combine falls from 24.282 to 19.677 ms. Its notification time rises enough to erase the gain. HybridEP shows no corresponding payload response.
- HybridEP 32 SM is the only observed HybridEP trace whose backend residency beats all-to-all and DeepEP. It is 38.1% below all-to-all and 78.0% below DeepEP 16 SM in this one rank-local sample.
- That isolated 32-SM result comes from shorter rank waits, not faster dispatch/combine. It therefore needs multi-rank repeated evidence before being treated as an SM-count effect.

## Trace confounders

Whole-step trace time must not be used to select the HybridEP SM count here:

| HybridEP SMs | `ProfilerStep#0` | Non-EP NCCL all-reduce | Reported Kineto overhead |
|---:|---:|---:|---:|
| 16 | 1042.362 ms | 23.160 ms | 17.0% |
| 20 | 1479.454 ms | 69.754 ms | 71.3% |
| 32 | 1822.855 ms | 772.646 ms | 106.5% |

The 32-SM trace has a 516.934 ms F32 all-reduce and a 254.825 ms U64 all-reduce. These are downstream rank-arrival waits, not HybridEP transport. The same problem appears in the sibling DeepEP traces: non-EP all-reduce is 1.644 ms at 16 SMs and 151.153 ms at 20 SMs. This run-to-run skew is why dispatcher-local kernels and waits are separated from step wall above.

The unprofiled controls provide the necessary SM-pressure sanity check:

| HybridEP SMs | Mean control FB | Control TPS/GPU | Control FB range |
|---:|---:|---:|---:|
| 16 | 826.486 ms | 9911.8 | 824.807-829.561 ms |
| 20 | **825.338 ms** | **9925.6** | 821.491-829.720 ms |
| 32 | 869.658 ms | 9419.8 | 824.952-905.409 ms |

Sixteen and twenty SMs differ by only 0.14%, well inside run variance. Thirty-two SMs is 5.2% slower than 16 SMs and has a 9.3% within-run range, versus 0.6% at 16 SMs. That contradicts treating the single 32-SM wait sample as a production win.

## Decision

Set HybridEP to **16 SMs** now.

Twenty SMs is dominated: it does not accelerate HybridEP payload kernels and produces no measurable control-step improvement over 16. Thirty-two SMs is an interesting communication-only retest candidate because its observed device-sync tails are far smaller, but this dataset does not show a repeatable end-to-end benefit and does show a larger SM budget plus worse control performance.

Confidence is **moderate** because only rank 0 and one traced step per configuration are available. To reconsider 32 SMs, capture several steps from all eight EP ranks and compare the rank-wide maximum duration of each dispatch/sync/combine cycle. Recommend 32 only if the lower `device_sync_kernel` tail persists across ranks and the unprofiled forward/backward mean improves.
