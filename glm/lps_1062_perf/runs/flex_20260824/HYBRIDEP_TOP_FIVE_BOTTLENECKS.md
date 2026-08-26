# HybridEP full-trace bottleneck ranking

## Scope and method

This analysis covers the rank-0 Kineto trace from the full GLM-5.2-FP8
HybridEP 16-SM run:

`artifacts/full-131k-hybridep-sms16/trace/b300-1-izksekdp-0001_184807.1787648976440949488.pt.trace.json`

The workload uses 16 B300 GPUs with TP1/PP2/EP8/CP8/ETP1/DP1, sequence
length 131,072, four datums, and 524,288 tokens per step. The traced
forward/backward time is 39.795 seconds.

The ranking uses independent functional paths rather than blindly sorting
inclusive slice durations. In particular, checkpoint wrappers, host
synchronization calls, and their underlying GPU kernels are not added
together. Percentages are relative to traced forward/backward and are not
additive where streams overlap.

Perfetto reported 5,765 `slice_spill_overlapping_complete_event` imports.
It moved those events to overflow tracks and reported no data loss.

## Top five

| Rank | Bottleneck | GPU residency | Share of traced FB | Assessment |
|---:|---|---:|---:|---|
| 1 | PP peer-readiness waits | **9.692 s** | 24.4% | Largest avoidable serialized cost |
| 2 | HybridEP dispatcher | **7.008 s** | 17.6% | Main backend optimization target |
| 3 | Sparse attention/indexer | **5.611 s** | 14.1% | Largest productive compute path |
| 4 | Elementwise/routing/copy fragmentation | **5.346 s** | 13.4% | Small-kernel and materialization tax |
| 5 | NVJET GEMMs | **4.587 s union** | 11.5% | Productive compute; lower priority |

## 1. PP peer-readiness waits

The separate NCCL `SendRecv` track has exactly ten calls totaling 9.692
seconds. This is PP P2P, not EP traffic.

| Call by duration | Duration |
|---:|---:|
| 1 | 5.652 s |
| 2 | 2.848 s |
| 3 | 0.846 s |
| 4 | 0.311 s |

Those four calls contribute 99.6% of the total. The other six are about
0-8 ms each. The calls have zero overlap with other GPU kernels and mirror
`cudaDeviceSynchronize` on the host. Their duration is pipeline-peer
readiness wait, not evidence that a pipeline tensor needs several seconds
to transfer.

Next action: capture matching PP peers, preferably ranks 0 and 8, to identify
which stage reaches each exchange late. Then investigate stage balance,
pipeline scheduling, and overlap.

## 2. HybridEP dispatcher

The complete HybridEP path is serialized on primary GPU stream 7:

| Component | Calls | Time |
|---|---:|---:|
| `device_sync_kernel<8>` polling | 1,680 | **4.725 s** |
| Dispatch payload | 420 | 0.747 s |
| Combine payload | 420 | 0.801 s |
| Permute/unpermute/metadata | 1,960 | 0.735 s |
| **Total** | | **7.008 s** |

Polling is 67.4% of the dispatcher. Its median is 0.098 ms, p90 is 7.224
ms, and maximum is 0.980 s. Six calls over 50 ms contribute 1.961 s. This
long tail indicates rank-arrival skew. Payload kernels themselves are
regular and bounded around 1.8-1.9 ms.

The host-side `HybridEPDispatch` spans total only 0.158 seconds over 280
calls. Device polling, not host dispatch, is the relevant wait.

Next action: target asynchronous progress, earlier rank arrival, or actual
communication/compute overlap. Increasing HybridEP above 16 SMs is not a
promising fix: prior controls showed 20 SMs tied with 16 and 32 SMs slower.

## 3. Sparse attention and indexing

Sparse attention and indexer kernels occupy 5.611 seconds, all on the
primary stream. Major components are:

| Component | Time |
|---|---:|
| Sparse-attention backward | 3.281 s |
| Sparse-attention forward | 1.137 s |
| Indexer forward | 0.869 s |
| Indexer top-k and related work | about 0.3 s |

Backward is the dominant component. The trace also contains 466
`aten::nonzero` calls with 15.046 seconds of inclusive host duration. That
host duration must not be added to the GPU total because much of it is
waiting for underlying GPU work. It does indicate lost host launch-ahead.

Next action: optimize sparse-attention backward first, then remove remaining
CPU-dependent indexing and `nonzero` synchronization where correctness
permits.

## 4. Elementwise, routing, and copy fragmentation

The primary stream spends:

- 3.027 seconds across 119,072 elementwise/routing kernels.
- 2.319 seconds across 19,726 copy/concatenation kernels.

That is 138,798 kernels and 5.346 seconds of GPU residence. Major families
include 0.651 seconds of `CatArrayBatchedCopy`, 0.610 seconds of BF16 direct
copy, 0.460 seconds of elementwise add, 0.327 seconds of additional
concatenation, 0.268 seconds of elementwise multiply, and about 0.48 seconds
of gather/index-copy work.

This is not one slow kernel. It is a distributed tax from materialization,
routing bookkeeping, memory bandwidth, and launch fragmentation.

Next action: fuse adjacent elementwise operations, avoid intermediate
concatenations and copies, and keep routing metadata on device.

## 5. NVJET GEMMs

NVJET GEMMs contribute 5.269 seconds of summed kernel duration and 4.587
seconds after deduplicating simultaneous NVJET execution. Of the summed
duration, 2.173 seconds runs on the primary stream and the remainder runs
across streams 157-160.

This is useful model compute and already uses multiple streams. It likely
overlaps some primary-stream work, so removing one second of summed GEMM
duration would not produce one second of wall-time improvement.

Next action: preserve the multi-stream execution. Investigate grouped-GEMM
shape or tile selection only after PP readiness and HybridEP polling are
improved.

## Additional findings

- GPU union busy time is 34.515 seconds over a 40.510-second kernel window,
  or 85.2%.
- There is an unattributed 2.212-second kernel-free gap near the end of the
  step, between a device-to-host scalar transfer and the next all-reduce.
  The trace lacks a containing Python or system event. If repeatable, this
  could outrank the fifth item and needs Python annotations or a system trace
  for attribution.
- Other NCCL costs are modest: AllReduce 0.851 seconds, ReduceScatter 0.318
  seconds, and AllGather 0.288 seconds.
- `Optimizer.step#FusedAdam.step` is only 11 ms in this trace and is not a
  bottleneck.
- MoE checkpoint means are 49.920 ms for forward/recompute and 129.772 ms
  for backward. Non-MoE backward is stable near 97.197 ms.
- This is a single rank-0 step. Multi-rank traces are required to identify
  the lagging rank behind PP and HybridEP polling tails.

## Recommended order

1. Diagnose and reduce the four PP readiness bubbles.
2. Reduce or overlap HybridEP device polling.
3. Optimize sparse-attention backward and its host synchronization.
4. Fuse routing, copy, and elementwise kernels.
5. Tune NVJET GEMMs only after the serialized costs are addressed.

HybridEP remains the correct backend for this topology: 837.55 tok/s/GPU,
8.86% faster than all-to-all. Its main remaining opportunity is
synchronization, not payload bandwidth.
