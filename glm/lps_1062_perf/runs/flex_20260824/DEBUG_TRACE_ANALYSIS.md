# Flex dispatcher trace analysis

Scope: one rank-local Kineto trace from each of `debug-ep8-alltoall`, `debug-ep8-deepep-sms16`, and `debug-ep8-deepep-sms20`. Results come from identical `trace_processor` SQL over kernel slices and `ProfilerStep#0`. This analysis intentionally excludes model/attention/optimizer tuning and treats unchanged compute as a control.

## Summary

| Run | Step wall | Kernel span | Union GPU-busy in span | Dispatcher kernels | Non-EP NCCL |
|---|---:|---:|---:|---:|---:|
| EP8 all-to-all | 1229.6 ms | 1117.5 ms | 780.3 ms (69.8%) | 36.1 ms NCCL SendRecv | 60.9 ms |
| EP8 DeepEP, 16 SM | **944.6 ms** | **914.7 ms** | 787.7 ms (86.1%) | **101.7 ms DeepEP** | **1.7 ms** |
| EP8 DeepEP, 20 SM | 1174.2 ms | 1144.4 ms | 942.4 ms (82.3%) | 107.2 ms DeepEP | 151.2 ms |

The captured step is 23.2% shorter with DeepEP 16 SM than all-to-all and only 4.5% shorter with 20 SM. The 20-SM trace is 229.6 ms slower than 16 SM and contains 149.5 ms more non-EP NCCL time. The large `AllReduce` durations are rank-local wait/skew signals, not evidence that the dispatcher itself transfers data that slowly.

## Dispatcher and Flex findings

- Compute is controlled: GEMM kernel time is 545.3 / 544.0 / 543.8 ms across all-to-all / 16 SM / 20 SM. Sparse attention is 28.77 / 28.78 / 28.76 ms. The performance spread is therefore communication and synchronization, not changed model compute.
- No tested dispatcher overlaps with primary-stream compute. Pairwise interval intersection is 0.0 ms for NCCL SendRecv versus the main compute track and 0.0 ms for DeepEP kernels versus the main compute track. DeepEP uses a separate stream, but Flex dependencies serialize every observed dispatch/combine into compute gaps.
- Flex forces substantial host-visible synchronization through `aten::nonzero` and `aten::_local_scalar_dense`. Their `cudaStreamSynchronize` time is 375.6 ms for all-to-all, 439.1 ms for DeepEP 16 SM, and 469.7 ms for DeepEP 20 SM. These waits are inclusive of GPU work, so they are not additive to wall time, but they explain why the nominally asynchronous dispatcher does not overlap.
- All-to-all spends 36.1 ms in nine `ncclDevKernel_SendRecv` calls. DeepEP replaces these with longer-lived communication/polling kernels: 101.7 ms at 16 SM and 107.2 ms at 20 SM. DeepEP's step benefit in this capture is not lower rank-local dispatcher-kernel residency; it is better scheduling/less exposed rank skew elsewhere in the step.
- DeepEP is dominated by notification/polling rather than payload movement:

| DeepEP class | 16 SM | 20 SM | 20 vs 16 |
|---|---:|---:|---:|
| Dispatch/combine payload | 24.4 ms | **19.8 ms** | -18.9% |
| Notify/cached-notify | **77.3 ms** | 87.4 ms | +13.2% |
| Total | **101.7 ms** | 107.2 ms | +5.5% |

- The common first `cached_notify_combine` spin is pathological in both traces: 69.8 ms at 16 SM and 64.4 ms at 20 SM before a roughly 4-5 ms combine payload. This is waiting/dependency exposure, not useful transfer work.
- Raising the budget to 20 SM does accelerate payload kernels: dispatch averages 2.55 ms versus 3.17 ms, and combine averages 4.01 ms versus 4.92 ms. It loses that gain in notify waits. In particular, two `notify_dispatch` calls total 13.1 ms at 20 SM versus 1.7 ms at 16 SM. More communication SMs therefore increase pressure without improving the serialized Flex critical path.
- The 20-SM trace also shows severe downstream rank skew: BF16 all-reduce is 88.6 ms versus 0.12 ms at 16 SM, and a U64 all-reduce is 48.9 ms versus 1.49 ms. The all-to-all trace has a similar 56.6 ms BF16 all-reduce wait. With only one rank traced, this cannot be assigned conclusively to network throughput; it is consistent with ranks arriving unevenly after dispatcher/Flex work.

## Recommendation

Use **DeepEP 16 SM** for this Flex configuration. It has the best captured wall time and GPU occupancy, lower total DeepEP time, dramatically lower `notify_dispatch` time, and avoids spending four extra SMs for payload gains that are erased by synchronization. Do not select 20 SM from these data.

The next useful optimization is not a higher SM count. It is removing or deferring the Flex-side `nonzero` / scalar-read synchronizations and making dispatch/combine genuinely overlap the primary compute stream. Multi-rank traces would then distinguish true DeepEP transfer imbalance from collective arrival skew.
