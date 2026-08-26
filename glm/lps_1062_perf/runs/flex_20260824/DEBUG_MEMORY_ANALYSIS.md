# Flex CUDA Memory Snapshot Analysis

Analyzed all 24 snapshots: 8 ranks each for `alltoall`, `deepep-sms16`, and `deepep-sms20`. Values are binary GiB. All snapshots use identical allocator settings (`expandable_segments:True`), reconcile exactly against their final segment state, and contain no OOM event.

## Summary

| Dispatcher | Active start, mean | Active peak, mean (rank range) | Active end, mean | Reserved peak/end, mean (rank range) | Inactive end, mean | Max dispatcher-attributed live, mean (rank range) |
|---|---:|---:|---:|---:|---:|---:|
| `alltoall` | 7.215 | 25.615 (25.613-25.629) | 7.470 | 27.953 (27.613-28.648) | 20.483 | 3.520 (2.245-5.338) |
| `deepep-sms16` | 7.199 | 25.598 (25.597-25.605) | 7.453 | 28.998 (28.510-30.014) | 21.545 | 1.885 (0.519-2.975) |
| `deepep-sms20` | 7.199 | 25.598 (25.597-25.605) | 7.453 | 28.993 (28.510-30.014) | 21.541 | 1.885 (0.519-2.975) |

## Findings

- DeepEP reduces the maximum dispatcher-attributed live set by 1.635 GiB/rank on average (46%) versus all-to-all. It does **not** reduce the model-wide active peak materially: DeepEP is only 17 MiB lower, of which 16 MiB was already present at trace start. Peak growth from trace start differs by only 1 MiB (18.400 GiB all-to-all versus 18.399 GiB DeepEP), because dispatcher and model-wide peaks occur at different times.
- Dispatcher memory follows routed-token imbalance. All-to-all ranges from 2.245 GiB on rank 7 to 5.338 GiB on rank 1. DeepEP ranges from 0.519 GiB on rank 7 to 2.975 GiB on rank 1. Despite this, total active-peak rank spread is only 16 MiB for all-to-all and 8 MiB for DeepEP.
- DeepEP's visible payload allocations are bounded and transient. Per-source maxima include `deep_ep/buffer.py:407` at 145-771 MiB/rank, `deep_ep/buffer.py:396` at 144-765 MiB/rank, combine output at 96 MiB/rank, and permutation/restoration tensors whose sizes track rank token load. No DeepEP-attributed allocation remains active at snapshot end on any rank.
- `sms16` and `sms20` have effectively identical live memory. Active start/peak/end are identical; per-rank dispatcher maxima differ by less than 0.7 MiB. `sms20` reserves only 5 MiB/rank less on average, caused by two ranks mapping one fewer 20 MiB allocator extent.
- The only material SMS-specific trace difference is transient `torch.topk` scratch under `_DeepepManager.setup_metadata`: `sms16` records 241 allocations/rank and 2.20 GiB/rank of throughput versus 25 allocations/rank and 47.5 MiB/rank for `sms20`. On rank 0, `sms16` adds 217 x 8 MiB and 4 x 126.5 MiB allocations; median lifetime is 14 us and all are freed. Its source-live maximum is 143 MiB versus 8.5 MiB for `sms20`, without changing the run's active peak. This is scratch churn, not a retained communication buffer or leak.
- There is no Flex-related active-memory leak. Rank 0 returns to exactly 7.4834 GiB after each of five all-to-all optimizer steps and 7.4596 GiB after each of five steps for both DeepEP settings. Ranks 1-7 finish 258.7 MiB above trace start in every configuration, so that retained initialization state is dispatcher-independent. The sole final dispatcher-attributed allocation is 8.125 MiB on all-to-all rank 0; DeepEP retains none.
- Reserved memory reaches its high-water mark and stays mapped: no snapshot records a segment unmap. DeepEP reserves about 1.04 GiB/rank more than all-to-all despite the same active peak. At the active peak, inactive reserve averages 3.321 GiB for DeepEP versus 2.059 GiB for all-to-all; after the step, 73-75% of reserved memory is inactive cache. With expandable segments and no OOM, this is allocator caching/high-water behavior, not evidence of a live leak, though DeepEP's allocation shape leaves roughly 1.26 GiB/rank more cached headroom.

## Buffer Visibility

The snapshots expose tensors allocated by DeepEP dispatch/combine, but show no long-lived allocation attributed to the process-global `Buffer(...)` construction in `fused_a2a.py`. If that DeepEP buffer uses custom CUDA/NVLink/RDMA allocation outside PyTorch's caching allocator, it is not represented in these pickles. Therefore the snapshots rule out a retained **PyTorch-managed** Flex buffer, but cannot quantify or exclude externally managed DeepEP communication memory.
