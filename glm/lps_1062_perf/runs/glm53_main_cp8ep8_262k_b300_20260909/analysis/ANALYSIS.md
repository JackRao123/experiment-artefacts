# GLM-5.3 rank-0 runtime analysis

Trace: `baseten-training-job-32vj99q-multinode-0_47142.1788990007547624189.pt.trace.json`

## Headline

The profiled step is 23.129 s on the CPU timeline, with 22.535 s of rank-0 GPU-active interval union. HybridEP communication occupies 6.241 s, or 27.0% of the step. Its interval union has no overlap with non-communication GPU kernels, so it is exposed critical-path time in this capture.

Removing HybridEP communication while leaving all current compute and CP work unchanged gives an idealized 16.888 s profiled step. Applying the same 6.241 s saving to the 22.133 s unprofiled control mean gives 15.892 s and about 2,062 tokens/s/GPU. This is a ceiling, not an FSDP prediction: expert weight all-gathers and the EP1 grouped-GEMM shape would consume part of the saving.

## Communication

| Component | Rank-0 interval union | Notes |
|---|---:|---|
| HybridEP dispatch/combine path | 6.241 s | Fully exposed; no overlap with non-communication kernels |
| CP-related collectives, observed | 1.178 s | Fully exposed in this capture |
| CP bulk transport excluding scalar barrier wait | 0.797 s | All-gather + reduce-scatter + final BF16 reduction |

HybridEP breakdown:

| Component | Summed kernel time |
|---|---:|
| Device synchronization kernels | 4.106 s |
| Dispatch kernels | 0.967 s |
| Combine kernels | 0.871 s |
| Expert-group metadata all-reduces | 0.285 s |
| Other HybridEP metadata kernels | about 0.012 s |

The dispatch and combine kernels alone understate the cost. The device-sync kernels are the largest part of the transport critical path.

CP breakdown:

| Component | Calls | Time |
|---|---:|---:|
| BF16 reduce-scatter, 150,994,944 input elements | 78 | 0.633 s |
| BF16 all-gather, mostly 18,874,368 input elements | 199 | 0.163 s |
| Float scalar all-reduce, 2 elements | 1 | 0.380 s |
| BF16 DP-with-CP all-reduce, 193,564,672 elements | 1 | 0.001 s |

The 380 ms scalar all-reduce is a `CONTEXT_PARALLEL_GROUP` barrier after forward, not bandwidth cost. It exposes cross-rank arrival skew and may move to a later collective if removed. Report 1.178 s as observed CP occupancy, but only about 0.797 s as directly transferable CP collective work.

## Bottlenecks

Ranked by rank-0 GPU interval union:

| Rank | Category | Time | Percent of 23.129 s |
|---:|---|---:|---:|
| 1 | HybridEP communication | 6.241 s | 27.0% |
| 2 | DSA attention compute | 5.899 s | 25.5% |
| 3 | GEMM compute | 4.504 s | 19.5% |
| 4 | Other compute | 3.907 s | 16.9% |
| 5 | CP communication | 1.178 s | 5.1% |
| 6 | Local MoE permutation | 0.816 s | 3.5% |

GPU-active union is 22.535 s, leaving about 0.593 s outside categorized GPU activity.

## Per-layer timing

Times are rank-0 GPU interval unions. Kernel launch flows were traced back to their `CheckpointFunction` layer. Within `CheckpointFunctionBackward`, kernels with a nested `*Backward*` operator ancestor are actual backward; the remaining kernels are forward recomputation. This avoids assigning asynchronous GPU work from CPU slice duration alone.

| Layer type | Original forward | Forward recompute | Actual backward | Total backward |
|---|---:|---:|---:|---:|
| Dense, 3 layers | 109.97 ms | 38.49 ms | 81.87 ms | 120.36 ms |
| MoE, 75 layers | 88.48 ms | 65.74 ms | 121.96 ms | 187.71 ms |

Median interval unions are 100.58 ms for dense forward and 67.94 ms for MoE forward. MoE averages are higher than their medians because HybridEP synchronization has severe outliers, including 541 ms in forward and 830 ms in actual backward. The averages are the correct values for reconstructing total step time; medians describe a typical layer.

## Zero-MoE-communication estimate

The estimate removes HybridEP kernels and expert-group metadata all-reduces, then recomputes each layer's GPU interval union. CP communication, local token permutation, and all compute remain.

| Layer type | Phase | Current average | Zero-MoE-comm average | Saving |
|---|---|---:|---:|---:|
| Dense | Forward | 109.97 ms | 109.97 ms | 0 |
| Dense | Recompute | 38.49 ms | 38.49 ms | 0 |
| Dense | Actual backward | 81.87 ms | 81.87 ms | 0 |
| MoE | Forward | 88.48 ms | 58.67 ms | 29.81 ms |
| MoE | Recompute | 65.74 ms | 44.16 ms | 21.59 ms |
| MoE | Actual backward | 121.96 ms | 90.15 ms | 31.81 ms |

For one MoE layer, total backward falls from 187.71 ms to 134.31 ms. Across 75 MoE layers, the savings are:

| Phase | Step saving |
|---|---:|
| Original forward | 2.236 s |
| Forward recomputation | 1.619 s |
| Actual backward | 2.386 s |
| Total | 6.241 s |

## Interpretation for expert FSDP

The trace supports the motivation for replacing EP8 dispatch/combine: HybridEP communication is the largest single bottleneck and is not hidden. The 6.241 s result is the maximum available budget for an EP1/FSDP replacement.

It does not establish that FSDP wins. Expert parameter all-gathers must fit inside that budget, and EP1 changes each rank from 32 larger local expert GEMMs to 256 smaller expert GEMMs. A successful candidate must therefore keep total expert all-gather exposure plus any GEMM regression below about 6.24 s per step.

This is a rank-0-only capture. The 380 ms CP scalar barrier and HybridEP sync outliers prove cross-rank skew exists, but they do not identify which rank is slowest. Multi-rank traces are required to attribute that skew globally.

## Reproduction

The SQL files beside this report reproduce trace bounds, collective groups, kernel critical-path unions, per-layer phase attribution, and the zero-communication estimate. Perfetto's `slice_spill_overlapping_complete_event` warning moved ambiguous CPU complete slices to overflow tracks; it did not drop GPU events.
