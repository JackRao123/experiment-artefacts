# Uneven expert GEMM replay — findings

The main win in the traced CP8EP1 gate/up calls was reducing CPU-side
preparation before the first kernel, not eliminating large empty gaps between
GPU kernels. TE kept roughly 92–95% of SMs active once execution started.

## Unprofiled results

Milliseconds, averaged over all eight captured rank shapes. Each entry sums
two separately timed projections (gate/up and down), not a full MLP. Twenty
measured repetitions per implementation/projection follow five warmups.

| Layout | MoE | TE forward | Grouped forward | TE input-gradient | Grouped input-gradient |
|---|---:|---:|---:|---:|---:|
| CP8EP1 | 1 | 12.348 | 8.969 | 10.184 | 8.877 |
| CP8EP1 | 2 | 13.164 | 8.792 | 11.062 | 8.678 |
| CP8EP8 | 1 | 8.783 | 8.094 | 7.987 | 7.973 |
| CP8EP8 | 2 | 8.474 | 7.588 | 7.670 | 7.519 |

Grouped-MM reduced EP1 forward time by **27–33%** and input-gradient time by
**13–22%**. These are standalone GEMM results, not measured full-model speedups.
EP8 averages hide routing imbalance: in MoE 1, its slowest rank's forward
projection sum was 16.813 ms (TE) / 16.626 ms (grouped), versus EP1 maxima
of 12.755 / 9.315 ms. All per-rank values and samples are retained.

## What the timeline shows

Nsight Systems, CP8EP1 rank 0, gate/up forward, five repetitions:

| Measurement | MoE 1 TE | MoE 1 grouped | MoE 2 TE | MoE 2 grouped |
|---|---:|---:|---:|---:|
| CPU entry → first GPU kernel (ms) | 1.697 | 0.073 | 1.776 | 0.073 |
| First → last GPU kernel span (ms) | 5.791 | 5.723 | 6.001 | 5.775 |
| No-kernel time inside that span (ms) | 0.005 | 0.023 | 0.005 | 0.026 |
| Kernels | 273 | 2 | 260 | 2 |
| CUDA streams used | 4 | 1 | 4 | 1 |
| SMs Active, kernel span (%) | 94.9 | 97.9 | 92.1 | 98.0 |
| Tensor Active, kernel span (%) | 70.8 | 93.8 | 68.1 | 95.0 |

CPU entry → last GPU kernel was 7.488 → 5.796 ms for MoE 1 and
7.777 → 5.848 ms for MoE 2. Most of this improvement precedes the first kernel.
The CPU entry → first CUDA launch API delay was 1.675 ms versus 0.058 ms
in MoE 1; GPU launch latency after the API returned was about 2 microseconds.

TE's MoE 1 forward comprises 192 GEMM kernels plus 81 split-K reductions.
Its four streams overlap substantially: summing kernel durations would give
10.845 ms, but their actual execution envelope is only 5.791 ms. All 2,345
kernels in that trace and all kernels in the MoE 2 trace were matched to
their original NVTX phases through CUDA correlation IDs, without double-counting.

The host profile found about 150,440 calls over 20 TE forwards versus 440 for
grouped-MM. Work is in the native `te_general_grouped_gemm` bridge and
per-expert tensor/saved-state handling. cProfile locates code, but its
instrumented times are not used as benchmark numbers.

## What ncu adds

The biggest TE GEMM kernels already reached about 91% tensor-pipe activity.
The many smaller kernels do not imply that the entire GPU is spatially idle:
the concurrent timeline above is the relevant check for that claim.

For EP1 gate/up, both implementations perform 6.597 trillion useful FLOPs.
NCU counted 6.858 / 6.906 trillion BF16 tensor operations for TE (MoE 1 / 2),
but 8.620 / 8.452 trillion for grouped-MM: roughly **28–31% more tensor work
than the useful arithmetic** in the grouped implementation. Higher tensor
activity therefore does not translate directly into more useful throughput.

This is not just an inferred padding explanation: the grouped tensor-operation
counts **exactly equal rounding every expert's row count up to a multiple of
256** and evaluating `2 × padded_rows × K × N`, for all eight grouped range
profiles. EP1's 131,072 useful rows become 171,264 effective rows in MoE 1 and
167,936 in MoE 2. This describes executed tile work, not necessarily a physically
allocated padded input tensor. `build_summary.py` checks the equality.

Grouped-MM also had higher profiled DRAM reads: for MoE 1 gate/up, 13.125 GB
for TE versus 24.945 GB for grouped-MM in the range profiles. Cache state and
profiler scheduling affect these counters. Nsight Systems independently
showed higher DRAM-read throughput during the grouped kernel window
(about 52% versus 33%), confirming the direction rather than an exact
full-trainer traffic ratio.

## Measurement limits

- Frozen BF16 base experts, synthetic activations and identical synthetic
  weights for the two implementations. Shapes come from the original-forward
  routing captures. No checkpoint or Megatron trainer was loaded.
- Gate/up: `K=6144, N=4096`; down: `K=2048, N=6144`. Each expert's `M` comes
  directly from its recorded post-dispatch token count.
- No communication, routing/permutation, activation, LoRA adapter GEMMs,
  expert-weight gradients, or full-model cache/overlap behavior is included.
- Packing the alternative's weights happens once, outside timing. The
  production FSDP experiment instead uses a zero-copy contiguous weight view.
- Sampled rows from every nonempty expert were checked for forward and input
  gradients; maximum relative RMS error was 0.00014068 (0.0141%). This is not
  a full training-equivalence test.
- **NCU range durations/utilization are not benchmark results.** Its overhead
  inflated some TE ranges to 90–108 ms while unprofiled projections took
  roughly 8 ms. They are deliberately excluded from speed/idle-time claims.
  Kernel replay serializes launches; whole-range replay preserves related
  launches but still perturbs execution. See the
  [NVIDIA replay guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#replay).
- The initial application-replay sweep hung on TE multi-stream launches.
  Owned stalled processes were stopped; the final range runner uses process
  groups and bounded timeouts. All 16 selected forward range reports are
  available and their metric fields were validated after resuming.

## Files

- `expert_gemm.py`: small implementation; imports only PyTorch and TE GroupedLinear.
- `run_probe.py`: measured-shape loader, timing, numerical checks, and NVTX ranges.
- `run_suite.py`: unprofiled replay of all eight rank shapes.
- `profile_suite.py`: NCU range collection; EP1 rank 0 both MoEs, EP8 busiest
  rank per MoE (rank 0 for MoE 1; rank 4 for MoE 2), both forward projections.
- `analyze_timeline.py`: reusable CUDA/NVTX attribution and interval-union analysis.
- `build_summary.py`, `summary.json`: machine-readable timing/counter summary.
- `results/`: raw NCU reports/CSVs, two Nsight Systems gate/up traces, timings,
  and host profiles. **97 copied files passed SHA-256 verification.**

Environment: 8 HGX B300s (reported as L20D; 148 SMs each), driver 580.105.08,
PyTorch 2.11.0+cu130, TE 2.16.0, Nsight Compute 2025.3.1, Nsight Systems 2025.3.2.
`CUDA_DEVICE_MAX_CONNECTIONS=32`, one CPU intra-op thread per replay process.
The profiler pod remains available; no GPU workload is left running.
