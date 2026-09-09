# Perfetto analysis: why MFU is below 50%

Both final Kineto traces were loaded and queried using Perfetto trace_processor.
SQL, CSV rollups, import diagnostics and reproducible scripts accompany this
report. No trainer configuration or production code was changed during analysis.

## Main finding

These runs are not mostly GPU-idle, and attention forward is not grossly
inefficient. Low useful-FLOP MFU comes from full activation recomputation,
substantial non-GEMM work, slower attention backward, and—in the MoE run—many
short GEMMs and gaps. There are avoidable costs, particularly unfused RoPE.
The traces do not measure actual SM-active or tensor-pipe-active counters.

| Trace measurement | Qwen3-0.6B | Qwen3-30B-A3B |
|---|---:|---:|
| Unprofiled useful-FLOP MFU | 29.33% | 25.81% |
| First-to-last GPU kernel span | 1,038.74 ms | 4,487.30 ms |
| Union of GPU kernel intervals | 989.06 ms | 3,898.83 ms |
| Kernel-active fraction of that span | 95.22% | 86.89% |
| No-kernel gaps | 49.68 ms | 588.47 ms |
| Number of GPU kernels | 5,986 | 43,928 |
| Attention kernels | 555.41 ms (55.9%) | 1,838.51 ms (45.4%) |
| GEMM kernels | 114.15 ms (11.5%) | 862.30 ms (21.3%) |
| Elementwise/fusion kernels | 189.72 ms (19.1%) | 690.01 ms (17.1%) |
| Concatenation + copy/conversion | 95.25 ms (9.6%) | 222.06 ms (5.5%) |
| MoE permutation/sorting/top-k kernels | — | 305.46 ms (7.5%) |

Category percentages divide by summed kernel durations (992.71 / 4,046.47 ms),
not by wall time. Parallel streams overlap, so summed durations are not GPU
busy time. The union query accounts for overlap exactly. "No-kernel" is not
an SM-occupancy metric and can include transfer/host work. These are profiled
steps, not the controls: profiling added about 0.9% / 3.9% to forward/backward.

## 1. Full recomputation is intentional extra work

There are exactly 56 attention forward calls and 28 backward calls for 0.6B,
and 96 forward calls and 48 backward calls for 30B. This is two forwards plus
one backward per layer, confirming one checkpoint recompute per layer, not
an accidental additional traversal. The startup used the default full-layer
recompute configuration, which is conservative given the measured 15.45 / 79.47
GiB allocated peaks on a ~267.69-GiB device.

The MFU numerator intentionally excludes recomputation. The existing analytic
execution estimate, which includes full-block recompute and attention score
reconstruction, is approximately 43.4% / 38.4% of nominal BF16 peak using
control timings, versus 29.3% / 25.8% useful MFU. Neither figure is a tensor-pipe
hardware counter. Confusing useful MFU with hardware utilization overstates
the apparent underutilization.

50% useful MFU requires forward/backward times of about 0.604 s / 2.225 s,
versus observed 1.029 s / 4.311 s: approximately 1.70x / 1.94x speedups.
Eliminating GPU gaps alone provides at most ~1.05x / 1.15x on these traces.
There is no basis for expecting 50% simply from making the GPU continuously busy.

## 2. Attention forward is reasonably efficient; backward costs more

The primary kernels are cuDNN SM100 flash SDPA, not an eager S-by-S attention
fallback. Main forward times total 202.54 ms / 680.93 ms across both forwards;
main backward times are 350.51 ms / 1,151.97 ms.

Using the audited dimensions and causal pair count gives about 1.900 / 1.938
PFLOP/s for attention forward: 84.4% / 86.1% of nominal 2.25-PFLOP/s BF16 peak.
Backward is ~1.372 / 1.432 PFLOP/s (61.0% / 63.6%) under the estimate that
flash backward performs the four gradient GEMMs plus one QK score recompute.
These are model-derived rates, not measured FLOP counters; masking, tile
padding and scalar operations make exact execution differ. Nevertheless,
the evidence does not support treating attention forward as catastrophically slow.

## 3. Unfused RoPE is a concrete inefficiency

The GPU executes 69.46 / 165.49 ms of `aten::cat` kernels, with substantial
separate multiply, add, negation and copy work. Not all such operations belong
to RoPE: residual/LoRA additions, activation derivatives and layout conversions
also contribute. The combined 284.97 / 912.06 ms elementwise+cat+copy bucket
must not be presented as entirely removable RoPE time.

However, allocator stacks directly confirm the unfused Megatron RoPE path.
Within each memory-profiled step, `_apply_rotary_pos_emb_bshd:155` accounts for
42.28 / 108.72 GB of cumulative allocated temporaries, `_rotate_half:92` for
21.14 / 54.36 GB, and the final concatenation at line 164 for 14.09 / 36.24 GB.
These are cumulative requested allocation sizes, **not live memory or measured
HBM traffic**. Source uses `torch.cat((-x2, x1))`, separate cosine/sine casts,
multiplies/addition, and a final concatenation with the unrotated tail.
`apply_rope_fusion` defaults false and the trainer does not override it here.

This is a strong candidate for fusion. A fused kernel should remove temporary
materializations and launches, but its speedup and numerical parity need testing.

## 4. The MoE has an additional fragmented execution problem

There are 33,490 GEMM-named kernel launches for 30B versus 1,476 for 0.6B.
MoE GEMMs average ~25.7 microseconds across the aggregate; a grouped-linear
API does not imply one persistent grouped GPU kernel. Attribution connects
10,314 kernels to `_GroupedLinear`, 10,314 to `_GroupedLinearBackward`, and
many recompute kernels to `CheckpointFunction`.

MoE sorting/permutation/top-k consumes ~305 ms. Of the 588 ms with no kernel
running, ~371 ms consists of 10–100-microsecond gaps. Gaps immediately before
`_GroupedLinear`, `_GroupedLinearBackward` and checkpoint-owned kernels total
~455 ms. This is localization by the following kernel's CPU owner, not proof
that that operator caused every preceding gap. Still, it points to the expert
execution/submission path rather than one large external stall.

The 2.278 s of CPU `cudaEventSynchronize` is mostly waiting while GPU work
executes; adding that to GPU durations would double count. It is not 2.278 s
of GPU idleness. A better grouped/persistent expert GEMM path and fewer host
boundaries are worth investigating after recompute and RoPE.

## What is not pathological in these captures

- No NCCL kernels: network/collective communication is not the bottleneck.
- No `cudaMalloc` or `cudaFree` calls in either runtime capture.
- Memory histories show 5,726 allocations and matching frees for 0.6B, and
  13,082 allocations and matching frees for 30B; no segment growth or OOM events
  in the captured windows. This is no evidence of a leak in these steps, not
  a long-run leak test.
- Control forward/backward timing CV is ~0.4% for both models, arguing against
  intermittent large stalls in the measured sequence of controls.
- Both traces use modern SM100/SM103 cuDNN/GEMM kernels. There is no sign of
  an all-FP32 or CPU execution fallback.

## Next experiments, in order

1. Test reduced/selective recomputation with the same 10-control protocol.
   We have significant observed memory headroom, but the new peak must be
   measured; eliminating recompute is not guaranteed to fit or be optimal.
2. Enable/validate fused RoPE for Qwen, with numerical parity tests and the
   same performance protocol. Do not assume every elementwise kernel disappears.
3. Investigate the MoE expert GEMM implementation and host synchronization;
   distinguish actual expert compute from per-expert launch/setup overhead.
4. For a definitive saturation answer, collect Nsight Systems GPU metrics
   (`SM Active`, `Tensor Active`, DRAM bandwidth) or targeted Nsight Compute
   throughput/roofline metrics on representative attention and expert kernels.

Kineto's `est. achieved occupancy %` fields are estimates and are zero for
these cuDNN attention kernels. Zero here demonstrably cannot mean the GPU
did no work. It must not be reported as measured SM/tensor-core saturation.
NVIDIA documents the relevant hardware metrics in the
[Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpu-metrics)
and [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

## Reproduce

From the parent run directory:

```bash
python3 perfetto_analysis.py
python3 perfetto_analysis/supplement.py
```

`results.json` contains Perfetto rollups; `supplement.json` contains gap
attribution, allocator-stack evidence and model-derived attention throughput.
Perfetto reports overlapping CPU complete slices and places them on overflow
tracks without loss. GPU kernel counts/totals agree with the raw Kineto JSON;
CPU inclusive durations must not be summed as a wall-time breakdown.
