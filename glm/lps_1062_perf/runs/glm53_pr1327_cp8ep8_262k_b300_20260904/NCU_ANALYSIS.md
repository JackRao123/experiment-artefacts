# GLM-5.3 PR 1327: targeted Nsight Compute analysis

## End-to-end correction

The replacement fixed-input controls show that PR 1327 is faster in steady state. Across ten controls, forward/backward averages 24.9 s (1,315 tok/s/GPU), with eight runs between 24.8 and 25.1 s. The older baseline was about 26.7 s. The 28.9 s Nsight Systems capture is an allocator-growth outlier, not representative steady-state throughput.

The trace outlier is explained by two expandable-segment growth episodes during recompute MoE expert allocations:

- GPU 1 grows 1.816 GiB in recompute layer 78. `cuMemSetAccess` takes 887 ms and peers wait up to 1.258 s in HybridEP `device_sync`.
- GPU 6 grows 2.051 GiB in recompute layer 76. `cuMemSetAccess` takes 1.202 s and peers wait up to 1.509 s.
- Busy-laggard HybridEP wait is unchanged from the old trace (26.490 vs 26.456 all-GPU seconds). Host-idle-laggard wait rises from 2.754 to 26.739 all-GPU seconds.

Thus the increased HybridEP time in this capture is allocator-induced rank skew. The increase in MoE-attributed GPU time is mostly the same spin-wait kernels being charged to nested MoE/HybridEP NVTX ranges; expert GEMM and dispatch/combine work are nearly unchanged.

## DSA backward

Report: `glm53-pr1327-dsa-bwd-sorted-random-topk-detailed.ncu-rep`

Probe: `glm53_dsa_bwd_probe.py`

The probe uses the traced production geometry: 32,768 local queries, 262,144 gathered keys, 64 heads, Q/K width 576, value width 512, and sorted scattered top-k 2,048 indices. Absolute NCU replay duration should not be compared to the unprofiled trace, but the launch geometry matches: grid 32,768, block 640, 96 registers/thread, and 231,424 bytes dynamic shared memory.

Key results:

| Metric | Value |
|---|---:|
| Compute throughput | 36.52% |
| DRAM throughput | 34.79% |
| L1/TEX throughput | 47.37% |
| L2 throughput | 27.76% |
| L2 hit rate | 28.81% |
| Issue slots busy | 15.49% |
| Achieved occupancy | 26.19% |
| Active warps per SM | 16.76 of 64 |
| Local-memory spilling requests | 174,882,816 |

Warp-stall samples are dominated by long scoreboard (62.6%), then barriers (21.2%) and wait (6.0%). The kernel is latency-bound on irregular key/value dependencies, not saturated on DRAM or tensor-core throughput. Its single 640-thread CTA consumes enough registers and shared memory to limit residency to one block per SM, leaving too few warps to hide those dependencies. The heavy local spilling is additional evidence of register pressure.

Implications:

- The direct kernel lever is a different cuDNN/CUTLASS schedule with lower register/shared-memory footprint or better pipelining/locality. This is vendor-kernel work, not a Python launch-overhead fix.
- Lowering DSA top-k would reduce this 3.72 s/GPU step bucket approximately proportionally, but changes model quality and must be evaluated as a model tradeoff.
- The indices are already sorted before the production kernel. Sorting materially improves locality versus unsorted scattered indices, so there is no obvious application-side sorting win left.

## FP32 LM head

Report: `glm53-pr1327-lm-head-fp32-detailed.ncu-rep`

Probe: `glm53_lm_head_probe.py`

The probe profiles the production forward chunk geometry: `[4096, 6144] @ [151552, 6144].T` in FP32. The traced step spends 1.85 s/GPU across forward and backward LM-head kernels.

Key results:

| Metric | Value |
|---|---:|
| Compute throughput | 88.70% |
| FP32 FMA pipe active | 88.48% |
| Tensor-core activity | 0% |
| DRAM throughput | 7.84% |
| Achieved occupancy | 12.50% |
| Registers/thread | 255 |

This kernel is compute-bound on the FP32 SIMT FMA pipe. More occupancy or memory tuning will not fix the main issue. The actionable change is to use BF16 tensor-core inputs with FP32 accumulation/output, subject to the existing numerical parity check. That moves work to otherwise-unused tensor hardware and is the highest-confidence single-kernel speedup remaining.

## Priority

1. Treat 24.9 s as the representative PR 1327 steady-state result; do not use the allocator-outlier trace as the throughput headline.
2. Move the frozen LM head from FP32 SIMT GEMM to a BF16 tensor-core GEMM with FP32 accumulation/output.
3. Evaluate a lower-resource or better-pipelined DSA backward kernel; alternatively test lower top-k only as an explicit quality/performance tradeoff.
4. Pre-grow or reserve the CUDA allocator before measured windows so rank-local `cuMemSetAccess` does not become a HybridEP critical-path stall.
5. Continue measuring expert routing imbalance on customer-shaped data. The baseline HybridEP wait remains real, but it is not the source of the PR 1327 trace delta.
