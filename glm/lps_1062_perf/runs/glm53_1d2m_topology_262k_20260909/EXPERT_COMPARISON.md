# EP8 versus EP1: expert-path and straggler analysis

Both runs have CP8 and 8 GPUs. EP8 uses HybridEP; EP1 uses the existing alltoall local path after HybridEP EP1 failed. One rank-0 runtime capture per topology; all-rank CUDA-event block timings cover all five controls and the profiled step. No new experiment or trainer change was made for this analysis.

## Routed forward region in the runtime captures

The GPU interval is inferred from kernel ownership after RouterGatingLinearFunction through block end, completed with kernels from all streams. It excludes the shared expert, which runs before the router, and includes routing after the gating projection, dispatch/permutation, routed experts, combine, and final postprocessing. It is not an exact whole-MLP module timer. Filling GPU intervals is necessary because no-grad TE grouped GEMMs lack CPU-op ownership links.

| Mean per MoE block, two blocks | EP8 | EP1 |
|---|---:|---:|
| Routed forward-region GPU elapsed | 37.60 ms | 42.80 ms |
| Routed forward GEMM kernel-duration sum | 18.07 ms | 22.07 ms |
| Routed forward GEMM kernel count | 62 | 502 |
| Routed recompute-region GPU elapsed | 35.59 ms | 33.15 ms |

The forward expert path did not become faster in this captured step. GEMM work is split into many more kernels at EP1, offsetting communication savings. Recompute's routed region is slightly faster, so there is no uniform slowdown of every phase. GEMM sums can overlap across streams and are not elapsed time. These two-block, one-step measurements are not five-control averages.

## Large mean slowdown: rank stragglers, not a stable ~160 ms expert penalty

Control FB times: EP8 1.380, 1.269, 1.281, 1.271, 1.266 seconds; EP1 1.766, 1.300, 1.655, 1.266, 1.267 seconds. The non-outlying controls are comparable.

In EP1 control 2 (zero-based; worker step 5), block 1:
- Rank 6: recompute 466.28 ms, backward excluding recompute 93.57 ms.
- Rank 0: recompute 56.93 ms, backward excluding recompute 502.92 ms.
- Rank 7: recompute 57.04 ms, backward excluding recompute 502.76 ms.

The ~409 ms extra time appears in rank 6's recompute and in peers' backward. This supports a late-rank delay propagating through synchronization, rather than a uniform increase in expert compute or communication bandwidth cost.

In the EP1 runtime-profiled step (worker step 9), block 1:
- Rank 4: recompute 181.68 ms, backward excluding recompute 93.21 ms.
- Rank 0: recompute 57.28 ms, backward excluding recompute 217.58 ms.
- Rank 0's corresponding CP ReduceScatter: 120.86 ms, versus 7.34 ms in EP8.

Again the ~124 ms late recompute on one rank matches the extra time peers spend in backward. A rank-0 NCCL duration includes waiting for peers; it must not be interpreted as pure transfer time.

## What remains unknown

We cannot identify the initiating straggler's cause from these artifacts: the runtime trace is rank 0 only, whereas the profiled straggler was rank 4; the 466 ms rank-6 delay occurred in an unprofiled control. CPU scheduling/GC, allocation, and compilation are possible causes, not established diagnoses. Need multi-rank runtime capture of a slow step to discriminate them. Do not label this a CP bandwidth regression or an inherent EP1 throughput penalty.

Reproduce numerical extraction with expert_compare.py; results in expert_comparison.json. Underlying Perfetto CSVs and all-rank timing JSONL files are unchanged.
