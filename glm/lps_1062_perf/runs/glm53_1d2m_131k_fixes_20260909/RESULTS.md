# GLM-5.3 1d2m: 131072-token results with fixes

Tracking PR: [trainers #1355](https://github.com/basetenlabs/trainers/pull/1355), stacked on #1157. Dependencies: [Megatron-Core #76](https://github.com/basetenlabs/Megatron-LM/pull/76) and [Bridge #84](https://github.com/basetenlabs/Megatron-Bridge/pull/84).

## Configuration and protocol

Real GLM-5.3 blocks 0, 3, 6 remapped to dense/indexer, MoE/shared, MoE/indexer. LoRA rank/alpha 32, TP1/PP1/ETP1, full one-block recompute, native-FP8 expert storage. All three use BT_FREEZE_GC_AFTER_WARMUP=1. EP1 automatically selects local dispatch; EP8 uses HybridEP. Tested source c9a723bf431621ca05580726f4acd01ec618326a, Bridge 2076a9e81ae36e7890d017f4af8d7c707c2d633b, Core 3b893e39e (full pins available from dependency commits).
Three complete warmup FB+optimizer iterations, five unprofiled controls, one memory step, one all-rank runtime step. Headline timings retain all five controls and exclude optimizer time. CUDA-event instrumentation remains enabled in controls, with no synchronization between sublayers. One-time GC collection and its rank barrier are included in the warmup optimizer time.

## Five-control measurements

| Topology | GPUs | FB mean ± SD, s | Tokens/s total | Tokens/s/GPU | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|---:|
| cp8ep8 | 8 | 0.6448 ± 0.0653 | 203,278 | 25,410 | 24.512 | 26.793 |
| cp8ep1 | 8 | 0.6563 ± 0.0143 | 199,710 | 24,964 | 53.609 | 55.020 |
| cp1ep1 | 1 | 3.5149 ± 0.0174 | 37,291 | 37,291 | 136.151 | 143.400 |

Peaks are the distributed maximum of PyTorch allocator peaks across controls, not all physical HBM usage. TPS/GPU is efficiency, not single-request latency. These are three-block proxy TPS values, not full-model throughput.

## Separate attention / MLP measurements

GPU elapsed milliseconds, rank 0, mean ± sample SD over five controls. Attention includes its normalization/residual path and any indexer computation. MLP includes routed/shared experts, dispatch/combine, normalization and residual processing. Backward excludes forward recomputation; tensor-gradient boundary hooks partition the backward path. Full block and all-rank raw timings remain available.

| Topology | Block | Fwd attn | Fwd MLP | Recompute attn | Recompute MLP | Bwd attn | Bwd MLP |
|---|---|---:|---:|---:|---:|---:|---:|
| cp8ep8 | Dense + indexer | 39.72 ± 1.18 | 5.79 ± 0.01 | 14.66 ± 0.15 | 5.82 ± 0.01 | 35.01 ± 0.21 | 7.42 ± 0.06 |
| cp8ep8 | MoE + shared indices | 14.22 ± 0.05 | 22.15 ± 0.07 | 12.45 ± 0.12 | 26.37 ± 9.20 | 33.92 ± 0.44 | 22.52 ± 0.07 |
| cp8ep8 | MoE + indexer | 32.53 ± 1.89 | 27.20 ± 19.02 | 16.64 ± 0.44 | 18.38 ± 0.44 | 35.69 ± 0.20 | 17.28 ± 0.12 |
| cp8ep1 | Dense + indexer | 41.46 ± 2.95 | 5.77 ± 0.01 | 14.86 ± 0.39 | 5.78 ± 0.00 | 35.21 ± 0.46 | 7.35 ± 0.00 |
| cp8ep1 | MoE + shared indices | 14.18 ± 0.05 | 33.34 ± 2.10 | 12.32 ± 0.06 | 25.66 ± 0.07 | 35.01 ± 0.68 | 18.97 ± 0.08 |
| cp8ep1 | MoE + indexer | 31.71 ± 1.19 | 35.42 ± 2.58 | 16.97 ± 0.78 | 27.68 ± 0.79 | 39.65 ± 4.05 | 19.85 ± 0.24 |
| cp1ep1 | Dense + indexer | 240.61 ± 0.19 | 48.43 ± 1.49 | 109.92 ± 0.32 | 48.49 ± 1.65 | 294.00 ± 4.60 | 66.88 ± 2.09 |
| cp1ep1 | MoE + shared indices | 108.93 ± 3.08 | 99.94 ± 0.84 | 100.16 ± 0.36 | 99.99 ± 1.23 | 293.12 ± 2.26 | 97.87 ± 2.63 |
| cp1ep1 | MoE + indexer | 254.86 ± 1.21 | 99.44 ± 2.45 | 114.65 ± 0.87 | 100.75 ± 1.97 | 297.79 ± 7.37 | 100.23 ± 2.49 |

## Weighted full-model block estimate

Weight by 3 dense/indexer + 57 MoE/shared + 18 MoE/indexer blocks. This excludes embedding/head/loss, optimizer, and inter-block overhead. It is not a claim that the full model fits EP1; depth-dependent routing/cache lifetimes and memory pressure limit extrapolation.

| Topology | Forward s | Recompute s | Actual backward s | Total block FB s |
|---|---:|---:|---:|---:|
| cp8ep8 | 3.286 | 2.905 | 4.299 | 10.490 |
| cp8ep1 | 4.060 | 3.031 | 4.276 | 11.367 |
| cp1ep1 | 19.151 | 15.761 | 30.534 | 65.446 |

## Diagnosis and fixes

1. **CP1 indexer wiring:** plain-causal score chunks now call the existing cuDNN scorer with global query offsets. The old per-head FP32 bmm fallback was unnecessary for this case. Bounds/key-count probes passed; selected-key overlap was 1.0 at 2k/4k and 0.99999994 at 8k.
2. **Singleton HybridEP:** no network transfer does not mean no dispatch kernel. Its cooperative dispatch failed with 256 local experts. The singleton case now takes the existing local alltoall permutation path, preserving non-overlapped shared-expert scheduling.
3. **Host GC stragglers:** the 131k baseline recorded 598–636 ms generation-2 pauses in slow controls, and the all-rank trace showed one rank collecting for 591 ms while peers waited. No CUDA allocation/free driver activity accompanied that captured stall. The experimental GC policy freezes warmed long-lived objects but leaves automatic GC enabled for new objects, then unfreezes before resource teardown.
GC policy is default-off and remains experimental pending longer process-lifetime/full-model validation. No tests were added or modified. Three-warmup protocol and cached diagnostic counters avoid accepting cold or observer-contaminated measurements. Earlier exploratory attempts are retained but are not in this table.

## Additional stability checks

- cp8ep8: 20 additional controls, FB 0.6200 ± 0.0251 s, range 0.6054–0.7178 s; allocated peaks 24.512–24.512 GiB; reserved peaks 26.793–26.793 GiB.
- cp8ep1: 20 additional controls, FB 0.6534 ± 0.0176 s, range 0.6431–0.7272 s; allocated peaks 53.609–53.609 GiB; reserved peaks 55.020–55.020 GiB.
- cp1ep1: 20 additional controls, FB 3.5133 ± 0.0116 s, range 3.4892–3.5440 s; allocated peaks 136.151–136.151 GiB; reserved peaks 143.400–143.400 GiB.

## Traces and raw data

Runtime captures include profiler start/stop and may perturb the step, especially with all ranks recording. Their CPU API duration sums are not additive wall time. Use control CUDA-event timings for the attention/MLP comparison. Memory and runtime captures are separate steps. Trace and memory bytes were copied locally, hashed, and parsed.

### cp8ep8

[Detailed statistics](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp8ep8/result/summary.json) · [Raw benchmark](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp8ep8/result/benchmark.json)

- [Kineto: rank0.1789003083652794655.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp8ep8/result/runtime/rank0.1789003083652794655.pt.trace.json)
- [Kineto: rank1.1789003083649212868.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp8ep8/result/runtime/rank1.1789003083649212868.pt.trace.json)
- [Kineto: rank2.1789003083648122475.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp8ep8/result/runtime/rank2.1789003083648122475.pt.trace.json)
- [Kineto: rank3.1789003083649159542.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp8ep8/result/runtime/rank3.1789003083649159542.pt.trace.json)
- [Kineto: rank4.1789003083651674281.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp8ep8/result/runtime/rank4.1789003083651674281.pt.trace.json)
- [Kineto: rank5.1789003083650905041.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp8ep8/result/runtime/rank5.1789003083650905041.pt.trace.json)
- [Kineto: rank6.1789003083649332090.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp8ep8/result/runtime/rank6.1789003083649332090.pt.trace.json)
- [Kineto: rank7.1789003083650854564.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp8ep8/result/runtime/rank7.1789003083650854564.pt.trace.json)
- [Memory: memory.rank0.pickle](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp8ep8/result/memory/memory.rank0.pickle)

### cp8ep1

[Detailed statistics](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp8ep1/result/summary.json) · [Raw benchmark](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp8ep1/result/benchmark.json)

- [Kineto: rank0.1789002725283346351.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp8ep1/result/runtime/rank0.1789002725283346351.pt.trace.json)
- [Kineto: rank1.1789002725285834671.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp8ep1/result/runtime/rank1.1789002725285834671.pt.trace.json)
- [Kineto: rank2.1789002725285103684.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp8ep1/result/runtime/rank2.1789002725285103684.pt.trace.json)
- [Kineto: rank3.1789002725284746067.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp8ep1/result/runtime/rank3.1789002725284746067.pt.trace.json)
- [Kineto: rank4.1789002725297907469.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp8ep1/result/runtime/rank4.1789002725297907469.pt.trace.json)
- [Kineto: rank5.1789002725281472568.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp8ep1/result/runtime/rank5.1789002725281472568.pt.trace.json)
- [Kineto: rank6.1789002725280811944.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp8ep1/result/runtime/rank6.1789002725280811944.pt.trace.json)
- [Kineto: rank7.1789002725283760292.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp8ep1/result/runtime/rank7.1789002725283760292.pt.trace.json)
- [Memory: memory.rank0.pickle](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp8ep1/result/memory/memory.rank0.pickle)

### cp1ep1

[Detailed statistics](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp1ep1/result/summary.json) · [Raw benchmark](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp1ep1/result/benchmark.json)

- [Kineto: rank0.1789003444181873095.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp1ep1/result/runtime/rank0.1789003444181873095.pt.trace.json)
- [Memory: memory.rank0.pickle](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fixes_20260909/cp1ep1/result/memory/memory.rank0.pickle)
