# BF16 expert storage: GLM-5.3 1d2m at 131072 tokens

Same source c9a723bf431621ca05580726f4acd01ec618326a, checkpoint values, LoRA rank/alpha32, full one-block recompute, TP1 PP1 ETP1, GC policy and instrumentation as the FP8-storage run. Only expert_weight_storage changes to bf16. Three full warmups, five unprofiled controls, one memory capture and one all-rank runtime capture. No source/test changes for this storage experiment.

## Headline: five controls

| Topology | GPUs | BF16 FB mean ± SD (s) | BF16 TPS/GPU | FP8-storage TPS/GPU | Nominal change | BF16 peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| cp8ep8 | 8 | 0.6395 ± 0.0596 | 25,619 | 25,410 | +0.82% | 26.761 |
| cp8ep1 | 8 | 0.6247 ± 0.0028 | 26,229 | 24,964 | +5.07% | 58.261 |
| cp1ep1 | 1 | 3.4855 ± 0.0050 | 37,605 | 37,291 | +0.84% | 142.098 |

The five-control BF16 means match the proposed TPS/GPU ordering: CP1EP1 > CP8EP1 > CP8EP8. The middle pair is NOT robust: EP8 has one 0.743-second control, while three controls are near 0.603–0.609 seconds. Its 20 additional controls average 26,803 TPS/GPU, versus EP1's five-control 26,229. Do not treat this small, tuning-sensitive gap as a universal topology law or a statistically proven ordering.
Storage deltas are observational, not pure dequantization speedups: independently autotuned permutation launch configurations differ across ranks and runs. All five controls are retained; supplemental controls are not substituted for the headline.

## Control attention/MLP breakdown

GPU elapsed ms, rank0, mean over five controls. Forward/recompute include their complete sublayer paths. Backward numbers partition the gradient dependency path, including waits; they are not exclusive kernel ownership or pure communication times.

| Topology | Block | Fwd attention | Fwd MLP | Recompute attention | Recompute MLP | Bwd attention interval | Bwd MLP interval |
|---|---|---:|---:|---:|---:|---:|---:|
| cp8ep8 | Dense/indexer | 39.36 | 5.78 | 15.23 | 5.83 | 34.98 | 7.41 |
| cp8ep8 | MoE/shared | 14.12 | 21.42 | 12.54 | 28.31 | 34.18 | 22.55 |
| cp8ep8 | MoE/indexer | 31.63 | 18.78 | 16.57 | 17.36 | 35.98 | 17.27 |
| cp8ep1 | Dense/indexer | 39.95 | 5.79 | 14.67 | 5.81 | 35.34 | 7.39 |
| cp8ep1 | MoE/shared | 14.06 | 16.90 | 12.46 | 12.45 | 50.10 | 14.76 |
| cp8ep1 | MoE/indexer | 34.71 | 20.82 | 16.82 | 13.74 | 44.15 | 21.92 |
| cp1ep1 | Dense/indexer | 240.69 | 48.64 | 110.47 | 49.34 | 292.88 | 68.47 |
| cp1ep1 | MoE/shared | 111.25 | 88.87 | 100.78 | 87.53 | 292.21 | 99.55 |
| cp1ep1 | MoE/indexer | 260.47 | 87.79 | 114.72 | 89.52 | 297.23 | 103.64 |

## Account for the whole step

Sums across all three blocks, ms per control. Residual = measured FB minus the six sublayer timing sums; includes embedding/head/loss, RPC and unassigned scheduling gaps. It is not exclusively CPU time.

| Topology | Fwd attn | Fwd MLP | RC attn | RC MLP | Bwd attn | Bwd MLP | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| cp8ep8 | 85.11 | 45.98 | 44.34 | 51.51 | 105.14 | 47.23 | 260.23 |
| cp8ep1 | 88.71 | 43.51 | 43.96 | 32.00 | 129.59 | 44.06 | 242.83 |
| cp1ep1 | 612.41 | 225.29 | 325.96 | 226.39 | 882.32 | 271.66 | 941.47 |

## Why the intuitive ordering is not guaranteed

CP1 wins TPS/GPU here because its latency is less than eight times the CP8 latency. One GPU does all 131072 tokens; each CP8 GPU gets 16384. Removing CP also eliminates its collectives and amortizes fixed costs over more local work. CP1 remains much slower in request latency, despite better efficiency per GPU.
EP1 removes cross-GPU dispatch/combine, but retains token routing, permutation and expert GEMMs. EP8 has 32 local experts versus EP1's 256. The theoretical balanced mean is 4096 routed tokens/expert at EP8 versus 512 at EP1. Actual routing is not force-balanced. Expert weight reuse and launch geometry change, so communication removal alone cannot establish the ordering.

## A concrete remaining autotuning bug

Transformer Engine 2.16.0 common/triton/permutation.py decorates _sort_chunks_by_map_kernel with triton.autotune(key=["hidden_size"]). The key omits token count. Its choice is first made during the server's tiny 64-token startup warmup and can be reused for the 131072-row routed activation tensor. Different ranks select different cached BLOCK_SIZE values.

| Same-width BF16 chunk copy | Rank 0 trace | Rank 6 trace |
|---|---:|---:|
| Grid | 131072 × 6 | 131072 × 96 |
| BLOCK_SIZE inferred from H=6144 | 1024 | 64 |
| Kernel time | 0.522 ms | 6.464 ms |

The slow configuration launches 16 times as many blocks and takes about 12.4 times as long. An isolated same-input microbenchmark directly confirms it: BLOCK_SIZE64 6.485 ms, 1024 0.512 ms, 4096 0.476 ms, with bitwise-identical output and permuted probabilities. This is a real configuration problem, not a fundamental EP1 limitation. chunk_sort_probe.py and its log are retained. The installed TE implementation was NOT modified for this comparison.

The MoE/shared recompute trace shows almost identical GEMM kernel sums on ranks0/6 (14.40/14.12 ms), but permutation/top-k sums of 2.66/10.46 ms. MLP elapsed is 12.70/20.69 ms. The gap is primarily in permutation, not a large GEMM regression between these ranks. Kernel sums overlap across streams and are not additive wall time.
Five-control timing agrees: rank6 spends 20.49+20.04=40.53 ms in shared-MoE MLP recompute+backward versus rank0's 12.45+14.76=27.21 ms. Rank0 then spends 50.10 ms in the attention-backward interval versus rank6's 36.46 ms. A late rank's local work appears as waiting in its peers.

A proper next fix is to key the permutation tuner by workload size and relevant direction/probability/stride modes, or retune at the actual workload. Pinning a good configuration is a useful experiment, but is not yet an end-to-end fix in this run.

## Communication is not larger at EP1

Kineto process-group metadata identifies the CP operations directly. Both CP8 runs have 11 CP AllGathers with 130088960 input / 1040711680 output bytes per rank, and three CP ReduceScatters with 452984832 input / 56623104 output bytes. These are logical tensor sizes, not measured wire bytes.
The captured AllGather kernel sums are 21.55 ms (EP8) and 21.63 ms (EP1). ReduceScatter sums are 6.11 versus 30.06 ms despite identical sizes; those durations include peer waits. The 26,537,984-byte coalesced gradient allreduce is also the same in both. HybridEP additionally performs its own dispatch/combine and tiny metadata reductions at EP8.

## What removing dequantization changed

Old FP8-storage EP1 issued 512 weight-dequant kernels per MoE forward (~6.2 ms summed GPU kernel time); BF16 issues zero. Control MoE-forward MLP times changed 33.34→16.90 ms (shared-index block) and 35.42→20.82 ms (indexer block). Do not attribute the entire difference to dequantization: the old rank0 chunk sort used a slower cached configuration too.
Runtime inventories confirm every routed expert weight is an ordinary BF16 Parameter with no quantized payload. EP1 has 1024 weight tensors across the two MoE blocks; EP8 has 128 per GPU. Resident expert matrices total 36 GiB/GPU at EP1 versus 4.5 GiB/GPU at EP8. This is capacity footprint, not a DRAM-traffic measurement.

## Weighted full-model block estimate

3 dense/indexer + 57 MoE/shared + 18 MoE/indexer. This excludes embedding/head/loss/optimizer overhead and is not a claim that the full model fits EP1.

| Topology | Fwd s | Recompute s | Actual bwd s | Block total s |
|---|---:|---:|---:|---:|
| cp8ep8 | 3.069 | 3.004 | 4.320 | 10.393 |
| cp8ep1 | 2.902 | 2.033 | 5.015 | 9.949 |
| cp1ep1 | 18.544 | 14.890 | 30.630 | 64.064 |

## Files and limitations

Only expert storage changed. No trainer code or tests changed. GC fix remains enabled and no old hundreds-of-ms GC pauses appeared in the five accepted controls. Profile captures are separate steps; their startup/shutdown and synchronization overhead are not headline measurements. GPU-scope tables use interval envelopes across streams to include TE GEMMs lacking CPU ownership links; boundary attribution is inferred. No hardware counters were collected.

### cp8ep8

[Statistics](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep8/result/summary.json) · [Benchmark](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep8/result/benchmark.json) · [GPU scope breakdown](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep8/result/analysis/gpu-scopes.json)

- [Kineto rank0.1789009421354543970.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep8/result/runtime/rank0.1789009421354543970.pt.trace.json)
- [Kineto rank1.1789009421352193501.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep8/result/runtime/rank1.1789009421352193501.pt.trace.json)
- [Kineto rank2.1789009421351427590.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep8/result/runtime/rank2.1789009421351427590.pt.trace.json)
- [Kineto rank3.1789009421351845881.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep8/result/runtime/rank3.1789009421351845881.pt.trace.json)
- [Kineto rank4.1789009421352694831.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep8/result/runtime/rank4.1789009421352694831.pt.trace.json)
- [Kineto rank5.1789009421352412352.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep8/result/runtime/rank5.1789009421352412352.pt.trace.json)
- [Kineto rank6.1789009421352116940.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep8/result/runtime/rank6.1789009421352116940.pt.trace.json)
- [Kineto rank7.1789009421353062104.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep8/result/runtime/rank7.1789009421353062104.pt.trace.json)
- [Memory memory.rank0.pickle](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep8/result/memory/memory.rank0.pickle)

### cp8ep1

[Statistics](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep1/result/summary.json) · [Benchmark](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep1/result/benchmark.json) · [GPU scope breakdown](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep1/result/analysis/gpu-scopes.json)

- [Kineto rank0.1789009057160879720.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep1/result/runtime/rank0.1789009057160879720.pt.trace.json)
- [Kineto rank1.1789009057159052012.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep1/result/runtime/rank1.1789009057159052012.pt.trace.json)
- [Kineto rank2.1789009057159975579.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep1/result/runtime/rank2.1789009057159975579.pt.trace.json)
- [Kineto rank3.1789009057158744975.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep1/result/runtime/rank3.1789009057158744975.pt.trace.json)
- [Kineto rank4.1789009057159888329.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep1/result/runtime/rank4.1789009057159888329.pt.trace.json)
- [Kineto rank5.1789009057159712892.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep1/result/runtime/rank5.1789009057159712892.pt.trace.json)
- [Kineto rank6.1789009057159933580.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep1/result/runtime/rank6.1789009057159933580.pt.trace.json)
- [Kineto rank7.1789009057158962865.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep1/result/runtime/rank7.1789009057158962865.pt.trace.json)
- [Memory memory.rank0.pickle](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp8ep1/result/memory/memory.rank0.pickle)

### cp1ep1

[Statistics](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp1ep1/result/summary.json) · [Benchmark](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp1ep1/result/benchmark.json) · [GPU scope breakdown](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp1ep1/result/analysis/gpu-scopes.json)

- [Kineto rank0.1789009971294311572.pt.trace.json](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp1ep1/result/runtime/rank0.1789009971294311572.pt.trace.json)
- [Memory memory.rank0.pickle](/Users/jackrao/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_bf16_20260909/cp1ep1/result/memory/memory.rank0.pickle)
