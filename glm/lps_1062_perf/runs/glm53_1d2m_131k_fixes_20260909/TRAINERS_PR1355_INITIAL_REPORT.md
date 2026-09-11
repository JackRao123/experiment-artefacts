# GLM-5.3 131k topology experiment

Overnight investigation completed. Tracking draft PR: #1355, stacked on #1157
for multi-rank runtime capture. Production save/load and multi-adapter behavior
remain outside this experiment's validation.

Latest follow-up: all three BF16 topology reruns, the singleton identity-sort
fix, matched-initialization MFSDP validation, and grouped-GEMM probes are
complete. Full-model FSDP CP8 is validated at 1399 TPS/GPU; a lower-memory CP4
recipe is validated at 1522 TPS/GPU with zero allocator retries/frees across
forty rank-controls. CP2 failed activation capacity. Persistent buffers removed
the original full-model 84-second allocation-churn pathology (7.19x speedup).
Current summaries:

- `experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_ep1_rootcause_20260909/OVERNIGHT.md`
- `experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_ep1_rootcause_20260909/ARTIFACTS.md`
- `experiment_artefacts/glm/lps_1062_perf/runs/glm53_full_131k_fsdp_headmem_20260910/RESULTS.md`

- `experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_ep1_rootcause_20260909/RESULTS.md`
- `experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_ep1_rootcause_20260909/GEMM.md`
- `experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fsdp_cp8ep1_20260909/RESULTS.md`
- `experiment_artefacts/glm/lps_1062_perf/runs/glm53_1d2m_131k_fsdp_parity_20260909/RESULTS.md`

The original native-FP8-storage results below are retained as historical
comparisons, not substituted for the latest BF16 run. No test files changed.

## Scope

- 1 dense/indexer + 1 MoE/shared-index + 1 MoE/indexer block, real GLM-5.3 weights.
- 131072 tokens, one datum, LoRA rank/alpha 32, full one-block recompute, native-FP8 expert storage.
- CP8EP8 and CP8EP1 on eight HGX B300 GPUs; CP1EP1 on one GPU.
- Five unprofiled controls, separate memory and all-rank runtime captures.
- Attention and MLP CUDA-event timings for original forward, recompute, and backward.
- Cyclic-GC pause annotations to distinguish host stragglers from allocator/driver stalls.
- No tests added or modified by this experiment. Validation uses numerical probes and benchmark artifacts.

## Fixes under evaluation

1. Singleton HybridEP groups use the existing alltoall local permutation path. The cooperative HybridEP kernel can exceed its residency limit with 256 local experts even though no network transfer is needed. Preserve the requested non-overlapped shared-expert schedule when falling back.
2. Plain-causal CP1 indexer chunks use the cuDNN causal-offset scorer instead of per-head FP32 bmm. Dependency: basetenlabs/Megatron-LM#76, via basetenlabs/Megatron-Bridge#84.

The causal-offset numerical probes preserve causal bounds and valid-key counts. Selected-key overlap with the existing FP32 reference was 1.0 at 2048/4096 tokens and 0.99999994 at 8192 tokens (top-k 128/2048/2048). Probe wall times include compilation/autotuning and are not performance claims.

## Interpretation constraints

Do not assume a topology speed ordering: CP1 does eight times as much token work per GPU, and EP1 trades transfers for more local expert matrices with smaller batches. Report step latency and tokens/s/GPU separately. Collective kernel durations include waiting for peers, not just transfer cost. Changes to GC scheduling must account for collection cost rather than hide it outside the headline measurement.

Results and the straggler diagnosis will be appended after the 131k runs finish. Full artifacts and reusable scripts live under the experiment's runs directory, not in the source/test tree.

## Straggler diagnosis

The CP8EP1 131k probe reproduced 0.64–1.29 second controls at only 53.6 GiB peak allocated. Rank-local GC callbacks measured generation-2 pauses of 598.6 ms (rank 4) and 636.2 ms (rank 6) in slow controls. The all-rank runtime capture shows a 591.1 ms generation-2 collection on rank 7 while peers wait for roughly 600 ms. No cudaMalloc/cudaFree/cuMem allocation calls were present in the runtime API rollups. This is host cyclic-GC scanning, not evidence of tight GPU memory or allocator-driver churn.

The opt-in `BT_FREEZE_GC_AFTER_WARMUP=1` experiment collects unreachable warmup objects once after the first optimizer step, then freezes surviving long-lived objects out of future GC scans. Automatic cyclic GC remains enabled for newly created objects. Resource teardown unfreezes objects before the existing full collection. The one-time collection remains included in optimizer timing; it is not subtracted from measurements. This is experimental/default-off pending longer lifetime validation.

## First validated result

CP8EP1 with the active policy: 0.65631 ± 0.01433 seconds FB across five controls, 24964 tokens/s/GPU, 53.609 GiB allocated / 55.020 GiB reserved peak. An additional 20-control stability run averaged 25074 tokens/s/GPU. Final comparisons use three complete warmups before five controls, consistently across topologies.

The policy must extend an existing permanent generation (the interpreter already had 375 frozen objects before the model loaded). One-time collection is synchronized across ranks. Diagnostic `gc.get_freeze_count()` is cached on activation: it walks the permanent generation and must not be called on every GC event. Earlier guard-skipped and observer-contaminated attempts are retained separately, not used as final measurements.

## Final results

All three were measured on source c9a723bf431621ca05580726f4acd01ec618326a with the GC experiment enabled. Three complete warmups precede five controls; each topology also has 20 additional stability controls and an all-rank runtime capture plus a memory snapshot. Source changes are unchanged by this results-only update.

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

The CP1 trace contains 128 cuDNN indexer-forward kernels totaling 159.400 ms and zero calls to the former per-head FP32 SIMT fallback signature. No NCCL kernels occur in CP1. Numerical end-to-end loss/grad-norm results remain finite and close across topologies.

The 20-control stability means were 26427 tokens/s/GPU (CP8EP8), 25074 (CP8EP1), and 37307 (CP1EP1). These are supplementary, not replacements for the five-control headline. Do not infer that removing EP must improve every phase: EP1 trades transfers for 256 local experts and smaller per-expert batches; the separated MLP timings expose that tradeoff.

GC remains enabled for new objects; this opt-in/default-off process-wide freeze policy needs longer full-model and resource-reload validation before production rollout. No tests were added or modified. Pre-push lint/format/type checks passed.

## Resident BF16 expert-storage comparison

Same tested source, 131072 tokens, GC policy, three warmups and five controls; only expert storage changed. Loaded-weight inventories verify ordinary BF16 Parameters without quantized payload. No additional trainer/kernel/test changes for this comparison.

| Topology | FB mean ± SD (s) | TPS/GPU | Prior FP8-storage TPS/GPU | Peak allocated GiB |
|---|---:|---:|---:|---:|
| CP8EP8 | 0.6395 ± 0.0596 | 25,619 | 25,410 | 26.761 |
| CP8EP1 | 0.6247 ± 0.0028 | 26,229 | 24,964 | 58.261 |
| CP1EP1 | 3.4855 ± 0.0050 | 37,605 | 37,291 | 142.098 |

The five-control BF16 means match CP1EP1 > CP8EP1 > CP8EP8, but the middle pair is not robust. One 0.743-second EP8 control depresses its mean; a subsequent 20-control EP8 check gives 26,803 TPS/GPU. Storage deltas are observational, not fully isolated causal speedups, because permutation autotune choices differ across processes.

### Remaining issue: chunk-sort autotune key omits token count

Installed TE 2.16.0 `common/triton/permutation.py` uses `triton.autotune(key=["hidden_size"])` for `_sort_chunks_by_map_kernel`. The choice first made during the 64-token startup warmup can be reused for the long-sequence routed tensor. Different ranks cached dramatically different BLOCK_SIZE values:

- Same BF16 width 6144 and 131072 rows: rank0 grid 131072×6 / BLOCK_SIZE1024 takes 0.522 ms; rank6 grid 131072×96 / BLOCK_SIZE64 takes 6.464 ms.
- Isolated same-input experiment: BLOCK_SIZE64 6.485 ms, 1024 0.512 ms, 4096 0.476 ms, with bitwise-identical output and permuted probabilities.
- MoE/shared recompute: rank0/rank6 GEMM kernel sums are 14.40/14.12 ms, but permutation/top-k sums are 2.66/10.46 ms; MLP elapsed is 12.70/20.69 ms. This is not explained by a large GEMM regression between those ranks.
- CP AllGather and ReduceScatter tensor sizes are identical between EP8 and EP1. ReduceScatter kernel sums are 6.11 versus 30.06 ms in the captures; peer arrival delays contribute to this duration and must not be called extra transfer volume.

The FP8-storage EP1 path had 512 dequant kernels per MoE forward (~6.2 ms summed GPU time); BF16 has zero. But the old FP8 rank0 reverse chunk sort also used BLOCK_SIZE64 (6.482 ms), versus BLOCK_SIZE4096 (0.465 ms) in the BF16 run. Therefore the full MLP improvement cannot be assigned to dequantization alone.

The installed TE tuner was not patched in this storage experiment. Next step: key tuning by token count plus relevant direction/probability/stride modes and rerun a controlled comparison. Runtime traces, metadata, source snapshot, numerical probe, and reusable analysis scripts are retained in the BF16 experiment run directory.
# Follow-up: singleton expert chunk sorting

The all-to-all dispatcher performs a source-rank/expert chunk transpose even
when EP=ETP=1. Both chunk maps are identity in that case: the first token
permutation already grouped the rows by expert. Core now skips the redundant
sort and inverse sort for the singleton topology, retaining ordinary routing,
token permutation, expert GEMMs and final token merging. This also avoids the
previously diagnosed Transformer Engine chunk-sort autotuning pathology on
that path, without modifying Transformer Engine.

An experiment-only GPU probe confirmed bitwise-equal outputs, probabilities,
input gradients and probability gradients for fused and non-fused sorting,
including uneven expert counts. No tests were added or modified.

The first BF16 CP8EP1 follow-up retained three warmups, five controls, memory
and all-rank runtime captures. Its five controls averaged 26,130 TPS/GPU
(0.6270 ± 0.0343 seconds); 20 additional controls averaged 27,556 TPS/GPU
(0.5946 seconds). Both sets are retained, not selectively combined. The
new traces confirm the chunk-sort kernels disappeared. Other topology reruns
and FSDP feasibility work are in progress; this is not the final comparison.

The older BF16 EP8 trace contains about 20.5 ms total dispatch/combine GPU
kernel time across the complete step, not hundreds of milliseconds. Summed
kernel durations are not a promise of equal exposed wall-time savings.
Also, the HTTP TPS denominator and the backend timer have different boundaries:
the backend step metric includes forward/backward plus optimizer, whereas the
headline HTTP measurement is forward/backward only. Reports must explicitly
separate these rather than treating either as pure kernel time.

