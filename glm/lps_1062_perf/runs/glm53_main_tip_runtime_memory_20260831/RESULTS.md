# GLM-5.3 tip-of-main — runtime + memory profile (2026-08-31)

## Setup

- Devbox `w6172jq`, one 8xB300 node (ali cluster), trainers @ `4f740aa08` (tip of main; includes #1228 compiled native-FP8 materialization and #1243 GLM B300 fp8-expert-storage configs).
- GLM-5.3 full model (78 layers), seq 131072, 1 datum/step, TP1/PP1/CP8/EP8/ETP1/DP1, LoRA r=32, native_fp8 expert storage, HybridEP, full uniform 1-layer recompute, flash attention.
- Driver: `profile_driver.py`, 3 control windows + 1 memory-profiled + 1 runtime-profiled step.

## Headline

| Metric | Value |
|---|---|
| Control tok/s/GPU | **1189** |
| MFU3x | **10.8%** |
| HFU | 15.7% |
| Control FB | 13.8 s |
| Optim step | 0.1 s (LoRA-only, negligible) |
| Runtime-profiled step wall | 14.99 s |
| GPU busy / idle | 13.70 s / 8.6% |
| Memory: baseline / peak / headroom | 123.8 / ~172 / ~95 GiB per rank |

Consistent with this morning's `ceacc74fc` measurement (1175 mean / 1200 stabilized) — tip-of-main is perf-neutral so far.

## Ranked bottlenecks (exclusive GPU time, wall = 14.99 s)

| # | Bucket | Time | % wall | Notes |
|--:|---|---:|--:|---|
| 1 | HybridEP MoE dispatch/combine | 3.69 s | 24.6% | device_sync 2.39 s (900 calls × 2.65 ms) + dispatch/combine kernels 0.92 s + permute/unpermute 0.38 s. Intra-node EP8 — protocol/sync latency, not network. |
| 2 | DSA attention | 2.82 s | 18.8% | bwd 1.72 s (78 × 21.5 ms) vs fwd 0.58 s (156 × 3.7 ms) + indexer 0.52 s. **bwd is 5.8× fwd per layer** — theoretical is ~2×. Kernel-quality issue in cudnn DSA bwd on SM100. |
| 3 | GEMM (nvjet, dense+expert) | 2.64 s | 17.6% | The healthy compute core. Only ~18% of the step is real matmul. |
| 4 | Elementwise/copy/cat soup | 2.93 s | 19.5% | 77K elementwise (2.27 s) + 5.8K cat (0.66 s): casts, residual adds, FP8 materialization glue, gather/index for DSA top-k. |
| 5 | FP32 output head | 0.93 s | 6.2% | 8 × ~115 ms `cutlass SIMT sgemm_f32` (CUDA cores, no tensor cores) — the intentional fp32 logits projection (`chunked_lm_head.py`, sampler logprob parity). Also drives the memory peak (below) and serializes with `aten::item` loss-read syncs. |
| 6 | No-kernel gaps | 1.29 s | 8.6% | CPU-bound regions: 7,103 `aten::nonzero` + 8,763 `aten::index` calls (dynamic-shape DSA index path), 329 `.item()` syncs (1.88 s CPU-side), 113K launches. |
| 7 | NCCL | 0.32 s | 2.2% | ReduceScatter 185 ms (CP grad sync), AllReduce 83 ms, AllGather 57 ms. Negligible — **not comm-bound at NCCL level**. |

Cross-cutting: **full recompute tax ≈ 4.7 s (31%)**. CheckpointFunction fwd 60.2 ms/layer; bwd 118 ms/layer ≈ recompute(60) + dgrad(58). LoRA ⇒ no base-weight wgrad, so backward proper is only ~1× fwd — recompute adds a full extra forward (+50% over the 2× no-recompute step).

## Memory (rank 0, others within 0.05 GiB)

- Baseline 123.8 GiB (params native-FP8 + LoRA + optimizer + persistent buffers).
- Transient peak +48.2 GiB during the step. Live-at-peak attribution:
  - **`_project_logits` 26.0 GiB** — FP32 logits (16,384 tok/rank × 154,880 vocab × 4 B ≈ 9.5 GiB per tensor; ~2.6 tensors live). Single largest transient consumer.
  - `_bias_dropout_add_func` 14.6 GiB (layer-boundary saves), `sort_topk_by_index` 2.6 GiB, `_linear_forward` 2.4 GiB.
- Headroom at 131K: ~95 GiB/rank. (At 262K: ~49 GiB, per this morning's run.)

## Optimization levers, ranked by impact/effort

1. **Recompute policy (≈31% of step, config-level).** 95 GiB headroom at 131K ⇒ de-recompute half the layers (selective or `num_layers` sweep). Each kept layer saves ~60 ms; 39 layers ≈ −2.2 s ≈ **+17% TPS**. Validate peak ≤ ~240 GiB. Seq-dependent: at 262K headroom is 49 GiB, so make the policy seq-aware.
2. **HybridEP sync+movement (24.6%).** 900 device_sync calls/step (~11.5/layer) at 2.65 ms each is protocol latency on NVLink, not bandwidth. Overlap dispatch/combine with attention/dense GEMM (shared-expert overlap from #1197 notes); audit sync granularity; consider low-latency intranode path. Capture all ranks first — rank0's sync may be waiting on peers.
3. **FP32 output head (6.2% + memory + syncs).** BF16 tensor-core GEMM with FP32 accumulate (or 3xTF32/BF16x9 emulation) keeps logprob parity within tolerance, ~10× GEMM speedup (−0.8 s), halves the 26 GiB logits footprint — which in turn buys headroom for lever 1. Also overlap chunked-CE with the next chunk's GEMM to hide the `.item()` tail.
4. **DSA backward kernel (11.5%).** 21.5 ms/layer vs 3.7 fwd. Check newer cudnn/cutlass SM100 revision or alternative DSA bwd; target 2.5× fwd ⇒ −0.8 s.
5. **Elementwise/cat fusion (19.5%).** Fuse residual adds into GEMM epilogues, kill fp32 upcast copies in the loss path, pack the cats; move DSA index compaction (`nonzero`/`index`) to fixed-shape ops or into the indexer kernel — cuts both kernel time and the CPU-stall gaps.
6. **CUDA graphs / more compile (8.6% gaps).** Shapes are static per step (1 × 131K datum); graph the layer stack minus the dynamic DSA index part.

## Out-of-the-box / structural

- **Microbatch interleave:** 1 datum/step at DP1 leaves no overlap opportunity. 2+ datums per step (or pipelined fb without PP) lets fwd of mb2 overlap bwd of mb1 and amortizes every sync/launch overhead above.
- **FP8 compute, not just storage:** #1243 moved GLM B300 configs to fp8 expert *storage*; B300 FP8 is 2× BF16 — TE fp8 recipe on dense projections + FP8 DSA attention is the next hardware lever.
- **Keep EP intra-node.** NCCL is 2.2%; going 2-node EP16 would trade the cheap intra-node sync for inter-node all-to-all — wrong direction for this shape.
- **Re-examine `.item()` call sites:** 329 syncs/step; batch loss/metric reads to once per step.

## Stack-up estimate

Levers 1+2(half)+3+4+5(half)+6(half) ⇒ step ≈ 10.3–10.8 s ⇒ **~1,550–1,650 tok/s/GPU, MFU3x ~14–15%** near-term; FP8 compute on top is the path to 2×.

## Artifacts

- `glm53_main_tip_131k_runtime.pt.trace.json` (386 MB, rank 0, one step)
- `memory/memory.rank0-7.pickle` (~49 MB each)
- `result.json`, `driver_run.log`, `trainer_srun.log`, configs
