# GLM-5.3 at 262k tokens on 8×B300: where the step goes, and what to do about it

Written 2026-09-03 from two all-rank Nsight Systems captures of tip-of-main (trainers `27a688197`), config TP1/PP1/CP8/EP8, HybridEP, full recompute, LoRA rank 32, native-FP8 experts. The second capture adds NVTX ranges (trainers PR #1298) and CUDA-sync backtraces, so every number below is attributed to a phase and a module, and every host stall to the op that caused it. The previous write-up is kept as `ANALYSIS_20260902_v1.md`; where this document contradicts it, this one is right (the first one guessed at mechanisms this one measured).

Terms used throughout: **rank** = one of the 8 trainer processes, one per GPU. **Step** = one forward_backward of one 262,144-token sequence (CP8 means each rank holds 32,768 tokens of it). **Host sync** = a `cudaStreamSynchronize` where the CPU waits for the GPU; it also empties the GPU's launch queue, so the GPU then runs kernels one at a time as the CPU launches them. **HybridEP device_sync** = the spin-wait kernel HybridEP runs while a rank waits for the other 7 at an all-to-all; it keeps the GPU "busy" from the hardware's point of view but does no work.

## Headline numbers

| | value |
|---|---|
| untraced step (forward+backward), control | 26.7 s (1229 tok/s/GPU; 26.3 s in the 09-02 capture) |
| profiled step | 27.7 s (profiler overhead ~4%) |
| GPU kernel time inside the step, per GPU | 26.1 s |
| forward phase (GPU) | 8.3 s, of which the LM head is 0.96 s |
| backward phase (GPU) | 17.8 s = **6.2 s recomputed forward** + 11.6 s true backward |
| GPU idle (no kernel resident) | 1.4 to 1.9 s per GPU (5 to 7%) |
| GPU busy but only spin-waiting for other ranks (HybridEP device_sync) | 3.4 to 4.1 s per GPU (14%) |
| host blocked in cudaStreamSynchronize | 21 to 22 s per GPU, in 7,521 calls per GPU |
| tensor-pipe active / SM issue / DRAM read, whole step | 22% / 22% / 12% |

## The ranked list

The order is by expected gain per unit of effort, not by bucket size. "Confidence" is about the mechanism; the gain estimates are rougher.

### 1. The DSA context-parallel layout is rebuilt on the GPU in every attention call, with 45 host syncs each time

**What the trace shows.** 7,020 of the 7,521 host syncs per rank per step come from one signature: `nonzero` called by boolean-mask indexing (`tensor[mask]`), inside the `attention` range but outside `dsa_indexer` and `dsa_core`. That is 3,510 in the forward pass and 3,510 in the recompute pass: 45 per attention call, 78 layers, 2 passes (`analysis_nvtx_v3/sync_callchains.csv`).

**Where in the code.** `dsa.py` calls `dsa_layout.build_packed_allgather_cp_query_positions_and_key_reorder`, which calls `build_packed_allgather_cp_local_positions` once for the queries and once per CP rank for the keys, 9 calls. Each of those does five boolean-mask reads on tiny tensors (`seq_starts[nonzero]`, `seq_ends[nonzero]`, `seq_lens[nonzero]`, `segment_starts[nonempty]`, `segment_lens[nonempty]`, lines 248 to 279 of `dsa_layout.py`). 9 × 5 = 45. The inputs are `cu_seqlens` only, which are identical for all 78 layers and both passes, so this work is done 156 times per step when once would do.

**Cost.** Host blocked time inside attention is 9.4 s per rank, but most of that is the CPU waiting for GPU work queued earlier, which is not itself a loss. The measurable losses are: GPU idle of 1.4 to 1.9 s per rank (5 to 7% of the step), and the host-stall share of collective lateness, 0.42 s per step across ranks (see finding 2). Attention wall time on the host is 72 ms per call against 22 ms of GPU work, so the forward is host-bound inside attention.

**Fix.** Compute the layout once per microbatch and cache it on the `PackedSeqParams` (or compute it on the CPU from the host copy of `cu_seqlens`, which is a few integers, then move the result to the GPU once). Zero numerical risk: same tensors, computed once. Expected gain 5 to 8% of the step. Confidence in the mechanism: high (count and call chain match the code exactly). Confidence in the size: moderate; the GPU-idle number is the floor, the collective skew part is the uncertain extra.

The same fix class applies to two smaller siblings: one `.item()` per MoE layer per pass (150 per step) in HybridEP `setup_metadata` (`token_dispatcher.py:1103`, `padded_num_tokens = int(max_num_tokens_across_ep.item())`), and one `torch.nonzero` per layer in the DSA backward compaction (`dsa_cudnn_kernels.py:2102`) that blocks the host for up to 437 ms because it forces the whole layer backward to drain first. Those 228 syncs are few, but each one is a full queue drain at a point where the GPU has a deep queue to lose.

### 2. Ranks wait 3.9 s per step at HybridEP collectives, and it is expert load imbalance, not host jitter

**What the trace shows.** Each rank spends 3.4 to 4.1 s per step inside `hybrid_ep::device_sync_kernel`, 14% of the step, in 900 collectives per rank. 96% of it is in the combine-type collectives (forward combine 1.05 s, recompute combine 1.36 s, backward-of-dispatch 1.32 s), which is where a rank waits for every other rank's expert output.

**Why: measured, not inferred.** I matched the k-th sync on every rank (same collective) and asked, for the last-arriving rank, whether its GPU had been idle (host stall) or busy (more work) since the previous collective (`sync_arrivals.py`). Summed lateness of last arrival vs median arrival is 4.30 s per step; 3.88 s of it (90%) is time the laggard's GPU was busy, only 0.42 s (10%) is idle. So the waits are work imbalance, and the earlier idea that host jitter explains them is wrong.

The imbalance is per layer and rotates: the laggard is GPU 1 or 6 most often but every GPU is last sometimes, and total non-sync busy time across GPUs differs by under 1 s. `range_imbalance.py` measures it directly on the `moe_experts` range (the expert GEMMs): per MoE layer, the heaviest rank does 1.60× the mean expert work (median over 150 instances; p90 2.04×, max 2.50×). If every rank waited for the slowest at every MoE layer, that alone costs 1.81 s per rank in the two forward passes; the backward expert GEMMs are not under a range but scale the same way. Attention is balanced (heaviest rank 1.09× mean) and so is the sparse-attention core (1.01×).

**Caveat that matters.** The profiling input is 262,144 uniformly random token ids (`profile_driver.py`, seed 0xB300). Router load imbalance depends on the token distribution, so this has to be re-measured on real customer-shaped data before anyone invests in it. The `range_imbalance.py` output is the metric to re-run.

**Fix options.** The experts are frozen (LoRA training), so the inference-style remedies apply: place or replicate hot experts so per-rank token counts even out (EPLB-style; requires the routing statistics of real data), or split the heaviest experts' GEMMs. Expected gain: up to the 3.9 s (14%) if perfectly balanced; realistically a third to a half of that. Confidence in the mechanism: high. Effort: high. Do the measurement on real data first.

### 3. The frozen LM head runs in FP32 on CUDA cores, not tensor cores: 1.85 s per step

`chunked_lm_head.py` upcasts hidden states and the base weight to float32 for GLM-5.x DSA models (`fp32_output_head_enabled`). The GEMM lands in `cutlass3x_sm100_simt_sgemm` (SIMT = CUDA cores): 8 forward chunks of 117 ms and 8 backward chunks of 115 ms per rank, 1.85 s per step, 6.8% of the step, at roughly 66 TFLOP/s. A tensor-core path is at least 8× faster. Options: bf16 inputs with fp32 accumulation and upcast the logits afterwards (the loss kernel is what needs fp32), or TF32. Expected gain about 1.6 s (6%). Confidence high; the kernel name is unambiguous. Effort low; needs a parity check against the current fp32 loss.

### 4. Full recompute costs 6.2 s per step, 23% of GPU time, now measured

With NVTX per-layer ranges the recomputed forward is visible as the `layer:<n>` instances that sit inside the backward phase: 6.21 s of GPU kernel time per rank versus 7.34 s for the original forward layers and 11.6 s for the true backward (`fwd_recompute_split.csv`). Peak allocated memory was 213 GB of 287 GB. Selective recompute (skip recompute where the saved activations fit, or checkpoint only the attention core) can trade some of the 74 GB headroom for part of the 6.2 s. Confidence in the cost: high (measured). Gain depends entirely on the memory plan, so no number is claimed here.

### 5. DSA backward: 4.9 s per step, mostly one cuDNN kernel

`FusedSparseAttentionFuncBackward` is 4.86 s of GPU time per rank (18% of the step): the cuDNN `dsa_bwd_sm100` kernel 3.70 s (78 calls of 44 to 56 ms), plus about 1.2 s of index and copy kernels around it, plus the one host `nonzero` per layer mentioned above. The backward kernel is 5.9× the forward kernel (7.9 ms), where 2.5 to 3× is normal for attention backward. This is a vendor kernel: the levers are a cuDNN version check, and removing the compaction round trip. Confidence that there is a quick win: low.

### 6. Copies, cats and elementwise: 5.1 s per step spread over many owners

`cat_copy` (copies, `torch.cat`, gathers, index ops) is 2.95 s per rank and `elementwise` 2.19 s. Who launches them (`category_by_range.csv`, s per rank per step):

| owner | cat_copy | elementwise | notes |
|---|---|---|---|
| true layer backward | 1.70 | 0.76 | copies inside autograd: transposes made contiguous, `CopySlices`, split/cat backward |
| attention projections (`self_attention` minus DSAttention), both passes | 0.78 | 0.36 | MLA layout conversions; `aten::cat` is 1.33 s per step in total |
| MoE experts, both passes | 0.06 | 0.66 | SiLU and the gate multiply run as separate kernels (`activation` 0.37 s on top); a fused SwiGLU exists in TE |
| attention core (`attention` minus indexer/core) | 0.07 | 0.11 | small |

By aten op (autograd-nvtx, GPU time launched under each): `aten::cat` 1.33 s, `aten::mul` 1.07 s, `aten::copy_` 0.95 s, `aten::contiguous`/`clone` 0.61 s, `aten::add` 0.54 s, `aten::gather` 0.29 s, `aten::index_select` 0.26 s. Nothing here is a single fix; it is fusion and layout work, each item worth 1 to 3%.

### 7. Smaller items

- NCCL is 0.72 s per rank (2.6%): the CP key/value all-gather 0.28 s, reduce-scatter 0.29 s. The 405 ms fp32 all-reduce seen in the 09-02 capture did not recur (6.8 ms here), so it was a one-off.
- Rank 0 does about 9% more GPU work in the attention projections than the others (it is the heaviest rank in 119 of 156 instances), worth about 0.3 s of waiting per step. Not chased.
- HybridEP dispatch/combine kernels themselves are 1.75 s per rank at 1.7% NVLink utilization: latency-bound, not bandwidth-bound. Tuning transfer sizes will not help; fewer, larger a2a would.

## How the numbers were produced

Capture: `run_trainer_node_nsys.sh` (nsys launch with CUDA/NVTX/cuBLAS/cuDNN/OS tracing, autograd NVTX, `--cudabacktrace=sync:1000 --python-backtrace=cuda`), one untraced warmup and control step, then `nsys start` with process-tree CPU sampling and 10 kHz GPU metrics on all 8 devices around exactly one `/forward_backward` + `/optim_step` (`capture_one.py`). Reports: `glm53-full-b300-262k-nvtx-all-ranks-gpu-metrics.nsys-rep` (this analysis) and `glm53-full-b300-262k-golden-all-ranks-gpu-metrics.nsys-rep` (09-02, no NVTX). Both exported to sqlite on the pod and processed by the scripts listed in `REPRODUCING.md`; outputs are in `analysis_nvtx_v3/` and `analysis_old_v3/`.

Attribution rule: a kernel belongs to the NVTX range that contained its launch call on the host. Phase ranges never overlap, and per-layer ranges never overlap within one process, so this is exact for them. "GPU idle" is time with no kernel resident on the device inside the step window. Category totals are means over the 8 GPUs; per-GPU tables are in the CSVs.

Known limits: one step, one sequence, synthetic random tokens (affects finding 2 most), profiler overhead about 4% on the step, and the Python frames in the sync call chains are unresolved (the libtorch frames plus the enclosing NVTX range were enough to place every site).

## Files

- `ANALYSIS.md` (this), `ANALYSIS_20260902_v1.md` (superseded), `WORKLOG.md`, `REPRODUCING.md`
- `nvtx_ranges_pr1298.patch`: the instrumentation applied on the pod; PR https://github.com/basetenlabs/trainers/pull/1298
- `nsys_attrib.py`, `sync_callchains.py`, `sync_arrivals.py`, `category_by_range.py`, `range_imbalance.py`: analysis scripts
- `analysis_nvtx_v3/`: outputs for the NVTX capture (SUMMARY.md, categories_by_gpu.csv, kernels_by_category.csv, phases_by_gpu.csv, fwd_recompute_split.csv, layers.csv, sync_waits.csv, sync_wait_summary.csv, sync_arrivals.csv, sync_callchains.csv, host_syncs.csv, category_by_range.csv, imbalance_*.csv, gpu_metrics_by_phase.csv, gpu_idle.csv)
- `analysis_old_v3/`: the same for the 09-02 capture where applicable (no NVTX, no call chains)
- `pre-nsys-nvtx-driver.log`, `glm53-full-b300-262k-nvtx-all-ranks-gpu-metrics.json`: timings of the untraced and traced steps
