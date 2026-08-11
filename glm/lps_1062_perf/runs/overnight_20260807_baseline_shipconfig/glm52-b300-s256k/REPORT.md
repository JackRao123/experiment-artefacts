# LPS-1062 — GLM-5.2 B300 training profile (baseline)

> Imported 2026-08-10 from `~/perf_profiles/lps-1062/glm52-b300-s256k/`. The
> `node0/` / `node1/` trace + memory-snapshot files referenced below remain
> Mac-only at that path (see `DATA.md` at the lps_1062_perf root).

**Date:** 2026-08-06 · **Devbox:** `q480z53`, 2×8 B300 (ali, RoCE fabric) · **Trainer:** `trainers_main @ 0e0b65a6` (trainer-cuda13-sm103 stack, cuDNN DSA patch included) · **Model:** `zai-org/GLM-5.2-FP8` (dequant→bf16 on load)

**Config (golden B300):** TP1 / PP1 / EP16 / CP16 / ETP1, seq 262,144, attention-only LoRA r32/α64, full activation recompute, all comm-overlap flags off, `attention_backend=flash`, `moe_token_dispatcher=alltoall` (default). Synthetic random-token data (seed 0xB300), benchmark window structure mirroring `tests/benchmarking` (1 warmup + 2 main windows; main = 2 datums × 262,144 tokens = 524,288 tokens/step).

## Headline numbers

| Metric | Value |
|---|---|
| Step time (fb, 524,288 tokens) | **73.5 s** mean (76.2 s / 70.8 s) |
| Optim step | 0.06 s (LoRA, negligible) |
| Throughput | **7,134 tok/s = 446 tok/s/GPU** |
| **MFU** | **≈ 7.5–9.4%** (see `mfu_calc.py`; 42.1B active params/token, 745B total; 4×fwd for full recompute; vs 2.25–2.5 PF dense BF16/GPU) |
| Peak GPU memory | 172 GB allocated / 260 GB reserved (of 275 GB) — 95% reserved, fragmentation/retention risk |
| Loss | 12.356 → 12.310 (decreasing, finite grad norms 0.69–0.94) |

Note: this synthetic benchmark's 446 tok/s/GPU is ~2× the ~230 tok/s/GPU quoted in the issue from a real run — production overhead (packing, weight sync, sampler interplay, real data) is a separate gap worth its own look. Even the clean synthetic number is only ~8–9% MFU vs the 20–30% target.

## Where the 76.2 s step goes (rank 0 kineto trace, one full step)

| Bucket | GPU time | % of step |
|---|---:|---:|
| **NCCL total** | **49.4 s** | **64.8%** |
| — `SendRecv` (EP all-to-all + CP p2p), 1,350 calls | 44.9 s | 58.9% |
| — `AllGather` (698 calls) | 3.2 s | 4.2% |
| — `AllReduce` / `ReduceScatter` | 1.3 s | 1.7% |
| DSA sparse attention + indexer | 6.4 s | 8.4% |
| GEMMs (nvjet/cutlass, all matmul) | 6.8 s | 8.9% |
| MoE permute/sort | 1.8 s | 2.4% |
| Other (elementwise, copies, norms) | 7.0 s | 9.2% |
| GPU idle (wall − busy) | 4.8 s | 6.3% |

Structure checks: 156 `CheckpointFunction` = 78 layers × 2 microbatches; bwd/fwd = 1.91× (full recompute firing as expected). `SendRecv` duration is uniform (p50 34 ms, p90 58 ms, p99 84 ms) → **bandwidth-bound, not straggler-bound** — but max = 904 ms and one `AllGather` = 2.25 s show sync-stall tails.

**Reading:** at EP16×CP16 every MoE dispatch/combine crosses nodes over RoCE. ~34 ms/call × 1,350 calls ≈ 47 GB/s effective — the all-to-all is running at network bandwidth. Compute (GEMMs) is only ~9% of the step: the GPUs are mostly waiting on the fabric. This is exactly the issue's thesis (CP/EP-heavy mesh, "config that doesn't OOM" rather than optimized).

## CPU-side findings (per step)

- **`aten::nonzero`: 26,684 calls, 35 s CPU time** — synchronizing GPU→CPU roundtrips in the MoE routing/dispatch path (plus 29.1k `cudaStreamSynchronize`, 33.6k `cudaMemcpyAsync`). These serialize the CPU launch pipeline against the GPU, which blocks comm/compute overlap.
- `aten::item`/`_local_scalar_dense`: 1,621 calls — more syncs.
- 8 vocab-shaped **FP32 SIMT GEMMs** (grid [1210,32]→154,880×4,096, `bias_relu` epilogue, ~115 ms each, 1.85 s/step, 2.4%) — LM-head/CE-loss path running in FP32 on CUDA cores instead of bf16 tensor cores.

## Memory (rank 0 snapshot + /status; node-symmetric: 134.0 GB both nodes)

- Persistent (weights bf16 + LoRA + optimizer + grads): ~131 GB
- Allocated peak during steps: ~172 GB
- Reserved (caching allocator HWM): 260 GB / 275 GB = 95% — large reserved-but-unallocated retention; OOM risk is allocator-fragmentation, not capacity. (`expandable_segments:True` already on.)

## Bottleneck ranking (for the optimization phase, not acted on yet)

1. **EP all-to-all over RoCE = 59% of step.** Levers: EP8 intra-node + expert replication, `moe_token_dispatcher="flex"` (DeepEP fused), smaller CP with more DP, `overlap_moe_expert_parallel_comm=True`.
2. **26.7k GPU syncs/step (`aten::nonzero`)** killing launch pipelining and overlap.
3. **FP32 SIMT vocab GEMMs** (1.85 s/step) — move CE/logits path to tensor cores.
4. **Allocator retention** (260 GB reserved vs 172 GB peak) — constrains headroom for any config that raises activation footprint.

## Artifacts

- `node0/*.pt.trace.json` — rank-0 kineto trace, one full step (995 MB; open in Perfetto)
- `node0/memory.rank0.pickle`, `node1/memory.rank8.pickle` — CUDA allocator snapshots (both nodes)
- `results.json` — raw window timings + profiler responses
- `trainer_srun.log`, `profile_driver.log` — boot + driver logs
- `mfu_calc.py` — reproducible MFU math
- `../trainer-config.json`, `../trainer-server-config.json`, `../profile_driver.py` — exact config + driver used
