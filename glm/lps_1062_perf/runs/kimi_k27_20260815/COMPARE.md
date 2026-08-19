# GLM-5.2 vs Kimi-K2.7-Code — B300 profile comparison (2026-08-15)

Same box shape (2x8 B300, ali-apse7), same driver (`profile_driver_new.py`
protocol: warmup -> kineto-traced step -> untraced controls), same operating
point (131k x d4 = 524,288 tok/step, LoRA r32, full recompute). Each model runs
its **golden B300 parallel layout**.

| | GLM-5.2 | Kimi-K2.7-Code |
|---|---|---|
| layout | TP1/PP1/EP16/**CP16** | TP8/PP1/EP16/CP1 (SP on) |
| code | campaign branch (ship env + TF32 head) | `main` @ 29b59564 (clean) |
| control step | **46.4-46.7 s** | **50.8-51.3 s** |
| **tok/s/GPU (control)** | **~705** | **641** |
| traced step | 49.5 s (+6.4%) | 52.8 s (+3.3%) |
| MFU (LoRA-corrected, own arch) | 6.1% | **15.9%** |
| HFU | 8.8% | 21.7% |
| peak mem | ~201 GiB | 170 GiB |
| trace | `round3/anchor-318g61w/rank0_131k-d4-steady.pt.trace.json` | `kimi27/traces/...4347...pt.trace.json` |

**Headline: GLM is ~10% faster in tok/s/GPU. Kimi extracts ~2.6x the hardware
throughput per token (MFU 15.9% vs 6.1%)** — GLM wins only because DSA slashes
how much work a token is (see below).

## Where the step goes (rank-0 kernel-time buckets, traced step)

| bucket | Kimi (52.8s wall) | GLM (49.5s wall) |
|---|---:|---:|
| EP all-to-all (SendRecv) | 22.0s (38%) | 27.3s (56%) |
| attention kernels | 17.1s (30%) | 5.1s (10%) |
| TP / CP comm (AG+RS) | 8.5s (15%) | 2.5s (5%) |
| GEMM (nvjet etc.) | 4.3s (8%) | 5.4s (11%) |
| MoE permute/sort | 2.0s | 1.8s |
| other glue | 2.7s | 5.8s |
| **kernel-sum (overlap-inflated)** | 57.2s | 48.6s |
| compute-only sum (excl NCCL) | 26.5s | 18.5s |

Overlap (kernel-sum vs wall): Kimi 4.4s of comm hidden under compute; GLM ~0
(fully serial).

## Shared bottleneck (hypothesis confirmed): EP a2a

Both are EP-all-to-all dominated. Call counts match the structure exactly
(9 SendRecv per MoE-layer-microbatch = dispatch+combine+probs x 3 passes):
Kimi 1080 = 60x2x9, GLM 2700 = 75x4x9.

Byte model (measured durations match it): per dispatch per rank, Kimi moves
131072/SP8 = 16384 tok x topk8 x 7168 x 2B = 1.88 GB; GLM moves 131072/CP16 =
8192 tok x 8 x 6144 x 2B = 0.80 GB. Totals/step/rank: Kimi 2.03 TB, GLM 2.17
TB — **near-identical bytes** (7% apart). But GLM's a2a takes 24% longer for
those bytes — i.e. ~16% worse per byte, because CP16's smaller messages (0.80
vs 1.88 GB) amortize per-call latency + straggler wait worse across 2.5x more
rounds. Both otherwise run at the ali RoCE node-egress ceiling (~half the a2a
stays on-node over NVLink; the inter-node half saturates the 6x400G bonds).
Neither is at a code bug here; both are at fabric spec. The levers are fewer
bytes (topk/hidden) or hiding it under compute (the GLM overlap campaign's
W1/W2/W3 thesis applies to both).

## Where one is abnormally worse than the other

1. **Kimi's parallel-layout comm tax: TP8+SP = 8.5s (16%) vs GLM CP16 = 2.5s
   (5%).** Same 131k sequence, two sharding strategies. TP8+SP pays ~3900 small
   RS/AG collectives (~2ms avg, latency-bound on NVLink) on the critical path,
   mostly NOT overlapped. CP16 moves KV less often (1.7s AllGather). This is
   the clearest config-level gap: a CP-based layout for Kimi (e.g. CP8) would
   likely reclaim several seconds/step. (Why GLM is CP16: DSA forces TP1.)
   Confidence: high on the measurement, medium on the reclaim estimate.

2. **Attention: Kimi 17.1s (30%) vs GLM 5.1s (10%).** Full-causal MLA vs DSA
   top-2048. Per-token fwd FLOPs @131k: Kimi 163.7 GF vs GLM 21.7 GF (7.5x).
   This is architectural, not a bug — and Kimi's flash kernels are efficient
   (blended ~50% of peak: fwd ~76%, bwd ~47%). But it is *the* reason Kimi
   loses the tok/s race despite better MFU. GLM's DSA kernels are only ~23%
   efficient (sparse gather) but 7.5x cheaper in FLOPs.

3. **GLM's small-kernel glue: 5.8s (124k calls) vs Kimi 2.7s (32k calls).**
   DSA machinery (indexer fwd/topk, index_copy, gathers) + more elementwise.
   ~3s/step GLM-only tax. Confidence: high (kernel counts).

4. **GLM's host is sync-heavy: cudaStreamSynchronize 20.1s + cudaEventSync
   16.4s** (vs Kimi 0.6s + 45.6s). GLM's host blocks ~37s/49.5s on syncs
   (residual nonzero/sync classes the campaign chased); the GPU stays fed here
   (idle +1.9%) so it doesn't cost wall *today*, but it's fragile. Kimi's host
   waits are long (45.6s event-sync) but that's just "GPU is busy" — benign.

5. **Kimi had ONE 1.17s SendRecv straggler stall** (~2% of the traced step,
   at ~38% through). p99 is 50ms, so this is a ~20x outlier — a one-off
   cross-rank jitter, not systematic. Not present in the (faster) controls.

## Net

GLM faster (705 vs 641 tok/s/GPU) because DSA makes a token cheap; Kimi's
stack is otherwise the tighter one (MFU 2.6x, memory 170 vs 201 GiB, cleaner
host path, better overlap). Kimi's recoverable gap is the TP8+SP comm tax
(~6s/step, ~12%); GLM's recoverable gap remains the a2a straggler/latency
overhead the campaign already documented (its a2a is 27.3s vs Kimi's 22.0s for
the same bytes — 2.5x more rounds at CP16's smaller messages).
