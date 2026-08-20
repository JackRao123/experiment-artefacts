# TRACE ANALYSIS — rank 8 (pipeline stage 1 leader), PP2/CP8/EP8, GLM-5.2, 131k, d4

**Trace:** `/Users/jackrao/perf_profiles/lps-1062/pp2cp8ep8/diag_d4_rank8.pt.trace.json` (748 MB, kineto)
**First-ever capture of stage 1.** One d4 step (4 microbatches), window = ProfilerStep#0 = **69.16 s**
(t=0 ≡ ts 3605874125551827). 40 MoE layers + chunked TF32 LM head/loss, full activation recompute,
2 nodes × 8 B300 ("L20D" masquerade). Companion to the stage-0 skew analysis (`TRACE_SKEW_ANALYSIS.md`).

## Methodology

- Perfetto `trace_processor` (CLI, one-SELECT queries) → CSV extracts; Python interval math on top.
  Scripts + extracts in scratchpad `lps1062r8/` (`an1..an8.py`, `q_*.sql`, `allgpu2.csv`, `p2p.csv`,
  `a2a.csv`, `hostapi.csv`).
- "GPU busy" = union of all `kernel`+`gpu_memcpy`+`gpu_memset` slices with dur>0, across ALL streams
  (parked NCCL counts as busy). "Idle/gap" = window minus that union — the direct measure of
  CPU-blocked / launch-starved time.
- Durations from trace_processor are ns. Regions were delimited from ground truth in the trace
  (p2p kernel timestamps, TF32 GEMM clusters, first sparse-attn-fwd after LM head), not guessed.

### Validation of prior on-box aggregates (all confirmed)

| metric | prior | this pass |
|---|---|---|
| p2p (tid 35 / track 6) | 16 kernels, 27.10 s, max 8.07 s | 16 kernels, 27.097 s, max 8.071 s ✓ |
| EP a2a (tid 67 / track 16) | 1440, 8.56 s, max 156 ms | 1440, 8.564 s, max 156.3 ms ✓ |
| AllReduce_Sum_u64_TREE_LL | 2 calls, 6.49 s | 6.486 s — but split is **6.474 s + 0.011 s**, not 2×3.2 s |
| AllGather (CP) | 724, 2.15 s | 724, 2.147 s ✓ |
| compute busy | 18.79 s (27 %) | 19.40 s kernels (+0.15 memcpy/memset); compute-covered wall = 28 % ✓ |
| comm/compute overlap | ~1 % | 0.83 s of 45.05 s comm resident = 1.8 % ✓ |

**Stream map:** track 4 = main compute stream (stream 7, 214k kernels) — and it ALSO carries the CP
AllGathers, ReduceScatters, Broadcasts and the u64 telemetry AllReduce (they serialize with compute);
track 6 = pipeline p2p (stream 35); track 16 = EP a2a (stream 67); tracks 15/17/18–24/30 = side GEMM
streams (~3.4 s total); track 8 = spill-track of stream 7.

---

## Q1 — p2p park pattern (the convoy, seen from stage 1)

All 16 tid-35 kernels (t relative to window start):

| # | t (s) | dur | role |
|---|---|---|---|
| 1 | 0.256 | **5758.5 ms** | PARK — wait for f1 activations (stage-0 first fwd + warmup) |
| 2 | 6.015 | 3.74 ms | recv f1 payload |
| 3 | 17.434 | 0.034 ms | handshake (coalesced group open) |
| 4 | 17.434 | 4.41 ms | send b1 grads |
| 5 | 17.444 | **8070.6 ms** | PARK — wait for f2 |
| 6 | 25.515 | 3.79 ms | recv f2 payload |
| 7 | 33.082 | 0.027 ms | handshake |
| 8 | 33.084 | 3.83 ms | send b2 grads |
| 9 | 33.094 | **6610.6 ms** | PARK — wait for f3 |
| 10 | 39.705 | 3.75 ms | recv f3 payload |
| 11 | 47.144 | 0.030 ms | handshake |
| 12 | 47.144 | 4.15 ms | send b3 grads |
| 13 | 47.154 | **6625.7 ms** | PARK — wait for f4 |
| 14 | 53.781 | 3.94 ms | recv f4 payload |
| 15 | 61.376 | 0.031 ms | handshake |
| 16 | 61.378 | 3.92 ms | send b4 grads (no recv → no park) |

- Per-cycle pattern is the exact mirror of stage 0's `[handshake, transfer, park, transfer]`:
  stage 1 does **[send grads ~4 ms → PARK 6.6–8.1 s → recv activations ~3.8 ms → compute burst]**.
- Parks sit at the **start** of each microbatch cycle, chained into the fused
  `send_backward_recv_forward` — confirming the theory: stage 1 parks waiting for ACTIVATIONS.
- Parks total 27.07 s of the 27.10 s resident; wire time is only ~31 ms per step.
  Payload ≈ 16384 tok × 6144 × 2 B ≈ 201 MB in ~3.8 ms ≈ **53 GB/s = 400G wire speed**. The link is fine.
- Convoy arithmetic closes: stage-1 park (7.1 s avg steady) ≈ stage-0 work (6.9 s); stage-1 work
  (7.5 s) ≈ stage-0 park (8.1 s). Cycle ≈ work₀ + work₁ — the stages strictly alternate, zero overlap.

## Q2 — true GPU-empty gaps (host stalls)

Union of ALL kernel intervals (incl. parked NCCL): busy 63.18 s → **idle 5.98 s / step (8.6 %)**.
Of that: **3.41 s in 103 gaps >1 ms**; ~0.16 s in 0.1–1 ms gaps; **~2.4 s in sub-0.1 ms micro-gaps**
(launch latency drag — see Q4). Sibling stage-0 figure was ~5 s/step: same magnitude.

Gap mass by size: 1–2 ms: 0.06 | 2–5: 0.06 | 5–10: 0.10 | 10–50: 0.33 | 50–200: 1.21 | >200 ms: 1.66 s.

Top-20 gaps >1 ms, classified:

| t (s) | ms | region | bracket (before → after) |
|---|---|---|---|
| 68.471 | 652.1 | optim | Memcpy DtoH(pinned) → multi_tensor_apply (FusedAdam) — **CPU-side optimizer stall** |
| 6.879 | 353.7 | fwd1 | FillFunctor<int> → FillFunctor<float> — warmup (plan build/autotune) |
| 6.206 | 227.8 | fwd1 | same warmup signature |
| 61.407 | 222.7 | tail | Broadcast → Memcpy HtoD — post-step bookkeeping |
| 7.821 | 202.6 | fwd1 | warmup |
| 9.741 | 196.5 | fwd1 | warmup |
| 56.083 | 183.0 | **lmhead4** | direct_copy(fp32 cast) → CUDAFunctor_add<float> — **LM-head chunk-boundary host stall** |
| 8.828 | 136.4 | fwd1 | warmup |
| 11.595 | 132.8 | bwd1 | warmup |
| 14.537 | 99.2 | bwd1 | warmup |
| 12.453 | 89.3 | bwd1 | warmup |
| 0.133 | 83.9 | init | HtoD staging |
| 13.633 | 82.3 | bwd1 | warmup |
| 9.175 | 77.7 | fwd1 | warmup |
| 6.070 | 77.4 | fwd1 | warmup |
| 6.150 | 52.3 | fwd1 | warmup |
| 10.472 | 42.9 | fwd1 | warmup |
| 0.033 | 37.0 | init | HtoD staging |
| 6.040 | 29.1 | fwd1 | warmup |
| 0.006 | 26.1 | init | DtoH → HtoD staging |

Idle by region: init 0.25 | **mb1 (warmup) 2.63** | steady mb2–4 ≈ 0.65/mb (0.32 fwd + 0.33 bwd +
0.014 lm; >1 ms portion only 30–40 ms/mb) | lmhead4 anomaly 0.20 | tail 0.27 | **optim 0.65**.
So in steady state, stage 1's >1 ms host stalls are nearly absent; the recurring cost is micro-gap
drag (~0.65 s/mb) plus the two big CPU-side stalls at step end (652 + 223 ms).
All mb1 warmup gaps share one signature (FillFunctor<int>→FillFunctor<float>: cuDNN/DSA plan
building + workspace fills). This is ProfilerStep#0 — likely non-recurring; unverified (caveat).

## Q3 — LM head / loss region (chunked, TF32)

Found exactly 4 regions (one per mb fwd tail), delimited [first fp32-cast copy before first big TF32
GEMM → first sparse-attn-fwd recompute kernel after]:

| mb | extent | kernel time | idle (sync cost) | D2H copies |
|---|---|---|---|---|
| 1 | 156.5 ms | 105.8 ms | 13.1 ms | 62 |
| 2 | 120.8 ms | 106.6 ms | 13.6 ms | 62 |
| 3 | 120.2 ms | 106.1 ms | 13.6 ms | 62 |
| 4 | **303.0 ms** | 105.4 ms | **197.3 ms** | 62 |

Internal structure per mb (all four identical):
1. **Chunk loop ×4**: [2× fp32-cast copies 1.25 ms] → `nvjet_sss_tf32_128x256_TNT` **7.14 ms**
   → [2× copies 0.83 ms]. The casts alone are ~16.6 ms/mb — the TF32 path materializes fp32 copies.
2. Chunk-boundary host-sync point (0.9–1.8 ms gap in mb1–3; **183 ms in mb4**).
3. **Batched phase ×4**: `nvjet_sss_tf32_256x256_NNT` **8.1 ms** back-to-back, pipelined on 2 streams
   (tracks 4+8). 4096×6144×154k tf32 ≈ 7.7 Tflop → 8 ms ≈ ~0.96 Pflop/s: the GEMMs run at speed.
4. CE tail small kernels + 2–3 sub-1.5 ms sync gaps, then layer-40 recompute begins.

Only 2 big GEMM classes ×16 each (no third wgrad-sized GEMM) — consistent with a frozen (LoRA)
LM head: fwd logits + dgrad only.

**Verdict: the ~9-syncs/mb LM-head code costs only ~14 ms/mb of GPU idle in steady state**
(62 D2H copies/mb are tiny and mostly hidden). Total LM-head cost ≈ 0.10–0.12 s/mb ≈ 0.7 %/cycle.
The mb4 183 ms event shows the *exposure* the syncs create (zero CPU runahead → any CPU hiccup lands
straight on the GPU), but LM head is NOT a first-order stage-1 lever at d4/131k.

## Q4 — DSA backward host-sync fingerprint

- The fingerprint exists: inside bwd regions, repeating ≥0.3 ms gaps bracketed by
  `cub DeviceSelectSweep (nonzero) → index_kernel`, `index → DeviceReduce+item`, `Memcpy DtoH → …`
  — but it is small: ~10–15 ms/mb of ≥0.3 ms sync-branded gaps; total bwd idle 0.31–0.34 s/mb,
  dominated by sub-0.1 ms micro-gaps between the ~1800 small kernels per layer-pass.
- The real story is CPU-side: per mb work window (7.5 s wall) the host burns
  **2.2–2.3 s in 4259 cudaStreamSynchronize + 2.3 s in 5428 blocking cudaMemcpyAsync + 0.7 s event syncs
  ≈ 4.9–5.2 s CPU-blocked (≈21 s/step)** — ~35 sync round-trips per layer-pass. The CPU has zero
  runahead; GPU gaps stay small only because kernels are long enough to cover relaunch latency.
- vs the historical 33 s/step DSA-bwd figure (262k cross-node): at 131k/d4 the GPU-side damage is
  ~2.6 s/step of micro-gap idle (fwd+bwd), i.e. **the DSA-bwd host syncs are not a stage-1 bottleneck
  in this config** — but the sync *count* structure is unchanged, so it will re-bite at higher seq or
  faster schedules. (BT_DSA_CP_LAYOUT_CACHE / BT_THD_ROPE_HOST_CACHE attack exactly this class.)
- dsa_bwd main kernels: 160 (40 layers × 4 mb), 3.08 s total, 19.3 ms avg — pure compute, healthy.

## Q5 — stage-1 balance sheet

Steady-state microbatch cycle (avg mb2–4; cycle = park-start → park-start = 15.6/14.1/14.2 s, avg **14.6 s**):

| term | s/mb | % | notes |
|---|---|---|---|
| p2p park (wait for stage-0 activations) | 7.11 | 48.6 | convoy; = stage-0 work time |
| compute-covered wall | 4.74 | 32.4 | fwd 1.29 + lm 0.11 + bwd 3.34 |
| comm-only resident (nothing else running) | 2.09 | 14.3 | a2a-only 0.63 fwd + 1.52 bwd (a2a overlap ≈ 6–8 %) |
| GPU-empty idle | 0.66 | 4.5 | ~0.62 micro-gaps + ~0.04 in >1 ms |
| **cycle** | **14.6** | 100 | work window 7.5 s + park 7.1 s |

Work window split: fwd 2.24 s (compute 1.35, a2a 0.67, AG 0.03, idle 0.31) + LM head 0.12 s +
bwd 5.11 s (compute 3.47, a2a 1.45, RS 0.10, idle 0.33).

Per-step extras (10.7 s total, ≈2.7 s/mb amortized): init 0.25 + mb1 warmup excess ~3.9 +
**tail 7.10** (6.47 s u64 telemetry AllReduce parked ON the compute stream while stage 0 drains its
last backward, + 0.36 s Broadcast + 0.27 s gaps) + **optim 0.65** (652 ms pure CPU stall inside
FusedAdam step; GPU adam work is ~1 ms — LoRA) + post 0.04.
Check: 4 cycles + extras = 69.16 s ✓. EP a2a in-kernel wait (dur above 600 GB/s wire bound):
dispatch 0.80 + probs 1.58 + combine 3.20 = **5.58 s/step = 65 % of a2a residency** (combine skew,
max 156 ms — same expert-imbalance signature as stage 0).

### Largest removable terms, ranked

1. **p2p parks 27.1 s/step (48 % of steady cycle).** Schedule, not network (wire = 31 ms). Break the
   fused send/recv convoy (interleaved schedule, more microbatches, or decoupled isend/irecv) →
   ceiling ≈ cycle → max(work₀, work₁) ≈ 7.5 s/mb, i.e. up to ~1.9× step speedup.
2. **Tail + optimizer ≈ 7.7 s/step.** 6.47 s is drain skew surfaced by a 24-byte telemetry allreduce
   sitting on the compute stream — it shrinks with #1 (less skew at drain) and should be async/on a
   side stream regardless; the 652 ms optimizer CPU stall and 223 ms broadcast-HtoD gap are local
   fixes.
3. **EP a2a exposure ≈ 8.4 s/step unoverlapped residency (5.6 s of it in-kernel wait).** Overlap
   dispatch/combine with expert compute + fix combine imbalance — same lever as stage 0's board.
4. **Micro-gap drag ≈ 2.6 s/step** from ~4300 host syncs/mb (DSA layout/rope/nonzero + LM head) —
   the host-cache PRs address this; needed before any faster schedule can be realized.
5. **mb1 warmup ≈ 3.9 s/step** — only if it recurs beyond step 0 (unverified here).
LM head/loss itself: ~0.4 s/step steady — not a lever at this config.

### Caveats

- One step, one rank; ProfilerStep#0 → mb1 warmup and init may be non-representative of steady steps.
- LM-head region boundaries are operational (cast-copy → first recompute kernel); ±few ms.
- a2a "wait" uses a 600 GB/s single-rail wire bound (same convention as stage-0 analysis); it is an
  estimate of in-kernel straggler time, not a NIC counter measurement.
- Direction of the two TF32 GEMM phases (fwd-logits vs dgrad) inferred from layout/order, not from
  shapes metadata (trace has no `record_shapes`); frozen-LM-head inference is moderate confidence.
- The 6.47 s tail AllReduce parks on stream 7; nothing else was runnable then, so "on compute stream"
  costs nothing *today* — it becomes a real serialization only once the drain skew shrinks.
