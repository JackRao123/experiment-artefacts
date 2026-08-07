# LPS-1062 — GLM-5.2 256k throughput optimization (2×8 B300)

**Date:** 2026-08-07 · **Box:** tj-qzlr0o3 (2×8 B300, ali RoCE) · **Code:** trainers_main @ 0e0b65a6 (+ LPS-1003 warmup patch, + tonight's TF32-head and shared-expert-overlap patches) · **Model:** zai-org/GLM-5.2-FP8, golden config TP1/PP1/EP16/CP16, 256k, attention-only LoRA r32, full recompute.

## Result

| | tok/s/GPU | step (524k tok) | MFU (3×fwd / 2.5 PF) | worst-GPU peak |
|---|---:|---:|---:|---:|
| Baseline (exp00, this box) | 416.5 | 78.7 s | 5.9% | 263.0 GiB / 268.6 cap |
| **Ship config (exp06, 3-window soak)** | **629 (steady-state ~660)** | **52.1 s** | **8.9%** | **263.8 GiB (11 GiB headroom)** |
| Max-perf variant (exp05e) | 603 (2-window) | 54.4 s | 8.6% | 266.9 GiB |

**+51% throughput (steady-state +58%), memory flat vs baseline, loss canaries ≤2e-3 (run-to-run noise) on every experiment.** Note the first window after boot is consistently ~15% slower than steady state (allocator/autotune settling) — 2-window benches under-report; the soak's windows 2-3 agree at ~660 tok/s/GPU.

**Ship config = golden config + 2 changes** (both env/launch-level, no trainer-config change):
1. `BT_TF32_LM_HEAD=1` — GLM fp32 LM head projected via TF32 tensor cores (patch in `chunked_lm_head.py`, env-gated). Numerically equivalent: inputs are exact bf16 upcasts (TF32's 10-bit mantissa holds them exactly), accumulation stays fp32; only summation order differs. Verified: canary loss drift ≤1.5e-3 (run-to-run noise level), FP32 SIMT vocab GEMMs gone from trace.
2. `NCCL_IB_QPS_PER_CONNECTION=8 NCCL_IB_SPLIT_DATA_ON_QPS=1 NCCL_NCHANNELS_PER_NET_PEER=8` — the EP alltoall runs as NCCL SendRecv over LAG-bonded RoCE (6×400 Gb bonds/node); bonds hash flows per-QP, so the default 1 QP/connection pinned each peer flow to one bond slave. Spreading over 8 QPs × 8 channels/peer halved per-call a2a latency (p50 34→17.8 ms).

A more aggressive variant (16 channels/peer + `NCCL_MAX_NCHANNELS=64`) reaches ~600 tok/s/GPU but spends ~3.5 GiB more of the worst-GPU OOM headroom (8.9 GiB left vs 12.5 GiB) — not recommended for the ticket's no-OOM-at-256k requirement, available if throughput is ever prioritized over margin.

## What was ruled out (evidence in NOTEBOOK.md)

- **DeepEP (`moe_token_dispatcher: flex`)**: NVSHMEM IBGDA cannot move data over the LAG-bonded RoCE devices on ali B300 (QPs connect with GID 3, dispatch times out; both CPU and GPU NIC handlers; `data_direct support: 0`). Infra qualification gap — follow-up ticket.
- **EP a2a↔compute overlap (`overlap_moe_expert_parallel_comm`)**: mcore forbids it under full recompute AND under `"moe"`-recomputing selective lists; not recomputing MoE requires ~300 GB/rank of expanded-token activations vs ≤12 GiB headroom. Structurally impossible at 256k EP16/CP16 on 275 GB parts.
- **Lighter recompute lists**: same arithmetic (even layernorm-only-saved = ~31 GB > headroom). Full recompute stands.
- **EP8 (intra-node experts)**: +91 GB/rank expert weights at bf16 — doesn't fit.
- `NCCL_IB_SPLIT_DATA_ON_QPS` 0 vs 1: immaterial.

## Post-optimization profile (kineto, one 48.8 s step, exp05d env)

- SendRecv 24.5 s = 50% of wall (baseline 44.9 s = 59%); AllGather+RS+AR 1.6 s.
- **`aten::nonzero` 26,684 calls / 20.6 s CPU + 29k `cudaStreamSynchronize` — unchanged from baseline and now co-dominant.** The MoE alltoall dispatcher's host-side split bookkeeping (per-layer `nonzero`/`tolist`/`item` D2H syncs) serializes the launch pipeline; it is why halving per-call a2a latency didn't halve NCCL wall time. **Top remaining lever → follow-up: device-side/batched split computation in the mcore dispatcher, or DeepEP once the fabric supports it.**
- Trace: `~/perf_profiles/lps-1062/opt-night/exp05d.pt.trace.json` (1.05 GB, local only).

## Follow-up tickets to file

1. Qualify DeepEP/NVSHMEM IBGDA on ali B300 LAG-bonded RoCE (or expose non-bonded ports).
2. Eliminate MoE dispatcher host syncs (26.7k `nonzero`/step) in megatron-core.
3. Upstream the two small patches: TF32 frozen LM head (env-gated) in `chunked_lm_head.py`; clear `moe_shared_expert_overlap` when `overlap_moe_expert_parallel_comm` is set (validator trap).
4. Default the NCCL QP/channel env for ali-B300 multinode trainer pods (prod launch env, not just devbox).

## Reproduction

- Benchmark protocol + all per-experiment JSONs: `bench_driver.py`, `run_bench.sh` (per-GPU max-mem polling), results in `lps1062_bench/` on the box; canonical copies in this folder as `results/`.
- Trainer configs: `configs/`. Launch env per experiment: NOTEBOOK.md table.
