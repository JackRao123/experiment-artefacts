# LPS-1062 optimization night — notebook

**Date:** 2026-08-07 (overnight session) · **Goal:** maximize GLM-5.2 256k training throughput on 2×8 B300 without OOM or correctness regressions.

Baseline (2026-08-06, devbox q480z53, trainers_main @ 0e0b65a6, golden config TP1/PP1/EP16/CP16, full recompute, alltoall dispatcher):
**446 tok/s/GPU · 73.5 s / 524k-token step · MFU(4×fwd, w/ indexer, 2.5PF) 8.4%.**
Bottlenecks (from `~/perf_profiles/lps-1062/glm52-b300-s256k/REPORT.md`):
NCCL 65% of step (EP a2a SendRecv 59%), 26.7k `aten::nonzero` GPU syncs/step, 8 FP32 SIMT vocab GEMMs (1.85 s), allocator reserved 260/275 GB.

## Measurement protocol (apples-to-apples)

- `bench_driver.py` (this folder): synthetic random tokens seed `0xB300`, **identical rng consumption order to the baseline profile run**: 1×262k-token warmup window, then 2 main windows of 2×262k datums (524,288 tokens/step). Metrics = mean of the 2 main windows.
- Report per iteration: **tok/s/GPU, step time (s), MFU** (two conventions: `mfu3x` = model FLOPs 3×fwd — rewards removing recompute; `hfu` = hardware passes actually run), plus loss/grad_norm per window as the **correctness canary** — must stay ≈ baseline (12.356/0.940 warmup, 12.339/0.941, 12.310/0.693) modulo small reduction-order drift. Fwd FLOPs/token = 118.3 GF (84.3 matmul + 10.5 DSA + 23.6 indexer), see `mfu_calc.py`.
- No kineto/memory profiler in timed runs (profilers only for diagnosis, marked as such).
- **Per-GPU max memory**: every bench runs under `run_bench.sh`, which starts `poll_gpu_mem.sh` (nvidia-smi, 2 s cadence) on every node via srun and folds per-GPU max used MiB into the result json (`aggregates.per_gpu_max_used_mib`). nvidia-smi reports allocator-reserved memory, which is the OOM-relevant number; 2 s sampling can miss sub-second transients.
- Oversized artifacts (traces, pickles) → `~/perf_profiles/lps-1062/opt-night/`; everything else here.

## Lever map (from code reading, trainers @ 5191b710)

| Lever | Config | Attacks | Risk |
|---|---|---|---|
| DeepEP fused dispatch | `moe_token_dispatcher: "flex"` (wheel vendored, cu13) | a2a 59% + nonzero syncs | RoCE/NVSHMEM bring-up; fallback envs: `NVSHMEM_IB_ENABLE_IBGDA=1`, GID 3, `NVSHMEM_HCA_LIST` |
| EP a2a↔compute overlap | `comm_overlap.overlap_moe_expert_parallel_comm: true` (+ optional `delay_wgrad_compute`) | hide a2a latency | **requires `recompute_granularity != "full"`** (bridge validator, comm_overlap.py:493); THD-CP loop forbids only `overlap_grad_reduce` |
| Selective recompute w/ module list | `recompute: {granularity: "selective", modules: [...]}`; allowed: core_attn, moe_act, layernorm, mla_up_proj, mlp, moe, shared_experts | recompute pass replays fwd a2a under "full"; module list can approximate full-recompute memory | OOM at 256k if saved set too big |
| CE/vocab FP32 GEMMs | code path TBD (8×115 ms, grid [1210,32], bias_relu epilogue) | 2.4% of step | needs patch; loss parity check |
| CUDA_DEVICE_MAX_CONNECTIONS | launch env (=1 today) | serialized launch queue kills overlap | was the *mask* for LPS-1003 DSA stream race (fixed in PR#875 wheel +dsatopk5); only touch with loss parity validation |
| moe_expert_capacity pad | fixed-shape routing | nonzero syncs | pads inflate a2a volume — likely net loss; only if syncs persist after flex |

Notes: EP8 (intra-node experts) infeasible — +91 GB/rank expert weights at bf16. `moe_permute_fusion` already on, `moe_grouped_gemm` on, aux-loss off, `moe_router_fusion` hardcoded off.

## Experiment queue (revise as results come in)

1. `exp00-baseline` — golden config rerun on fresh box + current main (commit moved 0e0b65a6 → 5191b710): re-anchor.
2. `exp01-flex` — dispatcher flex, all else baseline.
3. `exp02-selective` — recompute selective `["core_attn","moe","layernorm","mla_up_proj"]` (memory probe; a2a still replayed via "moe").
4. `exp03-overlap` — flex + selective + `overlap_moe_expert_parallel_comm` (+ delay_wgrad if needed).
5. `exp04+` — CE path fix, recompute-list tuning (drop `core_attn` if TE fused attn makes it unnecessary — mcore warns it may be), NCCL tunables, env experiments. Data-driven.

## Iterations

| # | config | tok/s/GPU | step (s) | mfu3x | hfu | max GPU mem (MiB) | loss canary | notes |
|---|---|---:|---:|---:|---:|---:|---|---|
| exp00 | golden baseline (alltoall, full recompute) | **416.5** | 78.7 (82.1/75.2) | 5.9% | 7.9% | (poller missed — srun queued; see exp00b) | 12.3578/12.3401/12.3106 ≈ baseline ✓ | **tonight's anchor.** ~7% below q480z53's 446 — box/fabric variance; windows spread ±5% |

### exp01 — flex/DeepEP dispatcher (attempts, 2026-08-07 ~10:40–11:30 PDT)

- **Attempt 1** (no NVSHMEM env): crash at NVSHMEM IBGDA init — `mlx5dv_devx_obj_modify INIT2RTR_QP syndrome 1ffea3`, `ibgda_rc_init2rtr failed` on every RC; also `cudaHostRegister IoMemory error=800` + `ibgda_alloc_and_map_qp_uar` GPU-handler failures. Root cause: NVSHMEM defaulted to GID 0 (RoCEv1 link-local); fabric needs GID 3 (RoCEv2, routable — same as NCCL_IB_GID_INDEX=3). Fabric GID table confirmed via sysfs.
- **Attempt 2** (`NVSHMEM_IB_ENABLE_IBGDA=1 NVSHMEM_IBGDA_NIC_HANDLER=cpu NVSHMEM_IB_GID_INDEX=3 NVSHMEM_HCA_LIST=mlx5_bond_0:1,mlx5_bond_1:1`): QPs connect, boot reaches first MoE forward in warmup, then **`DeepEP error: timeout (dispatch CPU)` in `internode_dispatch`** + illegal-memory-access cascade on peer ranks. Transport connects but the IBGDA data path over the LAG-bonded RoCE devices (`mlx5_bond_*`) doesn't move data.
- **Attempt 3** (`NVSHMEM_IBGDA_NIC_HANDLER=gpu` + `NVSHMEM_DEBUG=INFO`): same dispatch timeouts (17 DeepEP errors). NVSHMEM logs `IBGDA: device used mlx5_bond_0, data_direct support: 0`.

**Verdict: flex/DeepEP is infeasible on ali B300 (LAG-bonded RoCE) tonight.** QPs connect with GID 3 but IBGDA never moves data over the bond devices, both CPU and GPU NIC handlers. This is an infra/qualification gap (DeepEP+NVSHMEM IBGDA over RDMA LAG bonds), not a trainer-config problem. Follow-up ticket material: qualify DeepEP on the ali fabric or expose non-bonded physical ports to NVSHMEM.

Fallback prepared: `exp03a-overlap-alltoall.json` (selective recompute + `overlap_moe_expert_parallel_comm` — the bridge validator accepts dispatcher `alltoall` too), `exp03b` adds `delay_wgrad_compute`.

### exp00 — baseline re-anchor (2026-08-07 ~01:45)

Box tj-qzlr0o3 inherited the baseline session's shared-FS state: trainers_main @ 0e0b65a6 + LPS-1003 full-footprint-warmup patch, fabric-aware run_trainer_node.sh, GLM-5.2-FP8 HF cache. Trainer boot ~13 min. Loss canaries match the q480z53 baseline to ≤2e-3 → correctness anchor holds. Rank-0 reserved peak 260.2 GB (matches baseline 260.2). Mem-poller srun queued behind the trainer job (fresh srun ≠ --jobid attach) — fixed in run_bench.sh by attaching to the devbox_trainer allocation; exp00b-memprobe (warmup + 1 main window on the hot trainer) captures per-GPU peaks for the baseline config.
