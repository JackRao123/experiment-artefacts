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
| exp00b | same, hot trainer, 1 window (mem probe) | 445 | 73.6 | 6.3% | 8.4% | **263,009 / 223,889** (max/min) | informational (3 optim steps applied) | worst-GPU headroom only ~12 GiB |
| exp01 | flex/DeepEP | — | — | — | — | — | — | **infeasible on fabric** (see below) |
| exp03 | EP a2a overlap (+selective recompute) | — | — | — | — | — | — | **infeasible: validator × memory** (see below) |
| exp04 | TF32 CE head (v1 patch) | 417 | 78.7 | 5.9% | 7.9% | 263,609/223,649 | drift ≤5e-4 ✓ | no-op — head isn't LoRA-wrapped (attn-only LoRA), unpatched branch ran |
| exp04b | TF32 CE head (v2, verified active) | **431** | 76.0 (80.1/71.9) | 6.1% | 8.2% | 263,989/223,729 | drift ≤1.4e-3 ✓ | **+3.5% vs anchor**, both windows faster; keep on |
| exp05a | + `NCCL_IB_QPS_PER_CONNECTION=4` `NCCL_IB_SPLIT_DATA_ON_QPS=1` | **464** | 70.6 (72.8/68.3) | 6.6% | 8.8% | 262,849/223,549 | drift ≤1.5e-3 ✓ | **+11.4% vs anchor.** LAG bonds hash flows per-QP — multiple QPs spread each peer connection across bond slaves. GDR confirmed enabled; 6×400 Gb bonds/node; NCCL 2.28.9 |
| exp05b | + QPS=8, `NCCL_NCHANNELS_PER_NET_PEER=4` | **559** | 58.6 (63.3/53.9) | 7.9% | 10.6% | 263,297/224,237 | drift ≤1.4e-3 (mains) ✓ | **+34% vs anchor** — fabric lever is rich; probing ceiling |
| exp05c | + `NCCL_NCHANNELS_PER_NET_PEER=8` | **583** | 56.2 (61.5/50.8) | 8.3% | 11.1% | 262,553/224,513 | drift ≤8e-4 ✓ | **+40% vs anchor**; channel scaling flattening (+4% for 4→8) |

### exp01 — flex/DeepEP dispatcher (attempts, 2026-08-07 ~10:40–11:30 PDT)

- **Attempt 1** (no NVSHMEM env): crash at NVSHMEM IBGDA init — `mlx5dv_devx_obj_modify INIT2RTR_QP syndrome 1ffea3`, `ibgda_rc_init2rtr failed` on every RC; also `cudaHostRegister IoMemory error=800` + `ibgda_alloc_and_map_qp_uar` GPU-handler failures. Root cause: NVSHMEM defaulted to GID 0 (RoCEv1 link-local); fabric needs GID 3 (RoCEv2, routable — same as NCCL_IB_GID_INDEX=3). Fabric GID table confirmed via sysfs.
- **Attempt 2** (`NVSHMEM_IB_ENABLE_IBGDA=1 NVSHMEM_IBGDA_NIC_HANDLER=cpu NVSHMEM_IB_GID_INDEX=3 NVSHMEM_HCA_LIST=mlx5_bond_0:1,mlx5_bond_1:1`): QPs connect, boot reaches first MoE forward in warmup, then **`DeepEP error: timeout (dispatch CPU)` in `internode_dispatch`** + illegal-memory-access cascade on peer ranks. Transport connects but the IBGDA data path over the LAG-bonded RoCE devices (`mlx5_bond_*`) doesn't move data.
- **Attempt 3** (`NVSHMEM_IBGDA_NIC_HANDLER=gpu` + `NVSHMEM_DEBUG=INFO`): same dispatch timeouts (17 DeepEP errors). NVSHMEM logs `IBGDA: device used mlx5_bond_0, data_direct support: 0`.

**Verdict: flex/DeepEP is infeasible on ali B300 (LAG-bonded RoCE) tonight.** QPs connect with GID 3 but IBGDA never moves data over the bond devices, both CPU and GPU NIC handlers. This is an infra/qualification gap (DeepEP+NVSHMEM IBGDA over RDMA LAG bonds), not a trainer-config problem. Follow-up ticket material: qualify DeepEP on the ali fabric or expose non-bonded physical ports to NVSHMEM.

Fallback prepared: `exp03a-overlap-alltoall.json` (selective recompute + `overlap_moe_expert_parallel_comm` — the bridge validator accepts dispatcher `alltoall` too), `exp03b` adds `delay_wgrad_compute`.

### exp03 — EP a2a overlap: structurally infeasible at 256k (2026-08-07 ~11:45)

Two boots, two mcore validator walls, then arithmetic kills it:
1. `disable moe_shared_expert_overlap when enabling overlap_moe_expert_parallel_comm` — GLM provider defaults it on; only the flex branch cleared it. **Patched** on-box (megatron_controller.py `_configure_moe_provider`: clear it under the overlap flag too) — keep for PR regardless.
2. `disable moe in recompute_modules when enabling overlap_moe_expert_parallel_comm` — the overlap schedule can't run under a checkpointed MoE replay. But *not* recomputing MoE means saving the expanded-token activations: ~262k×topk8/EP16 rows × 6144 h × 2 B ≈ 1.6 GB dispatch output + ~2.7 GB expert GEMM in/out per layer per rank ⇒ ~4-6 GB × 75 layers ≈ **300+ GB/rank** vs a 12 GiB worst-GPU headroom. Same arithmetic kills every selective-recompute variant that leaves MoE (or even just layernorms, ~31 GB) resident.

**Verdict: EP-overlap and all lighter-recompute configs are memory-infeasible at 256k on 275 GB parts with EP16/CP16. Full recompute stands.** Re-ranked queue: TF32 CE head (exp04), NCCL alltoall transport tuning (exp05).

### Ops incident — phantom trainer start (2026-08-07 20:21Z)

A trainer srun using MY `lps1062/ctl/run_trainer_node.sh` was submitted at 20:21:45Z by an unknown caller (not my launch tasks; coincided with a shared-FS `env.sh` regeneration at 20:19:26Z by another session's 4-node provisioning). Its env (NCCL knobs) would have been whatever the caller had — untrusted. Mitigations: caller-audit logging + `BT_LPS1062_LAUNCH=1` guard added to `ctl/start_trainer.sh`; job killed, exp05d relaunched deliberately. Result integrity: exp05b/c scored 559/583 vs 464 for correct-env exp05a and ~430 for no-NCCL-env — a default-env trainer can't produce those numbers, so their env was intact (pending sruns keep dispatch-time env). No benched result came from the phantom trainer.

### exp00 — baseline re-anchor (2026-08-07 ~01:45)

Box tj-qzlr0o3 inherited the baseline session's shared-FS state: trainers_main @ 0e0b65a6 + LPS-1003 full-footprint-warmup patch, fabric-aware run_trainer_node.sh, GLM-5.2-FP8 HF cache. Trainer boot ~13 min. Loss canaries match the q480z53 baseline to ≤2e-3 → correctness anchor holds. Rank-0 reserved peak 260.2 GB (matches baseline 260.2). Mem-poller srun queued behind the trainer job (fresh srun ≠ --jobid attach) — fixed in run_bench.sh by attaching to the devbox_trainer allocation; exp00b-memprobe (warmup + 1 main window on the hot trainer) captures per-GPU peaks for the baseline config.
