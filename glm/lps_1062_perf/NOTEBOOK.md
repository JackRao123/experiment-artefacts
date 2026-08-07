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

1. `exp00-baseline` — golden config rerun on fresh box + current main (commit moved 0e0b65a6 → 234d0784): re-anchor.
2. `exp01-flex` — dispatcher flex, all else baseline. **The critical experiment**: at ~47 GB/s effective the a2a is at/near RoCE line rate, so the win must come from moving fewer cross-node bytes (DeepEP NVL-forwarding sends once per node then fans out over NVLink) + killing the host-sync serialization.
3. `exp02-selective` — recompute selective `["core_attn","moe","layernorm","mla_up_proj"]` (memory probe; a2a still replayed via "moe"). Saving dispatcher outputs instead (`moe_act`-style) is a dead end: ~1.6 GB/layer/rank of dispatched tokens = ~120 GB — doesn't fit.
4. `exp03-overlap` — flex + selective + `overlap_moe_expert_parallel_comm`. **Caveat found in code (02:00):** the combined-1F1B schedule only overlaps bwd(mb i) with fwd(mb i+1), and controller.py's THD-CP loop calls the schedule per-partition with `num_microbatches=1` → flag degenerates to no overlap. Real version needs a controller patch grouping equal-length THD partitions into one schedule call (`exp03b`, our synthetic datums are all exactly 256k so grouping is trivial there); also overlap can at best hide min(comm, compute) ≈ 15 s of the 45 s a2a — byte reduction (flex) matters more.
5. `exp04+` — TF32 head patch (built, `patches/tf32-lm-head.patch` + parity test), recompute-list tuning (drop `core_attn` if TE fused attn makes it unnecessary — mcore warns it may be), NCCL tunables. Data-driven.

## Iterations

(filled in as the night progresses)

### 00:06–01:10 — blocked: ali CPFS inode quota (again)

Two devbox-up attempts (jobs `3yl6jeq`, `wd7x1nw`) died ~16 min in: `TRAINING_JOB_FAILED`, empty error_message. Deploy logs show node g00r0 reached RUNNING then `Failed to write cache file /root/.cache/user_artifacts/.baseten-internal/node-0/-1.bin` → exit 74. Exact LPS-1003-era signature: the org fileset `fset-38fdb0874929aac4` (CPFS `bmcpfs-3800206s6bsbnaa52jwuc`) is back at the 1M file-count quota — the Aug-1 escalation (CPFS_QUOTA_ESCALATION.md) was never posted, and q480z53's own venv builds (~180k files) on 08-06 plausibly refilled the cap. Yesterday's successful q480z53 deploy log shows the same benign `python3: not found` lines but a clean cache write — confirms the differentiator. No self-serve API (no cache delete; checkpoints list-only). Per Jack (1am): no Slack escalation, unblock via kubernetes. `~/.kube/ali-apse7-prod-1.yaml` auth works; subagent dispatched to free inodes from Jack's own project dir (dq47r1q: __pycache__/*.pyc first, sampler venv as last resort) via a PVC-mounted pod, with a touch-probe to confirm EDQUOT before deleting.

**01:35 — quota theory falsified; real cause = B300 capacity exhaustion.** Subagent's PVC-mounted pod probes: deep-path create matching the failing shape + 1,000-file create burst all SUCCEED; census counted 1,114,326 files in 183/215 top dirs (already >1M) with creates working → the fileset quota was raised at some point (nothing deleted; net-zero footprint; pod torn down). k8s reality check: all 28 B300 nodes (224 GPUs) taint-reserved (24× charles@parsed.com) and 218/224 GPUs allocated; even Charles's own 2-node job `q0l5p7q` is Pending ("NodePool ResourceNotMatch"). My failed jobs' leaders briefly scheduled, worker never got a node, platform tore down (the cache-write ERROR line was a red herring/secondary). Occupants: Charles 21 nodes (8× 1-node batch jobs started 04:05–05:00 UTC + 9 RL-trainer nodes + 4 inference), teammates 1 node each (jerry 328p1v3, jonathon q8x5e83 — 6 days old, ervin qjleylw) — DCGM shows all three actively used (no idle reclaim), dynamo/kimi 2, fde smoke 1, other-org 1. **Armed a 2-min kubectl watcher** that fires when ≥2 full nodes free (net of pending 8-GPU claims; ≥4 free fires unconditionally) → then instant devbox-up. Meanwhile: off-box work (TF32 LM-head patch, iteration playbook).

- Census detail for the morning (from the cleanup pod, partial): dq47r1q 463,841 files (top consumer, incl. 37,919 regrown pyc), 2qj2rj3 208k, dq450k3 197k, lqzj5xw 83k, 7qkk4lq 56k; team_qzr5p83 uncounted (slow walk). Two `baseten-training-cache-snapshotter` pods stuck ContainerCreating 17–39h on node ap-southeast-7.10.1.77.29 (CPFS "Fail to parse ip address" — CSI/DNS issue worth reporting).
