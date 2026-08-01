# B300 rwn24dw incident: overnight A/B runbook (2026-07-29)

Investigation state tracker. Task prompt: ../OVERNIGHT_AB_PROMPT.md

## Resolved facts (log recon, 2026-07-29 07:00Z)

- Topology: world=16 (2x8 B300), TP1/PP1/CP16/EP16/ETP1/DP1, seed 1234,
  LoRA rank=32 alpha=32 scale=1, attention_backend=flash, max_seq_len=262144.
- Image: baseten/trainers-server:trainer-cuda13-sm103-5a4ae4d.
- Model: zai-org/GLM-5.2-FP8 (756 GB), loaded from /b10/loaded_weights.
- Ops execute strictly serialized (single dispatcher queue); packing is
  deterministic, in submission order; DP=1 so no datum->rank sharding.
- Crash 1: 21:38:37.989Z op a369ffe9 (39 datums/1,329,907 tok/18,129..84,173),
  ~24.5s into execution, rank 3 (node 0), surfaced at RoPE .tolist() sync in
  absorbed_mla qkv_up_proj_and_rope_apply; also broadcast_object_list death.
- Crash 2: 22:00:36.262Z op 339c67dd (26/843,999/17,630..63,956), ~33s in,
  rank 0 (node 0), surfaced on EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP NCCL
  watchdog poll. After optim step + checkpoint glm52-bolt-mt-async-5 v1.
- BOTH crashed ops passed on byte-identical retries -> not shape-deterministic;
  likely run-to-run numeric wobble (MoE/fp8 atomics) flipping a marginal
  data-dependent kernel condition.
- ~5 healthy optimizer steps preceded crash 1 (ckpts async-0..4).
- No Xid -> software. No per-datum seqlens in logs. No payload bodies anywhere.
- Original client payloads NOT recoverable (no local client repro; loops
  client runs outside this cluster's Loki).

## Environment

- Devbox: 2x8 B300 via devbox-up (project jrao123-ali). See DEVBOX.md when up.
- Baseline source: 5a4ae4d1d (pyproject pins pristine nvidia-cudnn-frontend==1.26.0).
- Candidate patches (each applied ALONE to baseline venv site-packages):
  1. #814 top-k OOB: server/patches/cudnn-frontend-1.26.0-dsa-indexer-topk-oob.patch (at main)
  2. #396 TMEM WAR: server/patches/cudnn-frontend-1.26.0-dsa-bwd-tmem-war-race.patch (branch jackrao/cudnn-dsa-bwd-edge-fixes, 9c7d15ac7)
  3. empty-topk-row: server/patches/cudnn-frontend-1.26.0-dsa-bwd-empty-topk-row.patch (same branch)
  4. bwd index-bounds: server/patches/cudnn-frontend-1.26.0-dsa-bwd-index-bounds-hardening.patch (same branch)

## Plan

1. [ ] Devbox up; clone trainers @ 5a4ae4d; make server-venv sampler-venv;
       record env manifest (pip freeze, hashes).
2. [ ] Stage GLM-5.2-FP8 (check team_artifacts HF cache first; else HF download).
3. [ ] Trainer config: golden B300 leaf + lora_rank 32; start via .devbox_up.
4. [ ] Baseline replay loop: incident_replay.py cycles (data modes agent/corpus/
       random, rotating seeds) until >=2 IMA crashes. Freeze payloads + hashes.
5. [ ] Add rank-side kernel-input capture (top-k / sparse fwd / bwd / reduce_dKV),
       reproduce again, freeze kernel artifact; reduce to 1-GPU replay.
6. [ ] A/B: Arm A (pristine) >=2 fails on frozen artifact; Arm B one patch at a
       time, 10 consecutive passes incl. cold starts.
7. [ ] Minimal wheel/source PR + image build + prod validation + PR updates.

## Devbox environment notes (qexzp23, 2x8 B300, ali)

- GLM-5.2-FP8 already staged in team HF cache (141/141 shards, 704 GiB,
  snapshot ba978f7d). HF_HOME=/root/.cache/team_artifacts/huggingface.
- Boot failure 1: worker node's CPFS readdir cache went stale after the
  leader's venv rebuild -> `No module named cuda.bindings` on ranks 8-15
  (direct stat worked; ls missed the dir). Fix: bump dir mtimes on leader
  (touch/rm in site-packages/cuda{,/bindings}), verify full worker import
  on BOTH nodes before any launch.
- Boot failures 2-3: first NCCL collective (bootstrap allreduce) died with
  IBV_WC_RETRY_EXC_ERR both directions on mlx5_10. Cause: generated
  run_trainer_node.sh auto-selects the InfiniBand-link-layer HCAs
  (mlx5_10..13), but ali B300 inter-node fabric is RoCE on bonded NICs.
  Prod trainer pods (verified live on rwn2mdw) use NCCL_IB_HCA=mlx5_bond,
  NCCL_IB_GID_INDEX=3, NCCL_SOCKET_IFNAME=eth0, NCCL_IB_MERGE_VFS=0,
  NCCL_NET_PLUGIN=none, NCCL_COLLNET_ENABLE=0, NCCL_SHARP_DISABLE=1.
  Fix: dsa_ab/prod_fabric_env.sh with exactly those values, sourced before
  start_trainer.sh; run_trainer_node.sh edited minimally to honor pre-set
  NCCL_IB_HCA (${NCCL_IB_HCA:-$IB_HCA}). Matching prod fabric config also
  improves incident fidelity.
- All lifecycle/health strictly via .devbox_up scripts (user directive);
  ab_loop.sh ensure_trainer uses wait_trainer_health.sh with a bounded
  8-run budget.

## Forensic find: devbox contains an incident node (2026-07-29 09:06Z)

`b300-1-izksekdp-0001` (devbox worker, tj-qexzp23-1) hosted incident ranks
8-15 (g00r1). Its host dmesg (UTC) shows, at exactly incident crash 2:

    [Jul 28 22:00:30] Xid (PCI:0000:1a:00): 13, Graphics SM Warp Exception
                      on (GPC 3, TPC 2, SM 1): Out Of Range Address
    [Jul 28 22:00:30] Xid 13 Global Exception: Multiple Warp Errors
    [Jul 28 22:00:30] Xid 13 ESR 0x5377b0=0xc00000e ...
    [Jul 28 22:00:30] Xid 43, pid=3638041, name=python

- PCI 0000:1a:00 = GPU index 0 on that node (serial 1320926833397) =
  incident GLOBAL RANK 8, not rank 0: the "rank 0" in trainer logs was just
  the first NCCL watchdog to surface the async IMA. Loki had no Xid because
  host dmesg is outside container logs.
- "Out Of Range Address" warp exception = OOB global-memory access — the
  exact bug class under investigation (candidate-flood OOB writes /
  unbounded backward index gathers).
- No NVRM entries in the 21:38 window: crash 1's faulting GPU was on the
  OTHER incident node (b300-1-3xpzznsc-0018, not in this devbox).
- Any Xid attribution for tonight's runs must be timestamp-fenced after
  2026-07-29 08:00Z (these incident-era lines pre-exist in dmesg).
- Saved: manifests/izksekdp_dmesg_nvrm.txt.

## Attempt log

(append every run: UTC time, arm, commit, wheel ver, artifact sha, result,
first failing rank/kernel, notes)

### KERNEL A/B RESULT (2026-07-29 11:30-11:35Z) — CAUSAL PATCH ISOLATED

Hardware: B300 GPU 0 of b300-1-ana8db87-0004 (itself an incident-class node:
hosted the 4w79o03 crash Jul28 00:35). Same venv base (5a4ae4d, pristine
1.26.0), same frozen artifact topk_call_41.pt (sha256 8aaa196939c70d9c...),
same command; arms = PYTHONPATH shadow of cudnn pkg with exactly one patch
(tree md5s in ab/results.tsv). Artifacts: A1=replay_topk topk_call_41.pt,
A2=repro_cudnn_dsa_indexer_topk_oob.py (synthetic, no customer data).

| arm (one patch alone) | A1 runs | A2 runs | outcome |
|---|---|---|---|
| pristine 5a4ae4d      | 3 (1 cold) | 2 | 5/5 IMA |
| tmem-war (#396)       | 2 | 1 | 3/3 IMA |
| empty-row (#439)      | 2 | 1 | 3/3 IMA |
| bounds hardening      | 2 | 1 | 3/3 IMA |
| **topk-oob (#814)**   | **10 (2 cold)** | **1** | **11/11 PASS**, indices structurally valid (in-bounds) |

Only the #814 indexer top-k OOB-lane patch flips the outcome. 10 consecutive
fixed passes incl. cold-JIT; >=2 baseline failures incl. cold-JIT.

### Mechanism linkage to THIS incident (fingerprint evidence)

- Incident crashes show Xid 13 "Out Of Range Address" + "Multiple Warp
  Errors" + Xid 43, ESR words identical (0x1f81fb60/0x1174) across two
  independent events (worker node 22:00:30 = crash 2; leader node Jul28
  00:35 = same-org deployment 4w79o03 crash). Xid 13/warp-class = SM-local
  (shared-mem) out-of-range — the top-k candidate-flood signature (26
  concurrent invalid __shared__ writes per sanitizer in the #814 analysis).
- E0 differential: the backward index-value faults (#821's oob_hi/oob_neg/
  empty_row) manifest as Xid 31 MMU faults (observed 4x during E0 on this
  box), NOT Xid 13 -> the backward bugs do not match the incident class.
- Glue audit (Megatron-LM d3932e757): all indices reaching sparse attention
  are sanitized (-1/0 tails, in-range) -> backward index bugs unreachable
  from production inputs; consistent with E0 differential.
- Incident class recurrence: >=4 crashes in ~26h across >=3 deployments
  (dq48213 Jul27 22:02, 4w79o03 Jul28 00:35, rwn24dw 21:38 + 22:00), same
  org, same model, same signature, different nodes/GPUs -> software, GLM-5.2
  DSA-specific.
- Score-collapse reachability: synthetic data (natural text / repetition /
  random / chat-template+blobs) never produces scores <= -65504 (35k+ probed
  calls; extreme real_min -337); customer data evidently does (4 natural
  crashes). Exact customer payloads unrecoverable (documented) -> linkage is
  fingerprint-based, not payload-replay-based.

### Natural-scale hammer ladder (2026-07-29 11:20-11:50Z)

14 GPUs x 4 THD partitions of the exact 339c doc lengths (CP16, cp-ranks
2-15) x std ladder 0.02/1/2/4/8, full-indexer fused block: ZERO faults.
Positive control on the same box: B200-proven config (--doc-lengths 55111
--cp-size 32 --cp-rank 4 --std 8.0 --seed 1234) -> CUDA IMA (rc=1), proving
harness ignition capability. Conclusion: the CP16 short-doc geometry is
structurally hard to ignite synthetically; incident required real-data score
distributions. Logs: logs/hammer2/, logs/positive_control.log.

### Production validation (project zq857dq, job wpdvvpw, image 0e0b65a)

- WHEEL_VERSION rank0: 1.26.0+dsatopk1 (verified in prod pod) [11:50Z]
- Pod imageID (kubectl, not control plane): docker.io/baseten/trainers-server@
  sha256:77ecc07c921ea45e9621f9fd2facdcc816c84a9b0f44db26e5d3d8a6549b87e4
  (= tag trainer-cuda13-sm103-0e0b65a), pods baseten-training-job-wpdvvpw-
  multinode-0{,-1}, node e02-sg-e1n4vn65z0b (fresh hardware, not devbox nodes)
- Phase 1: frozen artifact (sha 8aaa1969...) PASS in-image; synthetic repro
  PASS in-image. [~11:52Z]
- Phase 2: 16-rank trainer booted in the image (CP16/EP16); full frozen-shape
  replay cycle (6d85/08ce/a369/339c aggregates) completed with finite losses
  8.247/1.731/2.852/2.720, no IMA/restart/NCCL failure. [12:07Z]
- Job TRAINING_JOB_COMPLETED with VALIDATION_PASS framed JSON. Full log:
  logs/prodval_wpdvvpw.log; launch record: manifests/prodval_launch.json.
- No Billip/user/project overrides were applied at any point (BYOI job spec
  carried the image directly) -> nothing to restore.

## Final status (2026-07-29 12:10Z)

- Causal patch isolated: #814 top-k OOB (dsatopk1). PR #829 description
  updated with full evidence and marked ready for review.
- PR #821 (backward hardening) remains valid defense-in-depth: its three
  fault classes are real on B300 (E0: 3/4 IMA pristine) but produce Xid 31,
  not the incident's Xid 13, and are unreachable from sanitized glue inputs.
- Devbox qexzp23 stopped after artifact sync (artifacts durable: this audit
  dir + CPFS user_artifacts of project jrao123-ali (dq47r1q) +
  team_artifacts/dsa_ab_validation with SHA256SUMS).
- Prod validation job wpdvvpw self-terminated (COMPLETED); no deployments
  or overrides left behind.

### Baseline replay phase 1 (uninstrumented), 2026-07-29 08:18-10:08Z

Arm A env: 5a4ae4d, pristine cudnn-frontend 1.26.0 (md5 b74f18f1), torch
2.11.0+cu130, driver 580.105.08, CP16/EP16 world16, GLM-5.2-FP8 ba978f7d.
All payloads SHA-256-frozen pre-submit (payload dirs deleted for clean
cycles; hashes in client logs).

| cycle | mode | ops | result |
|---|---|---|---|
| 0 | agent | 3 warmup steps (7 ops) | ops clean; harness bug (n=1 sampler assert) killed run at warmup 4 |
| 1 | corpus | crash step (3 ops) + optim | ops clean; harness bug (save_state 422 run_id) |
| 2 | random | crash step (3 ops) + optim | ops clean; same save bug (old code) |
| 3-11 | agent/corpus/random x3 | full crash+post session (4 fb ops + 2 optim + save) each | ALL CLEAN |

~46 crash-shape ops, 0 IMA. CuTe-DSL JIT confirmed active in ranks.
Conclusion: synthetic data (all 3 modes) not reaching the hazard regime at
these seeds -> pivot to BT_DSA_PROBE=1 phase to measure min_slack
(distance to candidate-flood condition, validated against topk_call_41.pt:
detects row 673 at min_slack=-171).
