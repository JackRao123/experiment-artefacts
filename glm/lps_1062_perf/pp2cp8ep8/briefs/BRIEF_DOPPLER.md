# BRIEF — doppler: PP2/CP8/EP8 @131k box + bring-up lane (gibbs → doppler, 2026-08-12)

You are **doppler**, succeeding **gibbs** (this Mac session) on the
PP2/CP8/EP8 @131k bring-up lane. Orchestrator is **borel** (succeeded
maxwell). Report via `~/.agents/scripts/send-message.sh borel "doppler: ..."`.
Fleet: poincare (packing/parity, ex-volta), hausdorff (export, ex-dedekind).

**Read first, in order**: `../GOAL.md` (mission source), `../HANDOFF.md`
(maxwell→borel orchestrator handoff — the current state + Jack's #1
directive), `../NOTEBOOK.md` (full timestamped log), this file. Then ack borel.

## Succession rule (from borel, via maxwell)

When YOU hand off later: tell borel AND wait for your successor's ack before
going quiet. Same rule I owe borel now.

## Box state (w56lorq, 2×8 B300 ali, ssh `tj-w56lorq` leader / `tj-w56lorq-1`)

- Shared FS: `/root/.cache/user_artifacts/` — clone `trainers_main`, venvs,
  `.devbox_up/` lifecycle scripts, `lps1062_pp2/` (our staged dir: configs,
  tools, traces/, export_test/ evidence).
- Trainer lifecycle ONLY via `.devbox_up/start_trainer.sh` /
  `wait_trainer_health.sh` / `stop_trainer.sh`. Health:
  `curl -sf http://127.0.0.1:8001/health` (leader, port 8001).
- **Venv carries my sequenced cutlass cu13 repair** (libs_base then libs_cu13,
  cu13 wins overlaps) — mainline lock still races (PR 987 unmerged — Jack's
  morning action). Do NOT rebuild the venv; if you must, re-apply the sequence
  (NOTEBOOK 19:0x has the exact commands).
- Box clone may carry the **uncommitted TF32-head patch**
  (`lps1062_pp2/tf32_head_port_21d0c578.patch`, env-gated BT_TF32_LM_HEAD=1) —
  check `git diff` before assuming pristine.

## Launch discipline (hard-won; follow exactly)

1. **Stale-process sweep before every launch** (orphaned workers survive
   stops holding 270+ GB): `srun --overlap -N2 -n2 bash -c 'pkill -f
   "[d]p_worker.main"; sleep 2; pgrep -af "[d]p_worker" || echo clean'`.
   The `[d]` bracket is load-bearing (pkill self-match otherwise kills the
   srun). Runbook: `tools/bringup_pp2cp8ep8.sh` (phases 0-6).
2. **Bounded probes**: virgin-topology anything gets 5-10 min; hung →
   stop_trainer.sh + read ALL ranks' logs (a worker death looks like a leader
   rendezvous timeout) → diagnose → relaunch. Boot weight-load gets 13-20 min
   (warm cache ~8; PP1 config loads FULL model per rank = slower, ~15+).
3. **Pull kineto traces off `/tmp/checkpoints/profiles/torch_trace/`
   IMMEDIATELY after each traced run** — the dir is CLEARED on every
   runtime_profile/start. Archive to `lps1062_pp2/traces/` (box, persistent).
4. **CPFS close-to-open**: scp'd files can read as NULs from a sibling node —
   sha256 per node before trusting (runbook phase 3b).
5. Configs via `BT_TRAINER_CONFIG_PATH` + `BT_TRAINER_SERVER_CONFIG_PATH`
   (source `lps1062_pp2/trainer_env.sh`; configs in `../configs/` Mac-side and
   `lps1062_pp2/` box-side). `max_seq_len` MUST equal the packed buffer
   (131072) — oversizing causes an 11.5× allocator thrash (finding F1).
6. Env for headline runs: `BT_TF32_LM_HEAD=1 BT_PROFILE_RANKS=0,8
   NCCL_IB_QPS_PER_CONNECTION=8 NCCL_IB_SPLIT_DATA_ON_QPS=1
   NCCL_NCHANNELS_PER_NET_PEER=8`. Telemetry gate BT_PEAK_MEM_REPORT exists
   (default on; A/B showed it absorbed-idle, not a lever).
7. B300 reports as "L20D" in nvidia-smi — normal.

## Branch map (all off trainers origin/main @ df831501)

- **`jackrao/lps-1062-pp2cp8ep8`** (MY branch, the bring-up mainline) @
  `73c24b00`: (78,2) PP layout 38/40 + DSA-only CP>1+PP>1 gate exemption
  (21d0c578) → BT_PROFILE_RANKS (ad39a97d) → telemetry gate (760021be) →
  VPP2 layout [18,20,20,20] + vpp config field (61c41d9e) → VPP per-chunk
  iterator fix (8c13ed31) → **M=N merge** (73c24b00, volta's restructure with
  my E2b seam `_schedule_data_iterator` at the call site).
- `jackrao/lps-1062-pp2-packing` @ 8b0ef108 (volta/poincare): pad-to-131k +
  M=N (merged into mine).
- `jackrao/lps-1062-pp2-export` @ ef2e8036 (dedekind/hausdorff): export tests;
  db5d1826 has the BT_SAVE_STATE_SYNC=1 sync-save tooling.
- PR 987 (Jack's, unmerged): cutlass-dsl cu13 overlay race resolution fix.

## Banked results (tonight, all on the merged branch, 131k, LoRA r32)

| config | tok/s/GPU | note |
|---|---|---|
| clean tip d4 | 550 | first light |
| +ship NCCL env | 552 | env ruled out |
| +TF32 head d4 | 589 | anchor-apples-to-apples |
| **+M=N d4** | **918** | 1.42× golden 645 |
| M=N d8 | 1000 | 1M tok/step |
| **M=N d16** | **1052** | 2M tok/step, mfu3x 9.6%, peak 194 GiB |

Root cause of the old wall + the M=N fix: HANDOFF.md + NOTEBOOK.md ~19:0x +
`../MN_SCHEDULE_MEMO.md` (volta). Exonerated-by-measurement list in
HANDOFF.md (do not re-litigate).

## JACK'S #1 NEXT: E2b (overlap compute + comms) — run it without rediscovery

Full spec: `../HANDOFF.md` §"JACK'S ACTIVE DIRECTIVE" + my
`../SCOPING_a2a_overlap.md` (the E2a/E2b ladder). Everything is unlocked:
- M=N gives M=4≥PP=2 → the interleaved schedule constraint is SATISFIED (the
  old E2a M=1 failure is gone).
- `overlap_moe_expert_parallel_comm=True` needs: VPP (field on my branch:
  `virtual_pipeline_parallel_size: 2` in the trainer config → VPP2 layout
  [18,20,20,20] auto-applies, chunk starts 1/19/39/59 verified topk-valid);
  NOT-full recompute (mcore :2633) → use block+K (`recompute: {"granularity":
  "full", "method": "block", "num_layers": K}` — recompute.py block method
  checkpoints only the first K layers/stage; K≈20-24 est. fits ~235-250 GiB,
  d1 ramp measures); the `moe_shared_expert_overlap=False` clearing hunk from
  `lps1062_pp2/pre_checkout_local_changes_0e0b65a6.patch` (first
  megatron_controller hunk) ported into `_configure_moe_provider`
  (megatron_config.py) — mcore rejects the combo otherwise (:2651).
- Validation ladder: d1 memory ramp (abort >255 GiB reserved) → d2 canary
  (loss 12.2-12.4 + gn match) → traced d4 (BT_PROFILE_RANKS=0,8).
- Watch item: CPU-blocked serialization (~21s/step stage-1, grows with M) may
  eat part of the overlap win — check CPU runahead in the traced d4.

## In-flight at handoff (finish these first — see HANDOFF.md §In-flight)

1. **Parity legs B+C** (poincare drives; you own the box lifecycle): PP1/CP8
   (DP2) boot for leg B was loading at my succession (BT_PACK_PAD_TO_MAX=1
   set; leg C needs a SEPARATE boot WITHOUT it — the var is process-env-bound,
   read per-call). Reference config staged:
   `lps1062_pp2/trainer_pp1cp8ep8_131k.json`.
2. **Bounded /save_state probe** (after parity, PP2/CP8 clean-init relaunch):
   `tools/save_state_probe.py` staged on box (10-min bound, auto py-spy all 16
   ranks on hang into `lps1062_pp2/export_test/`). Expect async hang at CP8
   (dedekind's finding); then optionally BT_SAVE_STATE_SYNC=1 at scale.
3. PR shaping + final report = borel/orchestrator lane; your evidence is in
   NOTEBOOK.md + `../results/`.

## Papercuts filed tonight (context)

pc_8cb1f2d3be0c (ty --fix hook strips suppression mid-push), pc_b488a720742c
(cutlass cu13 overlay race → merge PR 987; plausibly the a131 0-for-8).
File new ones with `--tag lps1062`.
