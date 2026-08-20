# BRIEF — gibbs: PP2/CP8/EP8 @131k bring-up lead (2026-08-10 night)

You are **gibbs**, working for **pauli** (manager, this Mac). Reply/report via:
`~/.agents/scripts/send-message.sh pauli "gibbs: <message>"`.
Report on: milestone reached, blocker hit, or anything surprising. Don't go
silent for >30 min while active. Jack is asleep; we work autonomously.

## Mission

Get the GLM-5.2 trainer working at **PP2 / CP8 / EP8 / TP1 / DP1, 131,072
seq len, 2×8 B300**, starting from tip of trainers (`origin/main` @
`df831501`). Then profile it (TPS + MFU). Overall program goal: maximize
MFU and tok/s/GPU for GLM-5.2 131k LoRA training. Full goal text:
`experiment_artefacts/glm/lps_1062_perf/pp2cp8ep8/GOAL.md`.

Why this config: PP is the only cross-node dimension → EP a2a + CP collectives
go over intra-node NVLink (EP a2a was ~50% of step wall cross-node in prior
traces). Per-rank expert bytes are identical to golden EP16/PP1 (half layers ×
double experts cancels), so memory should land near golden (~200 GiB/GPU used,
of 275).

## Your workspace

- Mac worktree: `~/Documents/wt-pp2-bringup`, branch `jackrao/lps-1062-pp2cp8ep8`
  (off origin/main df831501). Commit + push to origin to move code to the box.
  Do NOT touch `~/Documents/trainers` (Jack's checkout; read-only reference).
- Artefacts dir (log everything here):
  `~/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/pp2cp8ep8/`
  — use `configs/`, `results/`, `logs/`. Big traces stay on-box or gitignored.
- Box: 2×8 B300 on ali, provisioning now (pauli will send job id when ready,
  ~30-60 min). Until then do all Mac-side work below.

## Step 1 — PP2 pipeline layout (Mac-side, do first)

File: `server/src/trainers_server/dp_worker/backends/megatron_bridge/glm52_dsa.py`.
Add a `(78, 2)` entry to `_GLM52_DSA_PIPELINE_LAYOUTS`.

Constraint (verified by pauli against mcore + bridge source): GLM-5.2 DSA
shares topk indices in groups of 4 layers with offset = `first_k_dense_replace`
= 3. A layer computes its own topk iff 1-based layer_number ≤ 3 or
`(layer_number − 3) % 4 == 0` (see mcore
`megatron/core/transformer/experimental_attention_variant/dsa.py:104-123`,
`is_dsa_skip_topk_layer` / `source_dsa_compute_layer`; bridge sets the values in
`glm5_bridge.py` ~line 142). So PP stage 2 must START on a layer in
{7, 11, 15, …, 75} (1-based).

Primary: `[_pipeline_stage(38, embedding=True), _pipeline_stage(40, loss=True)]`
(stage 2 starts at layer 39; 39−3=36 ✓). Fallback if stage-1 OOMs or is
badly imbalanced: 42/36. Even 39/39 is INVALID (start 40 → 37 % 4 ≠ 0).
Sanity: existing (78,8), (78,16), (8,2) entries all satisfy the same rule —
check yours the same way. Add/extend a unit test if one covers the layouts
(grep tests for `_GLM52_DSA_PIPELINE_LAYOUTS` / `glm52_dsa`).

## Step 2 — trainer config (Mac-side)

Template (golden 2-node): `experiment_artefacts/glm/lps_1062_perf/runs/
overnight_20260808_context_sweep/trainer_ep16cp16_2node.json`. Ours:

```json
{
  "base_model": "zai-org/GLM-5.2-FP8",
  "max_seq_len": 131072,
  "tensor_parallel_size": 1,
  "pipeline_parallel_size": 2,
  "expert_parallel_size": 8,
  "context_parallel_size": 8,
  "expert_tensor_parallel_size": 1,
  "trust_remote_code": true,
  "attention_backend": "flash",
  "lora_rank": 32,
  "lora_alpha": 32,
  "weight_sync": {"type": "disabled"}
}
```

- `max_seq_len` MUST equal the actual packed-buffer size (131072). Finding F1
  from 08-09: oversizing it causes an 11.5× allocator-thrash slowdown.
- Save to `pp2cp8ep8/configs/trainer_pp2cp8ep8_131k.json` + server config
  (copy `trainer_server.json` from the same runs dir, new trainer_id).
- Check how the server validates/derives parallelism (world_size 16 = 1×8×2×1
  ✓, EP8 divides CP×DP=8 ✓) — grep the config validation path for asserts that
  might reject PP>1 with CP>1 preemptively.

## Step 3 — rank-mapping sanity (Mac-side, cheap but load-bearing)

The perf thesis requires PP to be the CROSS-NODE dim: stage 0 = node 0
(ranks 0-7), stage 1 = node 1 (ranks 8-15), CP/EP groups intra-node. Megatron
default order is `tp-cp-ep-dp-pp` (pp outermost/slowest) which gives exactly
that — but VERIFY in `parallel_state` initialization (and what our server
passes for `order`). Plan to log/dump the process groups at boot on the box
(rank→group listing) and eyeball that CP and EP groups never span nodes.

## Step 4 — PP+CP landmines (read before launching)

- volta (peer, packing stream) is tracing the datum→partition→microbatch path
  and how num_microbatches / p2p tensor shapes work under PP with THD. Their
  memo lands in `pp2cp8ep8/` and via message; read it before first launch.
  For bring-up you sidestep variable shapes: the profile driver sends
  synthetic datums of EXACTLY 131,072 tokens each → every partition/microbatch
  identical size. Same-size microbatches are REQUIRED across PP ranks per call.
- Number of microbatches: start d2 (2×131k datums → 2 microbatches, pipeline
  fill 2/3), then d4 (the 524k tok/step standard operating point, matches the
  notebook anchor table). Memory: stage-0 in-flight = min(m, PP) = 2 microbatch
  checkpoint sets regardless, so d4 costs no more than d2 at steady state.
- Full activation recompute must be ON (it's the default for this model path —
  verify, golden runs used it).
- DP=1 so the F2 DP-deadlock class doesn't apply. But watch for its PP analog:
  hung first collective / ALLTOALL all-enqueued-none-completed = deadlock
  signature, kill within minutes, don't wait.
- `BT_SKIP_WARMUP=1` exists at tip (`backend.py:182`) if the boot warmup path
  itself trips on PP; prefer fixing root cause, use it as a diagnostic lever.
- B300 GPUs report as "L20D" in nvidia-smi — normal.
- Never kill the bench driver mid-operation (orphaned server op poisons the
  next optim_step accounting — restart the trainer if it happens).

## Step 5 — box bring-up (when pauli sends the job id)

devbox-up will have provisioned: shared `trainers_main` clone + venvs at
`/root/.cache/user_artifacts/`, lifecycle scripts in
`/root/.cache/user_artifacts/.devbox_up/` (start_trainer.sh /
wait_trainer_health.sh / stop_trainer.sh), `env.sh` sourced by login shells.

1. `ssh tj-<job>` (leader; workers = `tj-<job>-<rank>`).
2. In the shared clone: `git fetch origin jackrao/lps-1062-pp2cp8ep8 && git
   checkout jackrao/lps-1062-pp2cp8ep8` (+ submodule update if the bridge
   submodule pin moved — you shouldn't need to move it).
3. Check GLM-5.2-FP8 weights present in the HF cache (`env.sh` sets HF_HOME;
   check team_artifacts/huggingface). If absent, start the download FIRST
   (it's the long pole) and tell pauli.
4. Configs onto `/root/.cache/user_artifacts/`, export
   `BT_TRAINER_CONFIG_PATH` + `BT_TRAINER_SERVER_CONFIG_PATH`, verify all
   nodes clean (`srun --overlap ... pgrep/nvidia-smi` check), then
   `bash .devbox_up/start_trainer.sh` and IMMEDIATELY run
   `wait_trainer_health.sh` in the background — it exits every ~3 min with
   diagnostics; inspect every checkpoint. Config/rank failures surface in the
   first 1-2 min; full boot (load+warmup) is ~13-20 min on a warm cache.
5. **Bounded-timeout discipline (Jack's explicit instruction): PP+CP is an
   untested combination — any single probe gets a 5-10 min timeout, checked
   periodically. If GPU util flatlines / a collective hangs / logs stop
   advancing → scancel (stop_trainer.sh), read ALL ranks' logs (a worker
   death looks like a leader rendezvous timeout), diagnose, relaunch.** Boot
   itself (weight load) is allowed its 13-20 min — distinguish "loading" from
   "hung" via log progression + GPU memory ramps.
6. First functional probe: trainer health 200 → status shows world 16 /
   PP2/CP8/EP8 → one `/forward_backward` with 2×131k synthetic datums + one
   `/optim_step` (the profile driver's warmup does exactly this — see below).
   Loss should be ~12.2-12.4 (random-token canary band from prior nights);
   grad_norm finite. Record per-GPU memory (worst GPU + headroom vs 275 GiB).

## Step 6 — profile + measure (the deliverable)

Driver: `experiment_artefacts/glm/lps_1062_perf/tools/profile_driver_new.py`
(runs on the leader against 127.0.0.1:8001; writes
`/root/.cache/user_artifacts/lps1062_bench/<label>.json` + SUMMARY).
It currently sends 1 datum × --seq-len per window ("all windows same shape").
**Extend it with a `--datums N` flag** (datums list is already plumbed through
`drive_window`) so a window = N×131k datums; keep the warmup→traced→control
protocol; headline TPS/MFU comes from the CONTROL window. `mfu.py` sits next
to it (note: its convention reads ~4% high; report both raw and ×0.96 as the
notebook does; B300 dense bf16 peak ≈ 2.25 PF).

Runs, in order:
1. `pp2-131k-d2` — first-light probe + numbers.
2. `pp2-131k-d4` — the headline (524k tok/step, comparable to the anchor
   table: golden EP16/CP16 ≈ 645 tok/s/GPU steady with ship env, ~445 without).
3. If healthy: also capture the kineto traced window (the driver does this) —
   copy the trace to `~/perf_profiles/lps-1062/pp2cp8ep8/` on the Mac if
   <a few GB, else leave on-box and record the path.

Env note: tip of main does NOT include the ship NCCL env defaults commit
(that's `5d6fae0f` on Jack's `jackrao/lps-1062-nccl-env-defaults` branch, 1
ahead of main). For an apples-to-apples "ship env" number, export in the
launch shell: `NCCL_IB_QPS_PER_CONNECTION=8 NCCL_IB_SPLIT_DATA_ON_QPS=1
NCCL_NCHANNELS_PER_NET_PEER=8`. Do bring-up with defaults first (less moving
parts), then re-launch with ship env for the headline number. Expect the env
to matter LESS here (a2a is intra-node now) — that delta is itself a finding
worth recording.

## Reporting

- Append every launch/result/failure to
  `pp2cp8ep8/NOTEBOOK.md` (timestamped, terse, include job ids + config paths).
- Message pauli at: layout+config pushed; trainer healthy; first fb done
  (loss + memory); each profile result (tok/s/GPU, step s, MFU, peak mem);
  any blocker >20 min.
- File papercuts for friction (`papercuts add "..." --tag lps1062`).
