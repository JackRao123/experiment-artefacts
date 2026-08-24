# Handoff to weil: fix PP2 offload reload-order bug → offload within 5% of baseline

Jack's order (2026-08-23 morning): fix the activation-offload machinery so
`offload` is within **5%** of `baseline` on the full model — "make sure it's
interleaved properly. Copy bandwidth is not the bottleneck, so it should be
possible. Fix the bug, keep working until it works. Use debug models for
fast iteration where you can. Iterate quickly."

Rehydrate: memory `lps-1062-fullmodel-offload-night` (+ its links), then
this run dir's `NOTEBOOK.md` end-to-end. Method rules are standing Jack
policy: **no guessing** (every claimed mechanism observed or bisected; one
change per A/B), **bounded waits** (every watcher gets a hard deadline),
plain-language updates.

## Arm names (Jack's standing vocabulary)

- `baseline` = no recompute, no offload → **706 tok/s/GPU** @32k/d4 full
  model (the reference; 5% bar = ≥671).
- `offload` = baseline + 4-group offload (core_attn, moe_act, qkv_linear,
  attn_proj) + fix knobs → currently 384-420 (−40%+), peak 232 GiB
  (≈ baseline's 228 — no worst-GPU saving either; secondary mystery).
- `fullrecompute` = 581 @32k / **790 @131k/d4** (786-792 controls) —
  the number to beat on both axes eventually.

## The bug you are fixing (mechanism PROVEN, root cause NOT yet located)

At PP2/d4 (multiple in-flight 1F1B microbatch chunks), the machinery
mispredicts the backward reload order: a **perfectly uniform 12.8%**
(legacy scheduling) / 20.4% (with the prefetch knobs) of ALL reloads
across EVERY group become demand misses — whole-chunk-shaped. Each miss =
reload on the consuming stream + full `current_stream.synchronize()`
(fine_grained_activation_offload.py `tensor_pop`, ~line 1415):
~150-200 misses/step/rank, ~1.8 s/step measured on stage-1's autograd
thread, compounding into 1.5-5.3 s pipeline peer-wait holes. Exonerated by
A/B: the valve (cap 4 vs 16 identical), copy bandwidth (~1 s/direction vs
20 s step), my prefetch knobs (misses persist with knobs off — they only
amplify 12.8→20.4%).

Uniform whole-chunk rate strongly suggests one in-flight chunk (or one
schedule phase — warmup vs steady vs cooldown interleavings differ under
1F1B) consumes in an order that was never recorded / is looked up against
the wrong chunk. Suspect code (VERIFY, do not assume — instrument first):
`_backward_group_order` (recorded per chunk during `is_warmup` in
`on_group_commit_backward`), `pop_backward_chunk`/`front_backward_chunk`/
`_cached_chunks_backward` chunk selection, `flush()` ordering, per-chunk
`is_warmup` lifetime across PP2 chunk reuse, and (my addition, gate to PP1
meanwhile) `on_backward_entry`→`front_backward_chunk`→`top_up_reloads`
possibly targeting the wrong chunk. A cheap first instrument: count misses
PER CHUNK INDEX and per schedule phase (extend `_reload_stats` keying) —
if all misses live in one chunk slot, the mapping bug is localized.

## Fast-iteration vehicle (Jack's explicit ask — use this, not full-model boots)

**PP2 on 2 GPUs with the debug proxy reproduces the regime in ~2-3 min
boots** vs 15-20 min full-model boots: snapshot
`/root/.cache/user_artifacts/glm52-debug-0d4m` (or 0d6m for more layers),
config = proxy config with `pipeline_parallel_size: 2`, EP1/CP1, driver
`--datums 4` (needs ≥2 microbatches for 1F1B; d4 matches the full-model
miss regime). There is even a `glm52-debug-0d2m-pp2` dir in shared
artifacts (unverified — check it). Run on wgm8row with
`CUDA_VISIBLE_DEVICES=0,1`, single node, torchrun nproc=2 (adapt the
proxy-era `run_arm_20260823.sh` in `lps1062_traces/`, or the lps1062_full
overlay with NUM_NODES=1/NUM_GPUS=2 — you must adapt; verify the launch
actually gives PP2). Telemetry (`BT_OFFLOAD_VALVE_TELEMETRY=1`,
`_EVERY=1`) prints per-group prefetch_hits/demand_misses — your fix
signal is **misses → ~0** on the proxy, then confirm on the full model.
FIRST STEP: reproduce the miss storm on the PP2 proxy (if it doesn't
reproduce, the bug needs d4/EP8/CP8 interactions — escalate to 32k full
model per cycle, still fine: ~25 min/cycle).

## Acceptance (in order)

1. PP2 proxy: demand_misses ≈ 0 in telemetry, loss parity vs its own
   baseline within the observed same-config band, no OOM.
2. Full model 32k/d4: offload ≥671 tok/s/GPU (≥8 control windows,
   fresh boot), misses ≈ 0, parity (cross-boot band observed ~3e-3 at
   matched window index — compare against same-index baseline windows),
   AND check peak: the saving should materialize (baseline 228 GiB;
   investigate if offload stays ~230 — memory pickles from the diagnosis
   boot are in `lps1062_full/traces/diag-offload-32k-d4/`).
3. Then 131k/d4 offload vs fullrecompute's 790, with kineto profiles
   (driver records them automatically; `BT_PROFILE_RANKS=0,8` works on
   this tree).
4. If misses hit 0 and the gap is still >5%: the next measured classes are
   `cudaMemcpyAsync` issuance inflation (0.27→5.5-7.2 s traced) and the
   dispatcher `cudaEventSynchronize` class (=layers×d4 exactly) — see
   NOTEBOOK trace findings; also `cudaHostAlloc` pool growth mid-run
   (shape churn — consider pool row-bucket tuning,
   `BT_OFFLOAD_POOL_ROW_BUCKET`).

## Infrastructure (all working, documented in NOTEBOOK + memory)

- Box: `tj-wgm8row` (2×8 B300), ONLY live devbox (others stopped on
  Jack's order). Currently idle, GPUs 0 MiB, no trainer.
- Patched tree: shared-FS worktree
  `/root/.cache/user_artifacts/devboxes/qkpox9w/trainers` (lps1062-scale
  `c226338a` + uncommitted 3-file patch; combined diff saved at
  `runs/debug_proxy_20260821/results/overlap_trace_20260822/bt_offload_backward_prefetch.patch`
  and on box `lps1062_traces/`). Its `.venv` is a real shared dir — works
  from wgm8row directly. EDIT THE MODULE via scp of the whole file (see
  proxy-night flow) — never inline-heredoc python through ssh single
  quotes (that bit us once: mangled quotes → NameError boot failure).
- Full-model launch overlay: `/root/.cache/user_artifacts/lps1062_full/`
  (`start_trainer_full.sh` → `run_trainer_node_full.sh`; arm configs
  `full_*.json`; driver `lps1062_262k/profile_driver_new.py`
  --seq-len/--datums/--control-repeats). Health waits: ONLY the box's
  `wait_trainer_health.sh`, re-run per 180 s checkpoint, reading logs each
  time (box rule embedded in start_trainer.sh). Boots 13-22 min
  (offload arms pay pinned-pool first-touch).
- Slurm gotchas: wait for `squeue` empty before redispatch; killing the
  trainer can leave a node `drained` ("Kill task failed") →
  `scontrol update nodename=<FULL-name, sinfo truncates> state=resume`.
- Baseten-ssh relay 429-throttles bursts → retry-with-backoff (bounded).
- NVTE_CPU_OFFLOAD_V1 must be LAUNCHER env (TE import latch) — the
  overlay sets it; check `env=1 latch=True -> OK` in the trainer log.

## Context you inherit (read before changing anything)

- Proxy-night record (single-GPU mechanisms, all A/B'd):
  `runs/debug_proxy_20260821/results/overlap_trace_20260822/MISSION_WITHIN2PCT.md`
  — engine won't interleave/honor priority; dispatcher sync class;
  ES×offload tail (kept ES ON for full-model arms — memory-tight).
- This night's full record: `runs/fullmodel_offload_20260823/NOTEBOOK.md`
  (arm log, telemetry numbers, stage-1 trace findings, all JSONs/logs/
  traces under json/ logs/ traces/ + box `lps1062_full/traces/`).
- Memory chain: [[lps-1062-fullmodel-offload-night]] →
  [[lps-1062-offload-within-2pct]] → [[lps-1062-offload-overlap-fix]] →
  [[lps-1062-activation-placement]]; rules [[no-guessing-bisect]],
  [[bounded-waits-only]], [[measure-first-methodology]],
  [[status-updates-plain-language]], [[lightweight-review-policy]]
  (ONE fresh-subagent review for box-bound code, no round-trips).

Keep the NOTEBOOK current as you go (Jack's standing instruction: durable
logs survive compaction). Good hunting.
