# HANDOFF: MoE dispatcher host-sync elimination (feynman → pascal, 2026-08-09)

You are **pascal** (claude fable 1m), orchestrator. Jack approved this project
this morning. Your predecessor (feynman) orchestrated last night's LPS-1062
parallelism sweep and stopped at 25% context after writing this handoff.
Operators **gibbs, ramanujan, laplace** (kimi k3 1m) are FRESH — cleared, no
memory of last night. Everything they need must come from files or your
briefs. Draft briefs are at the bottom — review, adjust, send.

## Mission (Jack's words: "that aten thing")

Eliminate the **26,684 `aten::nonzero` calls (~20.6s CPU) + ~29k
`cudaStreamSynchronize` (20.0s) + 33.6k `cudaMemcpyAsync` (15.0s) per
524K-token step** in the MoE alltoall dispatcher path (mcore, vendored in
trainers). Since the Aug-7 NCCL fix these host-side split-bookkeeping syncs
are co-dominant with comm (trace: SendRecv 24.5s = 50% of a 48.8s step;
nonzero chain ≈ the other half). This is the deepest remaining MFU lever:
best current numbers are ~745 tok/s/GPU steady (mfu3x ~9.5%) — a working
fix plausibly buys tens of percent.

**Secondary target (optional, only if the same person is already in that
code):** the F2 DP>1 partition-count deadlock fix (all-reduce max partition
count across DP + phantom partitions). Jack scoped the aten work; F2 is a
bonus, not the mission.

**Correctness rule (Jack, standing): no optimization that impacts
correctness.** Patch must be env-gated (e.g. `BT_FUSED_DISPATCH_SPLITS=1`),
parity-tested, and loss-canary-checked (drift >5e-3 vs unpatched = stop).

## Read these first (in order)

1. `../NOTEBOOK.md` — 2026-08-09 section: full results table, findings
   F1–F5, SYNTHESIS with the recommendation + tickets. Then the 2026-08-07
   section for the opt-night that found the nonzero problem.
2. `../REPORT.md` — Aug-7 ship config + "Post-optimization profile" section
   (the exact bottleneck statement).
3. `../overnight/PROTOCOL.md` — last night's ops discipline: reuse its
   structure (provisioning checks, canary rules, reporting format).
4. `../overnight/mfu.py` + `../overnight/mfu_audit.md` — MFU method
   (convention kept; absolute ~4% high).
5. Forensics with exact code line numbers:
   `../overnight/results/B_dp2_boot_deadlock/` and `B_custmix_f2/`
   (token_dispatcher.py:558 all_gather, :959 `_maybe_dtoh_and_synchronize`,
   megatron_controller.py:2411/2471, packing.py:415).

## Ground truth & assets

- **Trace**: `~/perf_profiles/lps-1062/opt-night/exp05d.pt.trace.json`
  (1.05 GB kineto, rank 0, one 48.8s 524K-token step, 256K golden config,
  ship env). Use the `analyze-torch-traces-loops` skill (perfetto
  trace_processor). Baseline (pre-NCCL-fix) profile write-up:
  `~/perf_profiles/lps-1062/glm52-b300-s256k/REPORT.md`.
- **Code**: megatron-core 0.19.0 vendored in trainers. Measured stack =
  trainers `0e0b65a6` + Aug-7 patches (TF32 head `chunked_lm_head.py`,
  overlap-validator `megatron_controller.py`). Check for a local checkout
  around `~/Documents/trainers` (this artefacts dir lives in it); otherwise
  operators read/patch on-box (the Aug-7 flow: patch on box, archive diff to
  `../patches/` or here).
- **Bench kit** (proven last night): `../overnight/bench_driver2.py`,
  `run_bench2.sh`, `../poll_gpu_mem.sh`, `../fold_mem.py`, configs in
  `../overnight/configs/`. Baselines to reproduce before any A/B:
  **A131-131k-d4 = 620 (steady ~645)** on `expA131-ep16cp16-max131k.json`
  and/or **B-131k-d4 = 691 (steady ~745)** on `expB-ep16cp8dp2.json`
  (needs `BT_SKIP_WARMUP=1` + `--warmup-datums 2`, F2 constraints — see
  PROTOCOL + NOTEBOOK F2).
- **Numbers to beat** (steady tok/s/GPU, 524K tok/step, ship env + TF32):
  golden@131K 645 · CP8/DP2@131K 745 · customer-shape 16k-d32 734.

## Hard-won gotchas (cost us hours — don't relearn)

- `devbox-up 2 b300` — the bare command defaults to **B200/hyd** (wrong
  hardware AND fabric). Always pass `b300`.
- Trainer HTTP lands on Slurm node 0 (alphabetical), not always the k8s
  leader — curl :8001/health on every node.
- Shared CPFS (`/root/.cache/user_artifacts`, project jrao123-ali) is
  mutable by ANY box's provisioner: md5-verify every config/kit file
  on-box vs Mac; don't trust `.devbox_up/trainer_srun.log` (cross-box
  truncation); verify env from `/proc/<pid>/environ`, not env.sh.
- **max_seq_len must equal the bench buffer size** (F1: mismatch = up to
  11.5× allocator thrash). For 131K benches boot a max-131072 config.
- DP>1: boot needs `BT_SKIP_WARMUP=1` (pass-1 warmup deadlocks, F2), bench
  needs `--warmup-datums <DP>`, datum counts = multiple of DP, equal
  lengths only (heterogeneous lengths deadlock — F2 confirmed).
- Avoid 65K-homogeneous multi-doc benches on CP8 meshes (F5 pathology,
  ~10× thrash, unrelated to this project) — A/B at 131k-d4, 32k-d16, or
  16k-d32 instead.
- Never kill the bench driver mid-op (orphaned server-side op poisons the
  next optim_step token accounting → restart trainer).
- Ship env on every launch: `NCCL_IB_QPS_PER_CONNECTION=8
  NCCL_IB_SPLIT_DATA_ON_QPS=1 NCCL_NCHANNELS_PER_NET_PEER=8
  BT_TF32_LM_HEAD=1`.
- First main window after boot ~15% slow (compile/allocator settling) —
  steady state = later windows; use r3 for anchors.
- rng-stream canary rule: warmup datum count shifts main-window content;
  keep `--warmup-datums` identical across runs you want loss-comparable.

## Suggested division of labor (feynman's plan, adjust freely)

- **laplace — attribution first (gate for the surgery):** from exp05d
  trace, break the 20.6s nonzero + sync time down by call site / phase
  (dispatch_preprocess vs permutation vs indexer vs elsewhere; per-layer ×
  per-partition counts). Output: which sites dominate, and an upper bound
  on the win (is it 5s or 20s of the 48.8s step?). Later: stream-safety
  review of the patch.
- **ramanujan — implementation:** device-side/batched split bookkeeping in
  the dispatcher (candidate directions: replace per-expert nonzero loops
  with sort/cumsum on device; batch per-layer D2H of splits into one
  transfer; cache/reuse routing metadata across the recompute replay —
  full recompute replays the same routing!). Env-gated, plus a standalone
  parity test (unpatched vs patched split tensors bitwise on random
  routing inputs) like Aug-7's `test_tf32_head_parity.py`.
- **gibbs — validation box:** one fresh 2-node B300 box, reproduce the
  A131 (and optionally expB) baselines with the kit, then A/B the patch:
  same-boot before/after (env toggle needs trainer restart — so
  boot-unpatched → bench → boot-patched → bench, canary both).

Milestones: (1) attribution report → go/no-go + expected win; (2) parity
test green; (3) on-box A/B with canary pass; (4) diff + writeup archived
here (`dispatcher_opt/`), tickets updated. Notebook: append a new dated
section in `../NOTEBOOK.md` — you are the single writer; operators message
you results (format in PROTOCOL.md).

## Orchestration mechanics

- Message operators: `~/.agents/scripts/send-message.sh <name> "..."`
  (gibbs / ramanujan / laplace; you are pascal; Jack's session may also
  message you). Replies arrive as `📨 Message from session <name>` turns.
- Watchdog pattern: `sleep 1800` via Bash `run_in_background: true` —
  re-invokes you when it exits; re-arm each time. Operators go silent
  sometimes; ping after ~30–40 min.
- Capacity: ali B300 had room last night (3 boxes at once). One 2-node box
  suffices for this project. Tear down when done (`truss train stop`),
  verify STOPPED via API.
- Durable facts → memory dir (`~/.claude/projects/-Users-jackrao/memory/`):
  see `lps-1062-customer-regime-sweep.md` and `lps-1062-opt-night-results.md`.

## Draft briefs (send after your own review)

**laplace:** "You are laplace, trace-attribution lead for the LPS-1062
dispatcher host-sync project (orchestrator pascal). Read
~/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/dispatcher_opt/HANDOFF.md
fully, then ../NOTEBOOK.md 08-09 synthesis + ../REPORT.md post-opt profile.
Task 1: using the analyze-torch-traces-loops skill on
~/perf_profiles/lps-1062/opt-night/exp05d.pt.trace.json, attribute the
~20.6s CPU of aten::nonzero (26,684 calls) + 29k cudaStreamSynchronize by
call site/phase (dispatch preprocess vs permutation vs indexer; fwd vs
recompute-replay vs bwd; per-layer/per-partition structure). Deliverable:
markdown report in dispatcher_opt/ + one-line summary to pascal — which
sites dominate and the realistic step-time win bound. This gates the
implementation; be quantitative."

**ramanujan:** "You are ramanujan, implementation lead for the LPS-1062
dispatcher host-sync project (orchestrator pascal). Read
.../dispatcher_opt/HANDOFF.md fully (esp. gotchas + correctness rule), then
NOTEBOOK.md 08-09. Target: the mcore alltoall dispatcher's host-side split
bookkeeping (vendored megatron-core 0.19.0 in trainers; start at
token_dispatcher.py dispatch_preprocess / _maybe_dtoh_and_synchronize —
exact line refs in overnight/results/B_dp2_boot_deadlock/). Wait for
laplace's attribution before committing to a design; meanwhile read the
code and enumerate candidate fixes (device-side sort/cumsum splits, batched
per-layer D2H, routing-metadata reuse across the full-recompute replay).
Patch must be env-gated + standalone parity test. Coordinate diffs through
pascal; archive to dispatcher_opt/."

**gibbs:** "You are gibbs, validation lead for the LPS-1062 dispatcher
host-sync project (orchestrator pascal). Read .../dispatcher_opt/HANDOFF.md
fully (esp. gotchas — devbox-up b300 arg, max_seq_len rule, canary/rng
rules), then overnight/PROTOCOL.md. Task: provision ONE fresh 2-node box
('devbox-up 2 b300'), stage/verify the bench kit per protocol, boot
expA131-ep16cp16-max131k.json with ship env + TF32, reproduce the
A131-131k-d4 baseline (620, steady ~645; loss refs in NOTEBOOK table) —
report to pascal, then hold for the patch A/B."

— feynman, 06:5x 2026-08-09
