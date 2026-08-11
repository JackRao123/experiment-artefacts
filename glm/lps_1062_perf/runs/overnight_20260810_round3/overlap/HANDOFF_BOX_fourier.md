# HANDOFF — boxes 318g61w + wxlgv5w (fourier → grothendieck), 2026-08-10 ~08:15 UTC

LPS-1062 round-3, two-box fleet. fourier's session is succeeded by
grothendieck; orchestrator is now **helmholtz** (Claude Fable), succeeding
kepler. This doc is the complete live state; nothing requires fourier's
session. Supersedes/extends HANDOFF_BOX_bohr.md (single-box era).

## 0. The two-box reality (THE critical structural fact)

- **Both boxes SHARE one CPFS filesystem.** `/root/.cache/user_artifacts/`
  is byte-identical and instantly consistent across 318g61w and wxlgv5w
  (sentinel-file proven). There is ONE trainers_main clone, ONE venv, ONE
  lps1062 kit dir, ONE lps1062_bench output dir.
- Consequence 1: **never git-checkout/patch the vendored mcore from box 2's
  lane** — it rewrites box 1's tree under a running arm. (Near-miss 08:00:
  branch-checkout plan would have corrupted the W3 canary tree mid-run.)
- Consequence 2: **per-box output isolation is mandatory.** Box-2 artifacts
  go under `lps1062_bench/wxlgv5w/` and `wxlgv5w/` (see §3). Same-name file
  collisions were tonight's near-miss class.
- Consequence 3: **the timing rule** (kepler, binding): never overlap a
  box-2 BOOT/model-fetch with a box-1 STEADY TIMED WINDOW (and vice versa).
  Boot-time CPFS contention inflates timed windows. Concurrent steady-state
  GPU-bound windows are fine.
- Compute is separate: each box has its own Slurm (`squeue` is per-box).

## 1. The boxes

- **Box 1 = 318g61w** (2×8 B300 ali, "L20D" in nvidia-smi), TIMING lane.
  leader `ssh tj-318g61w` (host b300-1-ana8db87-0004), worker
  `tj-318g61w-1` (b300-1-3xpzznsc-0018). **Trainer HTTP on the WORKER
  :8001** (alphabetical-hostname gotcha). Currently: NO trainer running
  (W3 boot torn down 08:10 UTC, drained clean).
- **Box 2 = wxlgv5w** (same shape), CORRECTNESS lane (then timing lane
  after anchors). leader `ssh tj-wxlgv5w` (b300-1-6hrsfsfh-0021), worker
  `tj-wxlgv5w-1` (b300-1-y4wkfkc4-0009). Slurm job-name
  `devbox_trainer_wxlgv5w` (NEVER `devbox_trainer` — that's box 1's).
- **Teardown whitelist (Jack's standing order, binding):** ONLY these two
  boxes may be stopped/torn down. Nothing else, however idle. Escalate via
  orchestrator → Jack otherwise.
- Housekeeping: `truss train stop --remote baseten --job-id <id>` kills a
  whole box.

## 2. Tree state (the shared vendored mcore @ trainers_main)

`server/vendor/megatron-bridge/3rdparty/Megatron-LM`, base d3932e757,
carries: **B/F/A-v2 + FIX C + verify-instrumentation (hand-placed, bohr §6)
+ W1 + C′ + W3 v2**. All gates default-OFF. W2 is NOT in this tree (v3 was
reversed out after the 0626 gate FAIL; it lives in the branch, §4).
diffstat: +1652/−26 over 8 files + untracked lookahead_checkpoint.py.
- `server/src` is PRISTINE @ 0e0b65a6 (F2 is NOT applied here — box 2 has
  its own copy, §3). Keep it pristine: box 1's A-v3 reboot depends on it.
- W3 is DISARMED for all subsequent arms (kepler, post-verdict: mechanism
  proven, win refuted — +25.8 GiB for ~0 wall, kicks land 99.9% serialized).
- **Branch route:** `jackrao/lps-1062-r3-stack` @ 15d5679e (8 commits,
  basetenlabs/Megatron-LM) carries the whole stack incl. W2 v3 + W3 v2;
  byte-verified vs the Mac ship-tree (4/5 spot files exact + the W3-v2
  call-site fix). F2 lives in trainers branch `jackrao/lps-1062-r3-f2`.

## 3. Box-2 (wxlgv5w) layout — all fourier-built tonight

- `/root/.cache/user_artifacts/mcore_wxlgv5w_r3stack/` — branch clone
  @15d5679e (W2 v3 ARMED-OFF-but-APPLIED for the T2 gate).
- `/root/.cache/user_artifacts/wxlgv5w/server/src` — server/src copy WITH
  f2.patch applied (dry-run-clean, offsets only; .orig cleaned).
- `/root/.cache/user_artifacts/wxlgv5w/devbox_up/start_trainer_wxlgv5w.sh`
  — isolated launcher: own log/state dir, job-name devbox_trainer_wxlgv5w,
  `F2_SHADOW=1` opt-in (PYTHONPATH=wxlgv5w/server/src shadows the venv's
  trainers_server .pth; megatron_core stays on the shared vendored tree via
  its PEP-660 meta-path finder — probe-verified). **Anchors and the W2
  timed arm run WITHOUT F2_SHADOW** (kepler's lane-purity rule).
- `/root/.cache/user_artifacts/lps1062/overlap_design/
  t2_w2_numerics_gate_d88d8b7d.py` + `run_t2_gate_wxlgv5w.sh` — the fixed
  gate (BT_TEST_MCORE_PATH points at the branch clone, SKIP export removed,
  cache-env unset kept). Output → `lps1062_bench/wxlgv5w/`.
- Resolution semantics (probe-verified): PYTHONPATH shadows
  trainers_server; megatron_core is meta-path-pinned to the vendored tree.

## 4. Ladder state + verdicts (all reported)

- **Anchors (box 1, B+F):** 131k-d4 ~705 tok/s/GPU; 16k-d32 ~703.5.
- **ARM 0 inertness:** PASS. **W1 (ARM 2):** mechanism PASS / wall MISS
  (tail-bound box), gate stays armed. **FIX C (ARM 1):** verify caught a
  real bitwise mismatch → PARKED, superseded by C′.
- **C′ (ARM 3): PASS (kepler-ratified).** Verify soak green (6300/6300);
  timed arm 666 tok/s/GPU (131k-d4) / 715 (16k-d32), canaries in band;
  trace rows all PASS (replay eventsync 0/0, allgather 600→300); 2 FAIL
  rows classified A-parked wait-redistribution (bounds re-baselined in the
  Mac checker, md5 befb52b4 — box copy still stale, refresh pending).
  Evidence: ~/perf_profiles/lps-1062/round3/arm-cprime-318g61w/.
- **W3 (ARM 4): mechanism-proven, win REFUTED.** Canary in band
  (warmup0 +0.0001), watch-list green (sweeps=0, fallbacks=0,
  kicks==hits==78/mb by minkowski's corrected 79-chunks/mb reading — the
  window lines are CUMULATIVE at 300-event boundaries, a named telemetry
  trap), memory +25.4 GiB PLATEAU (11.5 intrinsic + ~14 allocator
  retention, minkowski's snapshot decomposition). Traces + evidence:
  ~/perf_profiles/lps-1062/round3/arm-w3-318g61w/ (incl. mem_snapshot/).
- **W2:** v3 gate story CLOSED — 0626 FAIL was two harness artifacts
  (minkowski-proven, empirically sealed): T2 re-run on box 2 with gate
  d88d8b7d = **ALL PASS** (every case incl. input grads + imbalance +
  zero_peer_group; fixc asserts green). Log:
  lps1062_bench/wxlgv5w/t2_w2_gate_rerun_d88d8b7d.log. Fidelity caveat:
  that run was at DEFAULTS (hidden 2048, seq 8192); a BT_TEST_HIDDEN=6144
  confirmation run is QUEUED for a box-2 idle window (minkowski's ask).
  **W2 timed arm on box 2 is unblocked** (after F2 + box-2 anchors).
- **F2 (DP2 deadlock fix):** IN FLIGHT on box 2 — see §5.
- **A-v3 (ARM 5):** box 1's next arm. Staged+md5'd at lps1062/fixa_v3/
  (2f50c182). Stack: B/F+W1+C′, NO W3. Tempered expectations (~0 wall is
  publishable per the absorbed-wait finding).

## 5. In-flight RIGHT NOW: F2 on box 2

- Boot dispatched ~08:15 UTC: expB-ep16cp8dp2.json (EP16/CP8/DP2), ship env
  + B/F, levers OFF, F2_SHADOW=1, BT_SKIP_WARMUP unset (default warmup = 1
  datum = the deadlock trigger pre-fix). Log:
  /root/.cache/user_artifacts/wxlgv5w/devbox_up/trainer_srun.log.
- **NEXT: recipe (a)** — server must reach READY/health (pre-fix hangs
  >35 min in warmup pass-1). Then **kill-switch control (a3):** reboot with
  `BT_F2_PHANTOM_PARTITIONS=0` ⇒ hang MUST reproduce — TIME-BOX ~10 min:
  py-spy signature (node0 token_dispatcher all_gather, node1
  finalize_model_grads) then kill, don't burn 35 min. Then reboot with the
  flag on for (b) custmix / (c) canary vs DP1 golden / (d) perf ≈745.
- Recipe + fibonacci additions (NaN×0 canary, finalize-once): 
  ~/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/overnight_20260810_round3/f2/
  ONBOX_VALIDATION_F2.md. f2.patch md5 d10cd35a (sweep-verified).
- On-box unit sanity already run: 12 pass / 2 skip / 1 fail
  (test_max_over_real_gloo_group = spawn-can't-pickle-locals harness
  artifact, NOT an F2 defect; Mac gate was green per bohr's package).

## 6. House procedures (bohr's §5 stands, plus tonight's additions)

- Env truth: /proc/<pid>/environ, never env.sh.
- Waiter false-positive: wait_trainer_health.sh declares death in the
  post-stop GPU-drain gap; verify with `pgrep -fc "[d]p_worker.main"`.
- Persistent tee on box 1 appends .devbox_up/trainer_srun.log →
  lps1062_bench/trainer_srun_persistent.log. Box-2 logs go to
  wxlgv5w/devbox_up/ (no tee collision by construction).
- md5 EVERY relay-scp'd file from the consumer side (NUL trap is real;
  one silent scp failure hit tonight — retry + md5 caught it).
- Bench: run_bench2.sh from the HTTP node; canaries valid only with
  matched warmup-datums; band ≤2e-3 pass / >5e-3 stop; cross-boot bitwise
  unattainable (floor ~1e-3). Re-baseline vs the box's OWN anchors only.
- Trace capture: profile_window_131k_d4.py; NEVER window-1 captures;
  traces land per-rank on each node's /tmp/checkpoints/profiles/torch_trace/.
- Canary reference provenance: name the exact json; warmup_datums must
  match (a null-warmup_datums repro1-w1 file exists — do not use it).
- Patch surgery: git apply --check first; fuzz only with a documented
  reason (the instrumentation hand-fix makes W2-family patches fuzz-apply
  at the __slots__ hunk); verify with markers + reverse-check invariants +
  diffstat; preserve .orig under lps1062_bench/w2_surgery_provenance/.
- Failure protocol: any FAIL → stop the lever, capture trace+logs, message
  the orchestrator. Never stack a second lever on an un-passed first.

## 7. Messaging

`~/.agents/scripts/send-message.sh <session> "..."`. Report to
**helmholtz** (orchestrator). boltzmann = verification (wants un-truncated
window-line dicts + md5'd traces); minkowski = W2/W3 design owner;
helmholtz also owns the W2-backward thread remnants. My session = fourier.
