# HANDOFF — box 318g61w (bohr → fourier), 2026-08-10 ~04:30 UTC

LPS-1062 round-3 ladder, mid-flight. bohr's session is being succeeded by
fourier. This doc is the complete live state; nothing here requires bohr's
session to still be alive.

## 1. The box

- **Job 318g61w**, 2×8 B300 (report as `L20D` in nvidia-smi — rebadge, they
  are Blackwell Ultra), project `jrao123-ali`, team Parsed. Provisioned by
  `~/.local/bin/devbox-up 2 b300` at 2026-08-10 00:29 UTC.
- **SSH:** leader `ssh tj-318g61w` (= training-job-318g61w-0, host
  b300-1-ana8db87-0004); worker `ssh tj-318g61w-1` (= training-job-318g61w-1,
  host b300-1-3xpzznsc-0018).
- **TRAINER HTTP SERVER IS ON THE WORKER (`tj-318g61w-1`, :8001)** — Slurm's
  node 0 is the ALPHABETICAL first host (3xpzznsc < ana8db87), not the k8s
  leader. Run bench drivers / profile captures there. (Protocol gotcha #4.)
- **Slurm:** the trainer runs as job-name `devbox_trainer` via srun. Current
  job id 27 (C-prime timed arm, booting). `squeue` to see it.
- **Stop:** `bash /root/.cache/user_artifacts/.devbox_up/stop_trainer.sh`
  (scancel by name). **Start:** see the env blocks below.
- **Housekeeping:** `truss train stop --remote baseten --job-id 318g61w`
  kills the whole box.

## 2. Vendored tree state (the Mac integration tree)

`/root/.cache/user_artifacts/trainers_main` @ `0e0b65a6` (shared CPFS clone,
persists across boxes). Vendored mcore at
`server/vendor/megatron-bridge/3rdparty/Megatron-LM` carries the FULL patch
stack, all gates default-OFF (inert until env-armed):

| layer | patch | gate env |
|---|---|---|
| B/F/A (v2) | `box-applied-qr4ggv3-v2.patch` (was already byte-identical in the clone) | BT_DSA_CP_LAYOUT_CACHE, BT_THD_ROPE_HOST_CACHE (B/F, ON tonight); BT_DSA_BWD_ASYNC_NONEMPTY (A, PARKED/off) |
| FIX C | `fixc.patch` | BT_MOE_DISPATCH_REPLAY_CACHE |
| FIX C verify-instrumentation | `fixc_verify_instrumentation.patch` (fuzz-applied; **hunk 3 was hand-fixed on-box** — see §6) | (verify-mode logging) |
| W1 | `w1-probs-a2a.patch` | BT_MOE_PROBS_A2A_COMM |
| W2 | `w2-moe-a2a-pipeline.patch` | BT_MOE_A2A_PIPELINE |
| FIX C-prime | `fixc_prime_routing_force.patch` (router.py) | BT_MOE_ROUTING_REPLAY_FORCE |

`git diff --stat` on the vendored tree: **10 files, +2349/−25** (router.py
+259 is C-prime; token_dispatcher.py +1283 is C+W1+W2+instrumentation).

**server/src is PRISTINE @ 0e0b65a6** — the F2 patch (§7) must NOT be applied
until the F2 boots (it changes the controller; keep the ladder's tree clean).

## 3. Staged files (all md5-verified Mac↔box tonight)

- Kit root: `/root/.cache/user_artifacts/lps1062/` — bench_driver.py,
  bench_driver2.py (**patched tonight to record `warmup_datums` in the JSON**),
  bench_driver2c.py, run_bench.sh, run_bench2.sh, run_bench2c.sh,
  poll_gpu_mem.sh, fold_mem.py, profile_window_131k_d4.py,
  box-applied-qr4ggv3-v2.patch, **f2.patch** (the F2 fix, applies clean,
  check-only verified), configs/ (expA131-ep16cp16-max131k.json = golden;
  expB-ep16cp8dp2.json = the F2 DP2 mesh; server-config.json; more).
- `lps1062/fixc/` — all FIX C / C-prime files incl. check_acceptance.py,
  probe_gradmode_divergence.py (+ my box-local renamed copy
  probe_gradmode_divergence_boxlocal.py — see §6).
- `lps1062/overlap_design/` — W1/W2/W3 patches + notes + tests,
  t2_w2_numerics_gate.py, **run_t2_gate.sh** (my 2-node torchrun launcher).

Mac sources: `~/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/`
(dispatcher_opt/ and overlap_design/).

## 4. Launch env (the ship recipe — set on the ssh line before start_trainer.sh)

Base ship env (EVERY boot): `NCCL_IB_QPS_PER_CONNECTION=8
NCCL_IB_SPLIT_DATA_ON_QPS=1 NCCL_NCHANNELS_PER_NET_PEER=8 BT_TF32_LM_HEAD=1`
plus B/F gates `BT_DSA_CP_LAYOUT_CACHE=1 BT_THD_ROPE_HOST_CACHE=1`. FIX A
stays UNSET. Launch:
```
BT_TRAINER_CONFIG_PATH=/root/.cache/user_artifacts/lps1062/configs/<cfg>.json \
BT_TRAINER_SERVER_CONFIG_PATH=/root/.cache/user_artifacts/lps1062/configs/server-config.json \
bash /root/.cache/user_artifacts/.devbox_up/start_trainer.sh --num-nodes 2
```
Then arm the lever gates per arm (e.g. W1: `BT_MOE_PROBS_A2A_COMM=1`; C-prime:
`BT_MOE_DISPATCH_REPLAY_CACHE=1 BT_MOE_ROUTING_REPLAY_FORCE=1`; verify soaks
add `BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY=1`). Add `BT_PROFILE_RANKS=0,8` on
any boot where a trace capture is wanted (profiler is inert until the
/runtime_profile/start API call — timed runs stay unprofiled).

**CDMC note (my provenance catch, fibonacci-ratified):** this box runs
`CUDA_DEVICE_MAX_CONNECTIONS` UNSET (CUDA default 8) — verified via
/proc/<dp_worker>/environ. The "devbox runs CDMC=1" framing was stale
(qr4ggv3 also ran unset). A CDMC=1 boot is LOW priority (skip unless slack).

## 5. House procedures that matter tonight

- **Env truth:** always `/proc/<pid>/environ`, never env.sh (F3 hazard).
- **Gate telemetry:** grep trainer_srun.log for the per-gate ACTIVE/DISABLED
  lines — expect 16 (all ranks). Reference strings:
  `runs/overnight_20260809_dispatcher_hostsync/logs/gate-telemetry-qr4ggv3.txt` + the W1/W2/C-prime notes.
- **Bench:** `run_bench2.sh <label> --seq-len S --num-gpus 16 --datums D
  --warmup-datums 2 --repeats N --canary-json <ref.json>` from the HTTP node.
  Results → `/root/.cache/user_artifacts/lps1062_bench/<label>.json`.
  Synthetic data is seed-fixed 0xB300 → cross-boot canaries valid IF
  warmup-datums match (the rng stream aligns). House band: ≤2e-3 pass,
  >5e-3 stop. Cross-boot bitwise is UNATTAINABLE (boot-time kernel selection;
  floor ~1e-3, demonstrated) — fibonacci's ruling: bitwise bars are
  IN-PROCESS only.
- **Re-baseline rule:** all deltas vs THIS box's own anchors
  (A131-131k-d4-r3anchor-318g61w: steady ~705 tok/s/GPU, step ~46.5s;
  A131-16k-d32-r3anchor-318g61w: ~703.5 / 46.7s), never qr4ggv3 refs.
- **Trace capture:** `profile_window_131k_d4.py` on the HTTP node (drives one
  steady 4×131k step between /runtime_profile start/stop). Traces land in
  `/tmp/checkpoints/profiles/torch_trace/` on the rank's OWN node (rank 0 =
  worker tj-318g61w-1; rank 8 = leader tj-318g61w). scp both; md5-verify
  against box (relay-scp NUL gotcha is real — always md5 from the consumer).
  Analysis: check_acceptance.py needs a venv with `perfetto pandas numpy`
  (I used a throwaway venv; trace_processor binary is at ~/bin/trace_processor,
  Perfetto v57.2).
- **Never stack a second lever on an un-passed first** (failure protocol:
  any FAIL → stop the lever, capture trace+logs, message the orchestrator).

## 6. Gotchas hit tonight (read before touching the box)

- **WAITER FALSE-POSITIVE (papercut pc_5713f9b6d77d):**
  wait_trainer_health.sh pgreps `dp_worker.main` and exits "TRAINER PROCESS
  DIED" on the FIRST miss. After a stop, the old trainer's ~200 GiB takes
  minutes to drain, so the next boot's workers spawn ~10 min after srun
  start — the waiter declares death in that gap and exits. BOTH of tonight's
  "deaths" were this; the trainers were fine. Treat an early death report as
  suspect; check `pgrep -fc "[d]p_worker.main"` and GPU mem before believing
  it. Restart the waiter after any stop/start.
- **Persistent tee:** a `tail -F` appends trainer_srun.log to
  `lps1062_bench/trainer_srun_persistent.log` (survives the per-boot
  truncation — F4 hazard). Keep it running.
- **fixc_verify_instrumentation.patch** was authored against the fixc-only
  tree; over the w1/w2 tree it fuzz-applies with hunk 3 MISPLACED
  (`entry.routing_map_dev = ...` landed before `entry = _ReplayEntry()` →
  UnboundLocalError). I hand-fixed the placement on the box (moved after the
  num_global block; AST-verified). hilbert regenerates the patch for the
  archive; the on-box file is the working authority.
- **probe_gradmode_divergence.py** needs (a) `s/Float8BlockwiseScaling/
  Float8BlockScaling/` for TE 2.16.0 and (b) the venv's nvidia libs on
  LD_LIBRARY_PATH (`$VENV/lib/python3.12/site-packages/nvidia/{cu13,cudnn,
  cusparselt,nccl,nvshmem}/lib`) outside launch.sh. My box-local copy runs.
- **T2 launcher:** c10d rdzv fails (node 0 can't resolve itself — the
  /etc/hosts gotcha). Use static `--master_addr/--master_port` like
  launch.sh. My `run_t2_gate.sh` (in overlap_design/) does this correctly.

## 7. In-flight run state (C-prime TIMED arm)

- Booting NOW (job 27, dispatched ~04:17 UTC): ship env + B/F + W1
  (`BT_MOE_PROBS_A2A_COMM=1`) + C-prime (`BT_MOE_DISPATCH_REPLAY_CACHE=1
  BT_MOE_ROUTING_REPLAY_FORCE=1`), VERIFY **off**, `BT_PROFILE_RANKS=0,8`.
  At handoff-writing: 17 procs, 16 gate lines, 0 mismatches, warming.
- **Remaining steps:** (a) wait for health on tj-318g61w-1:8001; (b) timed
  bench `run_bench2.sh A131-131k-d4-cprime-timed --seq-len 131072 --num-gpus
  16 --datums 4 --warmup-datums 2 --repeats 3 --canary-json
  .../A131-131k-d4-arm2-w1.json` (base = ARM 2 = B+F+W1 on this box);
  (c) one steady capture via profile_window_131k_d4.py → scp both ranks to
  `~/perf_profiles/lps-1062/round3/arm-cprime-318g61w/`; (d)
  check_acceptance.py with profile `post-patch-BFC-4mb131k` (ship rows:
  eventsync replay ~0, allgather replay ~0; wall delta is INFORMATIONAL per
  the two-tier rule); (e) report to kepler.
- **C-prime verify soak already PASSED** (the hard part): 20/20 mains, zero
  ARM-1-class raises, stores/hits 6300/6300, misses 0, shape_mismatches 0,
  verify_asserts ~312/step, canary in band, +3.7 GiB cache cost (68 GiB
  headroom stands). The timed arm is the formality + the trace.

## 8. Ladder results so far (this box, all reported to fibonacci)

- **Anchors (B+F, golden EP16/CP16/DP1):** 131k-d4 steady ~705 (step 46.5s,
  peak 201,255 MiB); 16k-d32 ~703.5 (46.7s, 202,079 MiB). Canaries ≤3e-4.
  ~1.4-3% under qr4ggv3 refs = box fabric variance (tail-driven: SendRecv p50
  9.28ms ≈ ref 9.35, but max 63 vs 49ms, total +1.3s/step). CONFIRMED.
- **Two-rank straggler capture (W5):** ranks 0+8 traces at
  ~/perf_profiles/lps-1062/round3/anchor-318g61w/ (md5-verified).
- **ARM 0 (full stack, gates off):** inertness PASS (drift +0.0007..0.0017,
  inside the demonstrated cross-boot floor).
- **W1 (ARM 2):** MECHANISM PASS (probs 900/900 off-stream, gap −2.07ms/pass,
  token a2a flat, telemetry engaged) / WALL MISS on this box (−0.4..−0.6s
  honest vs −2.4 bar — tail-bound box). fibonacci two-tier ruling: gate stays
  ARMED for all later arms; +1.3 GiB = the second NCCL communicator.
- **FIX C (ARM 1):** verify gate caught a real bitwise mismatch at boot
  warmup (boundary flip + downstream-of-routing splits nondeterminism —
  instrumented signature captured). PARKED; superseded by C-prime.
- **C-prime:** verify soak GREEN (see §7); timed arm in flight.
- **W2:** T2 numerics gate HARD FAIL — W2-on backward defect
  (te_moe_chunk_sort_fwd backward returns a ~half-size grad at index 3;
  simplest case: balanced, no ckpt, no fixc). BLOCKED Mac-side (helmholtz).
  Log: lps1062_bench/t2_w2_gate_FAIL_318g61w.log + Mac round3/.

## 9. Remaining ladder (fibonacci's order, now kepler's)

1. C-prime timed arm (§7) — in flight.
2. W2 t2 gate + W2 arm — BLOCKED on helmholtz's backward fix; re-run
   `run_t2_gate.sh` (BT_T2_SKIP_FIXC=1, replay-cache unset, W1 armed) once a
   fixed w2 patch lands; ALL PASS before any timed W2 run.
3. W3 canary: stage done; arm = passing stack + `BT_MOE_LOOKAHEAD_RECOMPUTE=1`;
   bars: canary in band, kicks/hits telemetry, step delta (informational),
   peak mem +2-4 GiB expected, one steady capture. W3 applies over the FIX C
   tree (recompute.py hunk needs FIX C's pass-marker wrap as context).
4. **F2 boots (LAST):** the DP>1 deadlock fix. Package complete Mac-side at
   `~/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/overnight_20260810_round3/f2/`:
   `DESIGN_F2_bohr.md`, `f2.patch` (0e0b65a6, WITH the mandatory kill-switch
   WARNING; applies clean check-only; 9/9 Mac CPU + 39/39 existing tests
   green), `f2_main_rebase.patch` + `REBASE_NOTES.md` (ship-PR port),
   `ONBOX_VALIDATION_F2.md` (the a-d recipe + fibonacci's NaN×0 and
   finalize-once additions). On-box: apply f2.patch to **server/src** (NOT
   the vendored tree), boot expB-ep16cp8dp2.json (EP16/CP8/DP2), run the
   recipe. Time-box the kill-switch hang re-confirmation to ~10 min (py-spy
   signature, then kill — don't burn 35 min).
5. ARM 5 (fixa_v3, conditional): staged+md5'd at lps1062/fixa_v3/; only if
   the night has slack after all of the above.

## 10. Messaging

Cross-session: `~/.agents/scripts/send-message.sh <session> "..."`. Report
ladder results to **kepler** (the new orchestrator). fibonacci's orchestrator
handoff is at `runs/overnight_20260810_round3/overlap/HANDOFF_ORCHESTRATOR.md`.
