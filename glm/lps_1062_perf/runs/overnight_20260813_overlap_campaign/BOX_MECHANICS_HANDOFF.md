# BOX MECHANICS LANE — handoff (doppler/lovelace, LPS-1062 overlap campaign, 2026-08-14)

One page. The box driver owns: provisioning, tree/venv state, trainer
lifecycle, bench execution, evidence pulls. Everything below was learned the
hard way tonight on 2x8 B300 (ali-apse7-prod-1).

## Provisioning / the queue race

- The ali B300 cluster is often FULL. The platform reaps PENDING jobs at
  ~60-62 min (signature: the bt-interactive-session configmap is deleted
  FIRST, then rank1 FailedMount, then TRAINING_JOB_FAILED with a null
  error_message). A half-scheduled 2-node job (rank0 Running, rank1 Pending)
  is reaped the same way — partial progress is NOT preserved.
- `devbox-up` dies on API 429 storms at step 3 (wait_running). The recovery
  pattern is a RESUME DRIVER: reconstruct ctx for the EXISTING job (never
  re-call provision.create — no double-queue), wait kubectl-first
  (KUBECONFIG=~/.kube/ali-apse7-prod-1.yaml, ns org-99340d71961343c28c5c567d705ab0c0,
  pods baseten-training-job-<id>-multinode-0[,-1]), API only to confirm,
  429-tolerant. Working implementation: /tmp/devbox_resume.py (takes the job
  id as argv[1]). Poll >=60s; distinguish API-unreachable (retry) from
  job-genuinely-gone (die) or you false-die on DNS flaps.
- The stock wait_running's 15s API polling is a 429 magnet. If it dies, fire
  the resume driver on the new job id immediately.
- Standing authorization pattern: on pending-timeout FAIL, recreate
  immediately; never more than ONE pending job at a time; log each cycle.

## devbox-up gotchas

- **CUDA toolkit mismatch (B300/ali):** the pod image is cuda:12.8.1-devel but
  the lock's torch is cu13. Lock-faithful venv builds (make server-venv
  CUDA_FLAVOR=cu13) FAIL on the fast-hadamard-transform git-pin source build
  (torch cpp_extension enforces nvcc==torch cuda). Fix on the leader:
  `apt-get install -y cuda-nvcc-13-0 cuda-libraries-dev-13-0` (the umbrella,
  NOT per-lib whack-a-mole), `CUDA_HOME=/usr/local/cuda-13.0`. Papercut
  pc_bad16ec3b02c.
- The venvs use EDITABLE installs (trainers_server/bridge/mcore are .pth to
  the source tree) — a tree swap is LIVE, no venv rebuild. The cudnn-frontend
  wheel bump (1.27.0) persists in the venv across swaps.
- CPFS (shared FS) hazards: (a) files scp'd onto the shared FS can read back
  as all-NULs from a SIBLING node — md5/sha256 every staged file on the leader
  AND the worker; (b) partial package writes happen (nvidia-nvshmem-cu13
  landed dist-info but no libs once → torch import died on libnvshmem_host.so.3
  → reinstall --reinstall --no-deps fixes).
- The vendored-wheels build bug is unfixed: a fresh `make server-venv` runs
  `manage_wheels.py fetch` (needs DOCKERHUB_PULL_* creds in the env). After any
  rebuild, re-verify cudnn-frontend (the campaign bump = PyPI
  nvidia-cudnn-frontend==1.27.0 --no-deps from /tmp, backend cu13 9.19.0.56
  unchanged) + re-run BOTH DSA regression files (5-passed gate).
- Wheel provenance matters: an inherited shared-FS venv can carry mystery
  swaps (tonight: a PyPI fast-hadamard 1.0.4.post1 over the lock's git v1.1.0).
  Jack's rule: anchors measured on a mystery-provenance venv taint everything
  downstream — rebuild from the lock before measuring.

## Trainer lifecycle (the squeue rule + the wedge patterns)

- Lifecycle ONLY via /root/.cache/user_artifacts/.devbox_up/ scripts:
  start_trainer.sh / wait_trainer_health.sh (BACKGROUNDED — zero-tolerance
  wait rule: never block your session on a manual wait/sleep-loop; stay
  responsive) / stop_trainer.sh.
- **THE SQUEUE RULE (standing, tonight's F4-class lesson):** at ANY arm
  transition, verify `squeue` shows ZERO devbox_trainer jobs before
  dispatching the next boot. "Trainer down" must mean the SLURM JOB is gone,
  not just the processes. A leftover armed-idle trainer holds the 2-node
  allocation and the next boot's srun queues PENDING behind it (you'll watch
  the OLD trainer's idle footprint thinking it's the new boot).
- stop_trainer.sh does `scancel --name=devbox_trainer` — over-broad when two
  devbox_trainer jobs coexist (kills the new pending one too). When two
  coexist, scancel by JOBID the one you mean. (papercut filed by kolmogorov.)
- stop_trainer.sh's own GPU-state collection races the teardown ("Could not
  collect GPU state within 30s" + "Requested nodes are busy") — IGNORE it; the
  follow-up `srun --overlap` cleanliness sweep (no dp_worker procs + GPUs at
  0) is the real check.
- ARMED-IDLE posture pins GPUs at 100% (posted NCCL recv) — looks exactly like
  a wedge on nvidia-smi. Under BT_SKIP_WARMUP=1 the trainer goes armed-idle
  with NO "READY — pipeline" banner; the bench driver's status probe
  (world_size=16) is the readiness signal, not the banner.
- wait_trainer_health.sh reads the EXISTING trainer_srun.log on its first
  poll — after a relaunch, start_trainer.sh clobbers that log, and the waiter
  can read the PREVIOUS trainer's death tail and declare a false "TRAINER
  PROCESS DIED". Re-arm the waiter after the clobber settles; distrust a DIED
  verdict within the first minute of a relaunch.
- Snapshot failure logs BEFORE relaunch (start_trainer.sh clobbers
  trainer_srun.log). The clobbering rule: pull evidence Mac-side immediately
  (the box can die).

## Bench execution + evidence

- Kit: /root/.cache/user_artifacts/lps1062/ (bench_driver2c.py,
  run_bench2c.sh = driver + per-node mem pollers + fold_mem, parity_driver.py,
  profile_driver_new.py). dN = N datums x seq-len per window. The driver hits
  the trainer HTTP :8001 on the leader.
- F1 landmine: async save hangs under CP>1. The BT_SAVE_STATE_SYNC=1
  "workaround" was a NO-OP until the db5d1826 hunk (verify the hunk is applied
  — grep the tree, don't trust the env var being set). Papercut
  pc_fb3a242ec978.
- M=1 is illegal under VPP2 (the interleaved schedule needs M>=PP=2). This
  bites the BENCH driver's warmup0 (default --warmup-datums 1) one layer up
  from the trainer warmup — use --warmup-datums 2 on VPP2 arms.
- profile_driver_new.py --control-repeats 0 crashes the post-capture summary
  math (ZeroDivisionError) — cosmetic, the trace is intact; use
  --control-repeats 1 to avoid the noise.
- BT_PROFILE_RANKS=0,8 arms the profiler at boot; the actual capture is
  triggered by the driver's /runtime_profile/start-stop. Traces land in
  /tmp/checkpoints/profiles/torch_trace/ (POD-LOCAL — rank8's is on node 1);
  copy to the shared FS + pull Mac-side IMMEDIATELY.
- Evidence stamping (tonight's collision lesson): the parity driver writes
  parity_<label>.json with NO provenance fields. Stamp the LABEL with
  job-id + UTC timestamp (it lands in both the filename and the JSON's label
  field), verify the stamp box-side before any pull, sha256 box-vs-Mac on
  every pull. Papercuts pc_e624bb46a4bd / pc_24578fc149c8.
- Memory reads: the poller (nvidia-smi reserved) is the OOM-relevant metric;
  the trainer's /status gpu_memory is the torch view (~12 GiB lower = the
  non-PyTorch NCCL/driver overhead). NEVER mix the two metrics against a bar
  (pauli's rule). torch-reserved creeps ~+20 GiB over early steps before
  plateau — take fit reads at matched step positions or plateau, never
  window-1.

## Environment fragility (the Mac driving the box)

- The Mac can lose opendirectoryd (getpwuid fails → ALL ssh exits 255 "No user
  exists for uid 501", sudo breaks too — needs Jack/GUI or spontaneous
  recovery), the gh keychain token (deleted → gh auth token empty; the BOX's
  /root/.git-credentials from provisioning still works for box-side git), and
  DNS can flap (kubectl/API blind in waves). Check these before concluding a
  box-side fault. The resume driver's kubectl-first + API-tolerant design rides
  out the DNS flap.

## Contacts / where the docs live

- Campaign spine: experiment_artefacts/glm/lps_1062_perf/pp2cp8ep8/NOTEBOOK.md
- Tonight's run folder + my full status log:
  experiment_artefacts/glm/lps_1062_perf/runs/overnight_20260813_overlap_campaign/
  (DOPPLER_STATUS.md, BOX_SCHEDULE.md, CAMPAIGN_ORDERS.md, the W-specs, the
  S2 noise-matrix prereg, s3_wedge_snapshot/, s1_soak_mem/, bench_jsons/)
- Traces Mac-side: ~/perf_profiles/lps-1062/pp2cp8ep8/
