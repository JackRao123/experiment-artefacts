# DOPPLER STATUS LOG — LPS-1062 overlap campaign (box mechanics)

Chronological, append-only. Written because fermi's iTerm tab died ~23:56 CDT
(send-message undeliverable) — this is the durable record until the tab returns.

## 2026-08-13 ~23:56 CDT — MAC-SIDE ENVIRONMENT FAILURE (three faults)

System load spiked to 25 around 23:50 (now falling, 4.x). Three concurrent
Mac-side failures:

1. **opendirectoryd user lookup DEAD** — `getpwuid(501)` fails system-wide
   (python KeyError, `dscl . -read /Users/jackrao` → eServerError, `whoami`
   prints "501"). Consequences: ALL ssh exits 255 ("No user exists for uid
   501"); `sudo` broken ("you do not exist in the passwd database") so I
   cannot restart opendirectoryd myself. Needs Jack (GUI/sudo) or
   spontaneous recovery.
2. **gh keychain item DELETED** — `gh auth token` → "no oauth token found";
   `security find-generic-password -s gh:github.com` → not found. Was valid
   at 23:16 (cycle-5 preflight passed), gone by 23:52. No GITHUB_TOKEN in
   zshrc/zshenv/env, no ~/.git-credentials fallback. Re-auth needs Jack.
3. **DNS flapping** on rancher.infra.basetensors.com (kubectl) — recovers
   for minutes, drops again. api.baseten.co affected the same way.

### Impact on the mission + workarounds in place

- **Box race continues.** Resume driver armed on wprm693 (pid 4396, log
  /tmp/devbox_up_campaign5_resume.log), kubectl-first, 60s polls.
  - Patch A: DNS-flap fix — API-unreachable (None) no longer counts as
    job-vanished; only a definitive API answer of "absent" + 5 consecutive
    empty kubectl reads triggers the vanish-die.
  - Patch B: lazy gh-token — step 8 falls back to `clone_trainers_tokenless`
    (verify existing shared clone + campaign branch ref + initialized
    submodules; skip credential write and /root/trainers root clone). The
    shared clone persists on the cluster FS from prior boxes.
- **Blocked when rank1 lands:** steps 4-13 are all ssh-based → hard-blocked
  until opendirectoryd recovers. Nothing box-side I can do from here without
  ssh.
- **For Jack when he's back:** (a) restart opendirectoryd
  (`sudo killall opendirectoryd`) or reboot; (b) `gh auth login` to restore
  the GitHub token; (c) check iTerm tabs (fermi's died).

## Box-race ledger (cycles, all 2x8 B300 ali, all reaped at ~60-62 min pending)

| cycle | job | created | fate |
|---|---|---|---|
| 1 | q8eg0gq | 19:02 | reaped ~20:02 (both pods Pending throughout) |
| 2 | wov4kzq | 20:08 | rank0 Running 20:10, rank1 never scheduled; reaped ~21:08 |
| 3 | qjlr5pw | 21:11 | rank0 Running ~21:14, rank1 never scheduled; reaped ~22:12 |
| 4 | qe5gepq | 22:13 | rank0 Running ~22:16, rank1 never scheduled; reaped ~23:13 |
| 5 | wprm693 | 23:16 | rank0 Running ~23:19, rank1 Pending; reaper due ~00:16 |

Reaper signature (both verified kills): platform deletes the
bt-interactive-session configmap first → rank1 FailedMount → whole job
TRAINING_JOB_FAILED. error_message is null at the API.

Node-freeing rate (namespace watch, /tmp/node_watch.log): ZERO frees
21:21→23:15. Map frozen: trainers yqv18gq×4 (2d3h+), 5qeke0q×1, 2qjk123×1;
jobs wgo8p43×2 + four ×1.

## Standing orders being executed (fermi, 20:0x)

- Recreate immediately on pending-timeout FAIL; never more than ONE pending
  job at a time; log each cycle. (Authorized explicitly.)
- On box up: wheel bump (nvidia-cudnn-frontend==1.27.0 --no-deps → trainer
  venv, backend cu13 9.19.0.56 unchanged; DSA regression files GREEN),
  confirm tree 73c24b00+TF32, d2 canary, re-anchor d4 (~878-886) and d16
  (~984 ±3% = pre-registered PASS band 954.5-1013.5; outside → STOP + report).
- Trainer lifecycle only via /root/.cache/user_artifacts/.devbox_up/
  start/wait/stop scripts; wait = wait_trainer_health.sh.
- BT_SAVE_STATE_SYNC=1 on all bench runs.

## 2026-08-14 ~00:5x-01:3x CDT — VENV PROVENANCE RECORD (Jack's rebuild order, wprm693)

Per Jack's direct order (via bayes, 00:3x): anchors must trace to a known-fresh
env, not the inherited shared-FS venv. Executed on wprm693 leader:

1. **Rebuild from the campaign tree lock** (tree 73c24b00, branch
   jackrao/lps-1062-pp2cp8ep8): `make server-venv CUDA_FLAVOR=cu13` +
   `make sampler-venv` — both rc=0 (~01:2x CDT).
   - Build blocker found+fixed: pod image is cuda:12.8.1-devel but torch is
     2.11.0+cu130 — the lock's fast-hadamard-transform git pin (v1.1.0,
     1cc807e) source-build needs nvcc matching torch's CUDA. Installed
     cuda-nvcc-13-0 + cuda-libraries-dev-13-0 on the leader
     (CUDA_HOME=/usr/local/cuda-13.0). NOTE: worker rank1 still has only
     CUDA 12.8 — runtime is fine (venv ships its own libs), only source
     builds need the toolkit.
   - CPFS casualty: nvidia-nvshmem-cu13 3.4.5 installed with dist-info but
     NO libs (partial write) → torch import failed on libnvshmem_host.so.3.
     Reinstalled (--reinstall --no-deps), verified.
2. **Lock-faithful diffs vs the inherited venv** (Jack's instinct confirmed):
   - fast-hadamard-transform: inherited ran PyPI 1.0.4.post1 (a mystery
     swap); rebuilt = lock's git v1.1.0 source build.
   - cudnn-frontend: lock installed the freshly-fetched vendored
     1.27.0.dev20260803+git7478516 (fetch-wheels refreshed today — the stale
     1.26.0+dsatopk1 vendored shim is GONE from vendor/wheels).
3. **Bump re-applied per order**: PyPI nvidia-cudnn-frontend==1.27.0
   --no-deps from /tmp → cudnn_frontend 1.27.0, backend nvidia-cudnn-cu13
   9.19.0.56 UNCHANGED.
4. **Import sweep GREEN** (9 modules): torch, transformer_engine, cudnn,
   deep_ep, flash_mla, fast_hadamard_transform, pybind11, megatron.core,
   flash_attn.
5. **DSA GATE GREEN**: test_cudnn_dsa_indexer_launch_stream.py +
   test_cudnn_dsa_indexer_topk.py = **5 passed, 33s** on this box's GPUs.

Venv state for all campaign measurement from here: fresh lock-resolve +
PyPI cudnn-frontend 1.27.0 override. Sampler venv also rebuilt (rc=0).

## 2026-08-14 ~01:4x CDT — W1a RE-ANCHOR RESULTS (fresh provenance venv, wprm693)

All on the rebuilt lock-faithful venv + PyPI cudnn-frontend 1.27.0, tree
73c24b00+TF32, headline PP2/CP8/EP8 @131k, BT_SAVE_STATE_SYNC=1, bench kit
run_bench2c.sh, labels w1a-131k-{d2,d4,d16} (JSONs on box at
/root/.cache/user_artifacts/lps1062_bench/).

| rung | result | old anchor (fixed wheel, inherited venv) | delta |
|---|---:|---:|---:|
| d2 canary | loss 12.3258/12.3114/12.3229, gn 0.428/0.391/0.482 — PASS (703 t/s, fill/drain) | — | — |
| d4 | **848** t/s/GPU (847/849), step 38.6s, mfu3x 7.7%, peak 165 GiB smi | ~878-886 | **-3.6%** |
| d16 | **945** t/s/GPU (929→961, still warming), step 138.7s, mfu3x 8.6%, peak 164 GiB driver / 187 smi | 984 (window 947-1023) | **-4.0%** |

**d16 landed OUTSIDE the pre-registered band (954.5-1013.5) → STOPPED per
protocol, reported to bayes, no levers touched.** Consistent small negative
offset on both rungs; prime suspect = lock-faithful hadamard 1.1.0 vs the
mystery PyPI 1.0.4.post1 the old anchors ran on (or a low boot draw; d4
within-boot was tight, mains trended up at d16). Trainer left HOT on wprm693
pending bayes's ruling (accept-and-rebase / extend repeats / investigate).

## 2026-08-14 ~02:1x CDT — d16 EXTENSION + PLATEAU VERDICT (bayes ruling executed)

Extension (hot trainer, 6 more mains, label w1a-131k-d16-ext):
962/960/959/958/958/958 — **plateau 958-962, <1% consecutive deltas from
main1**. Warm-up hypothesis CONFIRMED; the fresh-boot 945 was fill/warm-up
depression. Plateau 958-962 is INSIDE the pre-registered band
(954.5-1013.5) → per bayes's pre-registered condition, the hadamard A/B
(swap 1.0.4.post1 in, one d16) is **CANCELLED**.

**OPERATIONAL ANCHORS (anchor-20260814, fresh lock-faithful venv):**
- d4 = 848 t/s/GPU (step 38.6s, mfu3x 7.7%)
- d16 = 958-962 t/s/GPU hot-plateau (step ~136.7s, mfu3x 8.7%, peak 164 GiB
  driver / 187 GiB smi); fresh-boot first-mains 945.

**⚠ PROVENANCE ASTERISK (standing flag per bayes order 5):** all of
tonight's absolute numbers carry "lock-faithful venv, -3.5-4% vs the
record's provenance" — the 984/878-886 records ran the mystery-hadamard
inherited venv. Tonight's lever verdicts are self-controlled fresh-boot A/Bs
on THIS venv, so they do not depend on matching the record.

Box handed to grothendieck for W1b at ~02:1x with the trainer HOT at
plateau (headline config, BT_SAVE_STATE_SYNC=1; bench kit
/root/.cache/user_artifacts/lps1062/).

## 2026-08-14 ~03:1x CDT — W1b (block+K21) FIT PROBE: FAILED (memory-infeasible at 131k)

grothendieck drove, doppler supervised. Boot #1 OOMed in trainer-internal
warmup (rank15 stage-1 MoE bias_act, 260.33 GiB alloc / 267.69 cap).
Mechanism (grothendieck): the seq-64 warmup datum PADS to the full 131k
buffer — trainer warmup is a full-shape M=1 fwd+bwd; blockK21's 19 eager
stage-1 layers add ~47 GiB stored over full-recompute at M=1.
bayes GO PATH A: boot #2 with BT_SKIP_WARMUP=1 (env delta #2; #1 =
BT_PROFILE_RANKS=0,8). Result: the OOM MOVED to the first bench warmup0
(full-shape M=1, post-settle, pollers running) — node-1 GPUs 266.3-266.7 GiB
vs 267.69 cap, ranks OOMing on 1.18-2.36 GiB allocs. **Verdict: blockK21 is
structurally memory-infeasible at 131k M=1 — skip-warmup moves the OOM, does
not remove it. Zero bench windows completed; no W1b perf data exists.**
Supervisor actions: shouted at the 266.6 GiB poller reading, then cleaned
the wedge (killed the hung bench driver facing a 1h fb timeout,
stop_trainer, both nodes verified clean, GPUs at 0). Box idle, awaiting the
arm's next config (resize K per jacobi's per-layer constants).
Pre-registered bayes contingency noted: any verdict within ±2% of win bar →
ONE env-symmetric OFF re-pair (moot for the OOM leg; applies to whatever K
fits next).

## 2026-08-14 ~04:4x CDT — W1b K=25 RETRY: SECOND FIT MISS → DOES-NOT-FIT (doppler driving)

Succession: grothendieck stalled ~1h (restart-wave pattern); bayes handed
W1b execution to doppler at the clean boundary; grothendieck acked, pit-wall
role. Executed bayes's takeover order: cancelled stray slurm 24, staged
pauli's ratified blockK25 config (sha256 ddc75eed… 3-way verified), booted
with W1a env + BT_SKIP_WARMUP=1 + BT_PROFILE_RANKS=0,8.

Ops note: the wait_trainer_health.sh first-poll read the PREVIOUS trainer's
log tail during the start_trainer.sh clobber race and declared a false
TRAINER PROCESS DIED at launch+5s — trainer was in fact healthy (verified by
bench status probe). Watch out: waiter verdicts within the first minute of a
relaunch are suspect; re-arm the waiter after the clobber settles.

**FIT READ (d2 warmup0, poller reserved): node-1 max 271381 MiB = 265.0 GiB
vs the 255 line — BREACH by 10 GiB; leader 218.3 GiB.** No OOM (no
traceback; window grinding at 96%+ reserved when stopped per protocol).
K=25 predicted 244.6 — the K-slope model under-predicts by ~20 GiB at the
margin. Per the runbook's second-miss rule + bayes's takeover order: STOP,
lever verdict = **blockK does not fit at 131k on this stack (K=21 OOM,
K=25 over the reserved line)** — a real answer. No d4/d16 pairs ran; no
perf data exists. Dial-family boots OFF until pauli revises the memory
model. Box idle-clean, trainer down, GPUs 0.

## CORRECTION (04:5x CDT) — W1b K=25 warmup0 DID complete

grothendieck's post-mortem recon (verified against the runlog): the K=25 d2
warmup0 COMPLETED — fb=193.4s (compile-heavy, 42 t/s/GPU), loss 12.3242
IN-BAND, gn 0.4185 in-band — then "Terminated" (my pkill during the STOP).
My earlier "zero bench windows completed / window grinding when stopped" was
wrong (runlog tail read too early; the driver was mid-main0). Also: trainer
peak line = 256.8 GiB reserved / 253.2 allocated vs my poller-max 265.0 —
an ~8 GiB poller-vs-trainer metric gap (non-PyTorch overhead). Both metrics
exceed the 255 bar (poller-reserved is pauli's bar metric per the runbook);
the DOES-NOT-FIT verdict is unchanged. fold_mem.py errored on the missing
result JSON (driver killed before write) — cosmetic, expected on a kill.

## 2026-08-14 ~05:0x-05:3x CDT — W1c BOOT 1 (off-arm, record env) COMPLETE

Env: canonical mission env (BT_TF32_LM_HEAD=1 + ship NCCL 3 knobs +
BT_SAVE_STATE_SYNC=1) + BT_PROFILE_RANKS=0,8; B/F gates OFF (default).
Mission config (md5 aecbff20…), tree 73c24b00+TF32, mcore dirty-diff
e1e46818… verified exact.

- d2 canary PASS: 12.3221/12.3126/12.3209 (band), gn 0.417/0.400/0.482,
  drift vs W1a d2 |dloss|<=0.0037.
- **d4 pair: 915 / 925 (mean 920)**, steps 35.8/35.4, peak 143 GiB driver.
- **d16 pair: 1038 / 1063 (mean 1050.5)**, steps 126.3/123.3, peak 163 GiB.
- Traced d16 window captured: rank0 (2.8 GB) + rank8 (3.0 GB) kineto +
  memory snapshots; shared-FS copies lps1062_pp2/traces/w1c_b1_off_d16_rank{0,8}
  .pt.trace.json; Mac pull to ~/perf_profiles/lps-1062/pp2cp8ep8/.
  (profile_driver_new.py --control-repeats 0 crashes post-stop summary math
  — ZeroDivisionError, cosmetic, trace intact; kolmogorov filing.)

**CROSS-CHECK (bayes rule (a) CONFIRMED):** record env on the fresh
lock-faithful venv = 920 d4 / 1050.5 d16, i.e. +3.9-4.8% / +5.5-8.0% OVER
the 878-886/984 record. The venv is EXONERATED — tonight's -3.5-4% anchor
gap was the missing env (TF32 head + ship NCCL), not provenance. The
hadamard A/B stays cancelled. kolmogorov: "the fixed wheel properly venv'd
recovers the stale wheel's perf class with correct numerics — the whole-wheel
saga closes." Reference class locked for the B/F A/B: d4 920 (spread 10),
d16 1050.5; win floor d4 on-arm = 938.4.

## 2026-08-14 ~05:5x-06:2x CDT — W1c BOOT 2 (B/F on-arm) COMPLETE + BOOT 3 launched

Boot 2 env: canonical mission env + BT_PROFILE_RANKS=0,8 +
BT_DSA_CP_LAYOUT_CACHE=1 + BT_THD_ROPE_HOST_CACHE=1. HARD GATE PASS: both
ACTIVE lines ×16 ranks at WARNING in the boot log (genuine engagement).

- d2 canary PASS: 12.3262/12.3110/12.3217, gn 0.420/0.390/0.481; drift vs
  boot-1 off-arm ≤4.1e-3 (< 5e-3 bar).
- d4 pair: 929 / 974 (mean 951.5) vs off-arm 920 → +3.4% but
  perf-indistinguishable as-stood (on-arm spread 45 > Δ 31.5). kolmogorov:
  the spread is a warmup trend, not noise; documented d4-r3 warm run =
  **966** → on-arm warm class ~970 vs off-arm warm 925 ≈ **+4.9%** (top of
  gauss's +0-5% band).
- **d16 pair: 1088 / 1118 (mean 1103)** vs off-arm 1050.5 → **+5.0%**,
  |Δ| 52.5 > max-spread 30 ⇒ DISTINGUISHABLE, central kepler band (+4-7%).
- Memory flat vs off-arm at every rung (143 d4 / 163 d16 driver).
- Traced d16 window: rank0 1.32 GB + rank8 1.43 GB — ~half the off-arm's
  2.8/3.0 GB (host-sync-removal signature visible in event count). Shared-FS
  lps1062_pp2/traces/w1c_b2_on_d16_rank{0,8}.pt.trace.json; Mac pulls in
  flight (relay flaked once; re-pull running).
- Canary-json caveat recorded: drift reads on runs deep into a boot (r3 was
  ~18 windows in) reflect training progression vs the fixed baseline, not
  numerics — first-window band checks are the numerics gate.

BOOT 3 (NCCL off-arm): ship env OFF (3 knobs unset), B/F off,
BT_TF32_LM_HEAD=1 + BT_SAVE_STATE_SYNC=1, d2+d4 only. /proc verification of
knob absence at boot.

## 2026-08-14 ~06:4x CDT — W1c BOOT 3 (NCCL off-arm) COMPLETE; W1c window CLOSED

Boot 3: ship env OFF (/proc-verified: all 3 NCCL knobs absent on the trainer
process; positive controls BT_TF32_LM_HEAD=1 + BT_SAVE_STATE_SYNC=1 present),
B/F off, mission config. d2 canary PASS (12.3252/12.3112/12.3206, gn
0.421/0.388/0.475). d4 pair: 923 / 929 (mean 926, spread 6), peak 144 GiB.

**NCCL ARM-1 (d4):** ship-ON 920 (915/925) vs ship-OFF 926 (923/929) —
|Δmean| 6 ≤ max-spread 10 ⇒ **ship NCCL env INERT on PP2/CP8/EP8 @131k**
(pre-registered rule; mechanistically consistent with EP8 = intra-node here).
Documented for the ship package.

W1c window summary: B/F on-arm d16 +5.0% (distinguishable, kepler central
band), d4 warm-class +4.9% (top of gauss's band), NCCL inert, memory flat,
canaries clean at every boot, d16 traces for both arms captured + safe.
Adjudication + NOTEBOOK = kolmogorov. Next: W2 (ramanujan shim canary) —
tree swap to a3da1223 is restore-safe (mcore dirty diff byte-exact snapshot
e1e46818… Mac-side at results/mcore_dirty_diff_w1c_snapshot.patch).

## 2026-08-14 ~07:1x-07:5x CDT — W2 (shim canary) leg 1: control PASS; flag-ON boot BLOCKED (validator wall)

Tree swap executed: trainers a3da1223 / bridge 146f2636 / mcore b37c01f2e
(landmine fix at tip). B/F gate stack carried over the submodule checkout
byte-exact and was cleaned to exact b37c01f2e (snapshot e1e46818… Mac-side).
Inert leftover: untracked megatron/core/lookahead_checkpoint.py (W3 debris,
unimported — left in place). Venv: editable installs => swap is live, no
rebuild; cudnn-frontend still 1.27.0. Configs staged + md5-verified 3-way.

Leg 1 control (flag-OFF plain PP2 32k, BT_SAVE_STATE_SYNC=1 +
BT_SKIP_WARMUP=1): **band 12.292-12.302, gn 0.356-0.410, peak 202 GiB
driver / 225 smi** — ramanujan adjudicated CONTROL PASS (matches the
selective twin + the 32k plain-selective class). Ops note: under
BT_SKIP_WARMUP=1 the trainer goes armed-idle with NO "READY — pipeline"
banner (GPUs pin 100% on posted NCCL recv); bench status probe = readiness.

Flag-ON boot BLOCKED at bootstrap by the known exp03 validator wall:
bridge comm_overlap.py:500 asserts moe_shared_expert_overlap off when
overlap_moe_expert_parallel_comm is on; the tree's clearing is flex-gated
(megatron_config.py:64-67) and our config runs alltoall. The exp03 standing
patch (clear under the overlap flag regardless of dispatcher) is NOT in the
shim branch. Reported to ramanujan with the proposed 3-line fix; ruling
pending. Box idle-clean, control trainer down.

## 2026-08-14 ~07:5x-08:2x CDT — W2 leg 1 flag-ON: STOP (executor-class error)

ramanujan committed the validator-wall fix to the branch (new tip d34f76d9:
clear moe_shared_expert_overlap whenever the EP-overlap flag is on,
dispatcher-independent + pinning test; reviewed SHIP). Re-swapped + verified
(bridge 146f2636, mcore b37c01f2e clean, cudnn 1.27.0). Flag-ON boot then
cleared the validator and reached armed-idle.

d2 canary: first attempt died on the M=1 illegality one layer up — the bench
driver's warmup0 is an M=1 window, illegal under VPP2 (the design doc's
"M=1 illegal" note was written for the trainer warmup; the bench warmup is
the same class). Documented deviation: --warmup-datums 2 (M=2 warmup); the
M=2 mains still pair against the control's M=2 mains.

Retry then hit the REAL failure: RuntimeError 'expected mat1 and mat2 to have
the same dtype, but got: float != c10::BFloat16' — chain: combined_1f1b
executor -> forward_step_calc_loss -> chunked_lm_head _project_logits ->
LoRA adapter linear_in. READ: the shim's fp32 boundary (hidden.float()
upcast, exact-for-CE) reaches the LoRA-wrapped head projection with no bf16
re-cast. The flag-OFF control ran the same chunked head + LoRA clean, so the
delta is the shim executor's loss path. NOT the landmine (b37c01f2e passed
first hardware contact). STOP reported to ramanujan; trainer up, drivers
cleaned, box holding. Also logged: the activation-offload engagement probe
fails harmlessly on this tree (TypeError, caught+warned — valve-commit probe
vs b37c01f2e).

Awaiting ramanujan adjudication. bayes's R2 micro-window for jacobi
(py-spy 250Hz x420s rank0+rank8, ~10 min, on-arm preferred) is queued after
the canary+parity legs resolve — currently blocked behind this bug.

## 2026-08-14 ~08:2x-08:5x CDT — W2 leg 1 CANARY PASS + landmine fix validated; leg 2 PARITY FAIL

ramanujan's fixes landed mid-leg: d34f76d9 (validator wall: clear
moe_shared_expert_overlap under the EP-overlap flag, dispatcher-independent)
then 1cd31535 (the REAL bug: latent chunked_lm_head LoRA-delta dtype bug —
the delta's hidden passed fp32 straight to bf16 adapter weights under the
shim's fp32 plan boundary; fix casts the delta input to the base working
dtype; no-op on the conventional path).

Leg 1 flag-ON d2 canary on 1cd31535: mains 12.3016/12.2925 IN the control
band (12.292-12.302), gn 0.361-0.372, loss trains — ramanujan adjudicated
CANARY PASS. warmup0 12.3052 marginality explained (M=2 documented
deviation). **b37c01f2e landmine fix HARDWARE-VALIDATED** (warmup0 full
fwd+bwd, zero stage-0-backward RuntimeError).

Leg 2 parity (fresh boots both arms, capped 6-datum set 9->6 per
pre-registration, default forward_backward mode): **FAIL both bars** —
loss rel_diff 1.259e-04 (bar 1e-6); logprobs max_abs_diff 3.758 (bar 1e-3).
Pattern: PERVASIVE (87-96% of tokens in every datum >1e-3, magnitude
~1.3-2.3, positions scattered) with aggregate loss nearly equal
(12.2974/12.2989) — reads as a LAYOUT/REPORTING misalignment in the shim
executor's per-token logprob surface, not precision noise; NOT the benign
fp32-boundary case. Reported to ramanujan with the per-datum breakdown;
box holds the flag-OFF parity boot. W2 PASS requires legs 1+2 green — leg 2
is red pending adjudication.

## 2026-08-14 ~09:3x CDT — W2 PAUSED (parity root-cause Mac-side, ramanujan); W1d GO

W2 paused on the leg-2 parity FAIL (ramanujan root-causing the divergence
structure Mac-side; the two parity JSONs delivered to
runs/overnight_20260813_overlap_campaign/ + ~/perf_profiles/lps-1062/incoming/).
W2 arm state preserved for the re-run: tree 1cd31535 committed+pushed at
origin, configs/parity logs/boot env files on the shared FS, JSONs Mac-side.

W1d (kolmogorov's 16k-d32 customer-shape B/F probe) GO. Tree RESTORED to the
canonical W1c bits and verified: trainers 73c24b00 + TF32 patch (stash-popped),
bridge 20fcf2ea, mcore 57efae08b + gate-stack diff sha256 e1e46818… EXACT
(re-applied from the byte-exact snapshot — the restore path worked as
designed). cudnn 1.27.0 persists in the venv (no rebuild needed — editable
installs read the live tree). Mission config intact (aecbff20…).
Boot 1 (off-arm) launching; canary 16k-d8, main 16k-d32 x2; win floor
off-arm+2% mean-vs-spread, band +5-12%, falsifier <+1.5%.

## 2026-08-14 ~09:5x CDT — OPS LESSON: arm-transition trainer leak

The W1d boot-1 "boot watch" ran ~25 min against the WRONG trainer: the W2
parity flag-OFF boot was left armed-idle (held for ramanujan's call) and its
srun job still held the 2-node allocation, so the W1d boot's trainer srun
queued PENDING behind it. The 223 GiB "warmup transient" I read was the OLD
parity trainer's idle footprint. Caught when the log sat at 2 lines (the
clobbered log not advancing). Fix: scancel the stale trainer's jobid
(stop_trainer.sh's scancel --name would have killed BOTH the stale and the
new pending job — surgical by-id cancel needed when two devbox_trainer jobs
coexist), plus a stray md5sum/bash srun. The relaunched boot is genuinely
running now (job 59 RUNNING, procs spawning). STANDING RULE going forward:
at any arm transition, verify squeue shows ZERO devbox_trainer jobs before
dispatching the next boot — "trainer down" must mean the SLURM JOB is gone,
not just the processes.

## 2026-08-14 ~10:4x CDT — W1d (16k-d32 customer-shape B/F probe) WINDOW COMPLETE

Tree: canonical 73c24b00+TF32 restored (mcore dirty sha e1e46818… exact).
OFF-ARM (boot 1): canary 16k-d8 12.2885-12.2975 in band; d32 pair 931/942
(mean 936.5), peak 141 GiB. ON-ARM (boot 2, gates ACTIVE x16 hard-gate PASS):
canary drift <=7e-4; d32 pair 955/1000 (mean 977.5), peak 141 GiB (flat).
Raw mean delta +4.4% (clears the +2% floor) but |Δ| 41 < on-arm spread 45 →
perf-indistinguishable as-stands (same warmup trend as W1c d4). Warm-class
read (steady-state convention): on 1000 vs off 942 = **+6.2%**, inside the
pre-registered +5-12% band, above the +1.5% falsifier. kolmogorov
adjudicates; box holds boot 2 hot. NEXT QUEUED: ramanujan's determinism
probe (bayes-approved) — fresh flag-ON boot on the preserved W2 tree
(1cd31535), parity driver 3x same capped set, ON-vs-ON compares decide
deterministic-vs-race. The W1d->W2-probe transition needs the tree swap back
to 1cd31535 (mcore gate stack cleans off again; snapshot e1e46818… remains
the restore path).

## 2026-08-14 ~11:2x CDT — W1d FINAL WIN (+6.2%); W2 DETERMINISM PROBE = ROUTE (b) BLOCKER

W1d FINAL (kolmogorov): WIN at the customer shape — on-arm 16k-d32 warm class
1000/1000 vs off-arm 942 = **+6.2%**, in the +5-12% band, falsifier clear,
memory flat, canaries clean. (Docs-per-partition scaling holds: +6.2% at
16k-d32 > +4.9% at 131k-d4.)

W2 DETERMINISM PROBE (ramanujan spec, bayes-approved): fresh flag-ON boot on
the preserved W2 tree (1cd31535), parity driver 3x same capped 6-datum set,
forward_backward, no optim_step. COMPARES: within-boot on-r1/r2 = 5.098,
on-r2/r3 = 6.343; boot-to-boot (leg-2 boot-A vs boot-B r1) = 3.424 — ALL
~3-6 THOUSAND-fold past the 1e-3 kernel-noise bar, while aggregate losses
stay tight (rel 2.3e-5/3.8e-5). **The combined-1F1B executor is
NONDETERMINISTIC on the GLM-5.2 DSA stack — a race, route (b) BLOCKER.**
Executor rungs held; escalate mcore-side DSA×executor. First suspect per
ramanujan: the DSA kernel race class. Recurring worst position (datum 3,
pos 25055, 92% through len-27111) on both within-boot compares = a
top-k-boundary-region tell for the investigation. Probe JSONs Mac-side at
~/perf_profiles/lps-1062/incoming/parity_overlap-on-r{1,2,3}.json. Box holds
the flag-ON probe boot (armed-idle); no executor boots while held.

## 2026-08-14 ~11:4x CDT — W3 (W1 probs-a2a) V0 PASS; V1 boot re-armed with profiling

W2 executor rungs HELD (route-b blocker); W2 arm stood down (state preserved,
tree 1cd31535 at origin). Campaign pivoted to W3 = W1 probs-a2a validation
(hausdorff spec, kolmogorov adjudicates as proxy, doppler drives).

Tree: the staged W1-port chain — trainers 11ad3e26 / bridge 595f0b7f / mcore
a58990a91 (submodule status exact gitlink matches, mcore clean; B/F gate
stack correctly absent, snapshot e1e46818… is the restore path). cudnn
1.27.0 persists. The W1 gate (BT_MOE_PROBS_A2A_COMM) + guard-relax (span
check now telemetry-only) confirmed in source.

**V0 PASS (kolmogorov-adjudicated):** 16/16 armed verbatim + 16/16
gate-ACTIVE; comm-2 rendezvous clean at first gated dispatch (warmup, no
hang); TWO full optimizer steps (d1: warmup0 fb 17.8s/optim 5.3s/loss
12.3242/gn 0.4157; main0 12.3085/0.4014, in band). The multi-EP-group
verification PR #28 deferred = DONE on the 2-EP-group topology. Peak 167 GiB
(+2.9 vs off-arm class; spec's +1.3 buffer estimate ~1.5x low — noted for
ship docs). Env: canonical mission env + the gate (kolmogorov-confirmed).

V1 next: the ON arm re-booted with BT_MOE_PROBS_A2A_COMM=1 +
BT_PROFILE_RANKS=0,8 (one boot serves V1 trace+counters, V2 canary, V3
on-arm perf — profiler armed-not-capturing doesn't distort perf). Counters
fire every 300 gated dispatches (crossed on the d16 runs). V3 OFF arm gets a
FRESH measurement on this w1-port tree per kolmogorov's same-tree discipline
(not borrowed from W1c's 920/1050.5).

## 2026-08-14 ~12:1x-12:5x CDT — W3 V1+V2 PASS; V3 ON-arm landed (OFF arm pending)

V1 (mechanism) PASS: counter telemetry firing — window 3 {token_issues 900,
probs_issues 900, waits 1800} = the 1:1:2 signature (every token dispatch
paired with a probs dispatch on comm 2, both waited). Traced d4 window:
rank0 702MB + rank8 749MB (shared FS + Mac pull). jacobi owns the comm-2
off-stream read.

V2 (numerics) PASS: d2 canary mains 12.3029/12.3060 in band, gn
0.389/0.457, no drift (W1 bitwise-safe by construction).

V3 ON-arm (gate ON, canonical env, w1-port tree): **d4 845/844 (mean 844.5,
spread 1); d16 937/952 (mean 944.5, spread 15)**; peak 144/162 GiB. Both
rungs landed BELOW the W1c reference class (920/1050.5) — trending toward
the lever model (-3.4..-5.1s/step win) being FALSIFIED on this topology, but
the decider is the FRESH OFF arm on the same w1-port tree (kolmogorov's
same-tree discipline), now booting. If OFF lands ~920-class, ON 844.5 =
~-8% at d4 = a real W1 perf LOSS (the second comm's wait not hidden at this
shape).

## 2026-08-14 ~12:5x CDT — V3 STOOD DOWN (demoted); W4 (option-6) transition

kolmogorov+bayes: V3 (W1-alone perf A/B) demoted — jacobi's trace read
predicts the null; recorded as "structurally in, prize unrealized without the
reorder". The V3 ON-arm numbers stand as evidence (d4 844.5, d16 944.5 on
the w1-port tree). V3 OFF-arm boot was mid-launch when the pivot landed
(bayes's 12:22 "idle" read caught the teardown/load gap; box was NOT idle).

W4 = option-6 (probs bwd reorder) on the staged chain: trainers 855d8d61 /
bridge d92774fa / mcore 70710d116 (clean; B/F gate stack parked Mac-side,
snapshot e1e46818… the restore path). Both gates verified in source: W1
(BT_MOE_PROBS_A2A_COMM, 12 refs) + option-6 (BT_MOE_PROBS_BWD_REORDER, armed
line :758 with an armed=NO fallback :763). W1 is VALIDATED (V0-V2 passed), so
co-arming both gates on boot 1 is compliant with never-co-armed-first-boot.
Boot 1 = W1+option-6 both ON + canonical env + BT_PROFILE_RANKS=0,8 (for the
ON-arm d16 traced window). Hard gate: BOTH armed-line families x16. Then d2
canary (watch gn closely — the reorder changes backward order). A/B: OFF =
W1-armed only (fresh boot), ON = W1+option-6; d4 x2 per arm, win floor +2%
mean-vs-spread; d16 x2 on clean d4 pairs.

## 2026-08-14 ~13:2x-13:5x CDT — W4 (option-6) ON-arm complete; OFF arm booting

W4 tree: option6-staged chain (trainers 855d8d61 / bridge d92774fa / mcore
70710d116 clean; B/F gate stack parked Mac-side). Boot-1 hard gate PASS: W1
armed x16 + option-6 armed x16 ('probs sort split + seq-bump + deferred
reverse wait'), zero armed=NO refusals. d2 canary PASS (kolmogorov-
adjudicated): 12.3256/12.3142/12.3206 in band, gn 0.40-0.48, NO drift — the
reorder's backward-order change is gn-clean on first contact.

ON-arm (W1+option-6, option6-staged tree): d4 814/846 (mean 830); d16
918/956 (mean 937); peak 143/162 GiB. Traced d16 window captured (rank0
2.8GB + rank8 3.0GB, shared FS + Mac pull in flight) for jacobi's re-size of
the exposed bwd probs-grad class. OFF arm (W1-armed only, option-6 unset)
booting fresh on the same tree — the A/B decider. Win floor +2%
mean-vs-spread vs the fresh OFF arm.

## 2026-08-14 ~13:5x-14:3x CDT — W4 (option-6) A/B COMPLETE: no win realized

OFF arm (W1-armed only, option6-staged tree, fresh boot): d2 canary PASS;
d4 849/851 (mean 850, spread 2); d16 912/960 (mean 936, spread 48).
ON arm (W1+option-6): d4 814/846 (mean 830, spread 32); d16 918/956 (mean
937, spread 38); canary PASS (gn clean). Traced d16 window captured (rank0
2.8GB + rank8 3.0GB, shared FS + Mac pull).

READS: d4 Δ -2.4% but |Δ| 20 < on-arm spread 32 → perf-indistinguishable;
warm class 846 vs 851 = -0.6%. d16 Δ +0.1%, |Δ| 1 < spread 48 →
perf-indistinguishable; warm 956 vs 960 = -0.4%. **option-6 on top of W1
realizes no measurable win at either rung tonight.** The lever model's
-3.4..-5.1s/step is NOT realized. Ship-decision picture: the W1+option-6
stack runs ~830-850 d4 / ~937 d16 vs the B/F class 920/1050.5 — W1 is a net
cost on this topology at these shapes and option-6 does not recover it.
kolmogorov adjudicates + NOTEBOOK; jacobi's trace read explains where the
prize went. Box holds the OFF boot hot.

## 2026-08-14 ~14:1x CDT — P4/S2 (B/F numerics parity) FAIL — STOP + escalate

Succession: bayes → turing (orchestrator); doppler keeps driving the soak
through S3; kolmogorov adjudicates. (Fleet note: the W1c/W2 docs call this
lane "lovelace" — same doppler session.)

Tree restore for P4 verified: 73c24b00 + TF32 patch BYTE-EXACT (applied-diff
sha256 f503c9bf… == the Mac reference; kolmogorov's grep-count>=2 line was
miscalibrated — the canonical patch has exactly ONE env-var occurrence; the
sha256-of-applied-diff is now the authoritative line, spec updated). mcore
gate-stack sha e1e46818… exact; wheel 1.27.0.

S2 (fresh boot A B/F-OFF, fresh boot B B/F-ON hard-gate x16, full 9-datum
set, forward_backward, no optim_step): **PARITY FAIL** — loss rel 9.113e-05
(bar 1e-6), logprobs max 3.495 (bar 1e-3), pervasive (87-98% of tokens in
every datum >1e-3, magnitude 1.7-3.5). SAME signature as the W2 executor
nondeterminism. STOP fired per the falsifier table. Escalated to kolmogorov
with the decisive localization proposal: B/F-OFF vs B/F-OFF self-compare
(two fresh boots) — separates a B/F cache bug from conventional-path
boot-to-boot nondeterminism (the fix-campaign's own cells agreed only to
~5e-5, already above the 1e-6 loss bar). S1 soak BLOCKED pending
adjudication; jacobi's R2 gap on hold (notified). Both S2 JSONs Mac-side
(~/perf_profiles/lps-1062/incoming/parity_ship-bf-{on,off}.json). Box holds
the B/F-ON boot armed-idle.

## 2026-08-14 ~15:2x CDT — S2 NOISE MATRIX: branch (b) — conventional-path nondeterminism, B/F EXONERATED

The S2 parity FAIL triggered a provenance scare + a decisive re-measurement.
turing's provenance cross-check (hertz caught it): kolmogorov's first
"within-boot control" ran on W2-ERA EXECUTOR files (parity_overlap-on-r1/r2/r3)
— a scope mislabel; the S2 files (parity_ship-bf-*) are the CONVENTIONAL path.
Verified: files distinct (mtimes 16:16 vs 19:10/19:27 UTC), no collision, all
6 Mac copies sha256-byte-exact vs box. turing's H2 env check: boots A/B
env-symmetric except the 2 gate vars (/proc-verified), byte-exact tree, same
wheel — precision-mismatch ruled out.

THE NOISE MATRIX (kolmogorov's prereg, turing's order, stamped files
<label>_wprm693_<utc-ts>.json, sha256-verified pulls):
| cell | design | max-abs | pervasive% | loss-rel |
| N1a | within-boot OFF (A2 prim vs r1) | 3.716 | 95.7% | 7.11e-05 |
| N1b | within-boot OFF (A2 r1 vs r2) | 5.455 | 95.7% | 3.18e-05 |
| N3 | cross-boot OFF (A vs A2) | 4.270 | 95.7% | 1.58e-04 |
| N4 | S2 off-vs-on (banked) | 3.495 | 95.7% | 9.11e-05 |
Branch-(b) rule: N4 <= max(N1..N3) on max-abs ✓, pervasive% in range ✓,
loss-rel within spreads ✓ — ALL MET. **The conventional PP2/CP8/EP8 path is
itself run-to-run nondeterministic at the 1e-6/1e-3 bars; the S2 'failure' is
the path's intrinsic floor, NOT a B/F cache bug. B/F EXONERATED.**
REFRAME (for the record): the conventional floor (3.7-5.5) is the SAME
magnitude class as the W2 executor's (5.098/6.343) — the executor's 'race'
and the conventional nondeterminism are ONE phenomenon: the DSA learned
indexer's discontinuous top-2048 KV selection flipping borderline per-token
choices run-to-run, schedule-independent. The 1e-6/1e-3 parity bars exceed
this stack's demonstrated floor at the mission topology. S1 soak gate =
turing's go after kolmogorov's provisional ruling. Box holds the A2 (B/F-OFF)
boot armed-idle.

## 2026-08-14 ~15:1x CDT — S2 CLOSED (branch b, B/F exonerated, PR #26 HOLD lifts); S1 GO

kolmogorov FINAL provisional ruling: branch (b) CONFIRMED (all three
conditions met); B/F EXONERATED; PR #26 HOLD lifts. The conventional path's
run-to-run floor is now measured (the noise matrix). S2's 1e-6/1e-3 bars
exceed the stack's floor at the mission topology — the parity claim is
re-worded noise-relative (no 1e-6/1e-3 claim imported anywhere).
turing: S1 GO. S1 boot (ship stack, B/F armed) launching; opens with the
parity driver x3 (N2 cell, stamped, pre-step-1, excluded from soak reads),
then the 60-step d16 soak (mem-poller 2s cadence). jacobi's R2 rides an idle
gap ~30-60 min in (ping with job id wprm693 at steady state). Leg-boundary
reports to turing + kolmogorov + hertz CC.

## 2026-08-14 ~15:5x-16:4x CDT — S1 SOAK COMPLETE (60/60); S3 BLOCKED (save hang + export gate)

S1 soak COMPLETE: 60/60 optimizer steps, 1091 t/s/GPU mean, step 120.1s,
mfu3x 9.9%. Trajectory clean (loss 12.316->11.980 smooth; gn 0.37->0.074);
throughput settled to the ~1080-1091 deep-settle class; B/F telemetry ZERO
fallback/miss/stale. TWO honest flags for adjudication: (1) memory crept
+2.8 GiB over 60 steps (178.9->181.7), 0.8 over the 2 GiB bar (slow creep,
poller max 183.8 within the 187-smi class); (2) main54 gn spike 0.97 + loss
jump, fully recovered main55, log clean = synthetic-data variance blip.
jacobi R2 rode clean (whale: per-step broadcast_object_list ~7.1s rank8).
Memory-state snapshot captured pre-S3; poller CSVs Mac-side at
runs/overnight_20260813_overlap_campaign/s1_soak_mem/ (kolmogorov's
creep-shape read).

**S3 BLOCKED, two separate causes:**
(a) SAVE LEG HANG: /save_state on the S1 boot hung — distcp write stalled at
1.3 GB, trainer API unresponsive, save in mcore save_checkpoint's closing
barrier. **SMOKING GUN: BT_SAVE_STATE_SYNC is read NOWHERE in the tree (grep
= 0 hits) — the standing F1 workaround is a NO-OP; megatron_config.py:325
hardcodes async_save=True unconditionally. The F1 async-save hang under CP>1
is LIVE (never exercised tonight because benches never call /save_state).**
(b) EXPORT LEG BLOCKED: /save_weights_for_sampler raises ValueError by design
under the mission config's weight_sync.type=disabled — needs a
weight_sync=local config delta + re-boot.

Box: trainer WEDGED on the hung save, needs a restart to recover. Reported
to turing; awaiting ruling on restart + whether the F1 async-save-hang fix
becomes a ship blocker. py-spy captures available.

## 2026-08-14 ~17:0x-17:3x CDT — S3 COMPLETE; P4 (S2→S1→S3) CLOSED; box idle-armed

Wedge snapshot banked Mac-side (38 files: py-spy all reachable ranks, distcp
listing, trainer log tail, /status + /operations timeout evidence rc=124/28)
at runs/.../s3_wedge_snapshot/ — a fresh F1 reproduction on soaked state.
Papercut pc_fb3a242ec978 (silent env-var no-op, silent-fallback family #4).
Tree delta: db5d1826 megatron_config.py hunk applied, applied-diff sha256
2f7dd3ea… == Mac reference byte-exact (BT_SAVE_STATE_SYNC now genuinely gates
async_save off); weight_sync=local config delta (trainer_config_s3.json).

S3 re-run (ship stack + hunk + weight_sync=local, canonical env + B/F gates +
BT_SAVE_STATE_SYNC=1), 5-step d4 burst first (adapters off zero-init):
- EXPORT: /save_weights_for_sampler completed (version 6); 1094-key complete
  GLM-5.2 LoRA set (the spec's "392 == reference" was the Qwen3-0.6B ref —
  GLM-5.2 = 1094 keys, internally consistent; shapes sane); L1 non-degeneracy
  PASS (547/547 B matrices non-zero); adapter_config correct (r=32/alpha=32).
- SYNC-SAVE: /save_state COMPLETED <5 min (the hunk makes the flag real; the
  in-process sync path ran; checkpoint written + uploaded via weight_sync
  local). The F1 workaround is now REAL on this tree.
- RESUME-CLEAN: post-save d2 window loss 12.2804 in band, gn 0.374, trainer
  continues.

**P4 CLOSED: S2 (parity, branch-b noise-relative) + S1 (soak, 60/60) + S3
(export + sync-save + resume-clean) all complete.** S3 artifacts Mac-side at
runs/.../s3_artifacts/. Box idle-armed on the S3 boot for Jack's stop/keep
(turing relays).

## 2026-08-14 ~17:4x CDT — LANE CLOSED (doppler/lovelace released)

P4 closed (S2 branch-b + S1 soak 60/60 + S3 all legs green). Lane handoff doc
written + delivered to turing: runs/overnight_20260813_overlap_campaign/
BOX_MECHANICS_HANDOFF.md (queue-race/resume-driver, devbox-up gotchas, the
squeue rule + lifecycle traps, bench/evidence discipline, Mac env fragility).
Box LEFT UP on the S3 boot (ship stack + F1 hunk + weight_sync=local) for
curie's blockK warm-pool probe (hertz adjudicates); idle-armed for Jack's
stop/keep after that. doppler out.
