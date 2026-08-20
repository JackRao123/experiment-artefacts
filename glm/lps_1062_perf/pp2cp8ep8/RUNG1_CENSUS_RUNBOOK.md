# RUNG 1 — MEMORY CENSUS BOOT (gate #1 of the Activation Placement Plan)

Owner: conway (handed off by banach 2026-08-20; originally godel).
Box: qed7z1w (2 nodes × 8 B300, ali) — **if the box is
reprovisioned, one `sed -i '' s/qed7z1w/<new-id>/g` on this file updates every
command; nothing else is box-specific.** Branch:
`jackrao/lps-1062-actplace` — pinned tip recorded at §1 (the branch moves as
the ladder's PRs land; the boot checks out the RECORDED commit, never the
moving tip). Plan of record: `ACTIVATION_PLACEMENT_PLAN.md` (same folder).

**What this run decides, in plain language:** the plan keeps each layer's
"glue" tensors (norms, residuals, router bookkeeping — dozens of small
tensors) on the GPU. The glue size was never measured; it was inferred by
subtraction as 0.44–0.89 GiB per layer per microbatch. At the low end the
GPU-storage plan fits comfortably; at the high end it is tight and the
biggest glue tensors must move to the CPU-offload bucket. This boot replaces
the guess with a number. It also banks the d2 baseline throughput and peak
memory that every later A/B (rung 2+) compares against.

**Rung 1 runs the baseline TWICE (banach/hilbert scope change, 2026-08-20):**
identical config, seed, and data order, giving (i) the memory census, (ii)
the d2 baseline, and (iii) the NOISE FLOOR — how much the SAME configuration
varies between identical runs. The floor is what rung 2's numerical gate is
judged against (the base path is intrinsically nondeterministic; "did
selective recompute change the math" is only answerable relative to this
floor). **Do NOT import the campaign's measured per-token floor of 3.7–5.5 as
the bar — different box, different wheel; our own repeat pair IS the floor.**
The campaign range is only a plausibility sanity check: a floor wildly
outside it is itself a finding to report before proceeding.

**Why d2 is enough (and what it costs):** the 1F1B pipeline schedule holds at
most 2 microbatches in flight no matter how many we feed it, so d2 (2
datums × 131,072 tokens = 262,144 tokens/step = 2 microbatches per schedule
call) has the SAME peak memory as d16 (campaign-verified). d2 throughput is
NOT comparable to the record numbers (pipeline bubble ~33% of the step at d2
vs ~6% at d16) — expected, fine here. The d16 headline at the END of the
ladder (rung 5) is a matched-pair RATIO on this tree, not a
record-comparable absolute (Jack: TF32 cancelled; no absolute-number
deliverable) — and it is not this rung's concern regardless.

---

## 0. Numbers of record (given — do not re-derive; check against these)

Per layer per microbatch-set, GiB: total eager saved set S_eager = 2.94;
full-recompute survivor S_ckpt = 0.19; core-attention internals 0.70; MoE
expert activations 0.75–1.15; dispatcher-combine 0.00 (VOID — see
configs/README.md; its bytes fall into glue); qkv-proj input
0.20; out-proj input 0.20; expert-fc1 0.00 (inert under LoRA); **glue =
0.44–0.89 BY SUBTRACTION — the gap this run closes.**

**Stage layout and in-flight counts (banach's correction of record, supersedes
the plan's "stage 1 = 40 layers × 2 in-flight = 80 sets" label):** under 1F1B
at PP2, a stage's in-flight microbatch count depends on its position — warmup
forwards = (stages − 1 − stage index).
- **Rank 0 (node 0) = FIRST stage:** embedding + 38 layers, holds **2**
  microbatches. With the dense/MoE overlay (mlp_layer_types = ['dense']×3 +
  ['sparse']×75 over 78 layers): layers 0–37 = **3 dense + 35 MoE** →
  35 × 2 = **70 MoE sets** + 3 × 2 = **6 dense sets** (76 total).
- **Rank 8 (node 1) = LAST stage:** layers 38–77 = **40 MoE, zero dense** +
  the loss/logits head, holds **1** microbatch → **40 MoE sets**, PLUS the
  loss/output-logit tensors (~5.1 GiB in bf16 per microbatch at 16,384
  tokens/rank × 154,880 vocab; ~10.1 GiB if the loss upcasts to fp32).
The two ranks are NOT two samples of the same quantity, and they bind on
DIFFERENT things (banach decision 2): **rank 0 binds on activation-set COUNT**
(drives offload sizing / PCIe bandwidth); **rank 8 binds on TOTAL PEAK** (the
logits/loss load — drives whether the keep-on-GPU bucket FITS). The plan's
"80 sets" matches neither rank; it errs ~5% toward safety (flagged to
hilbert). **Trust the measured per-rank numbers over the plan's label; report
disagreements.** The campaign pollers already showed node 1 peaking HIGHER
than node 0 at d2 (W1a: ~166 vs ~150 GiB smi) despite half the activation
sets — if the census confirms the last stage binds via the loss/logits load,
that is a headline finding: the plan's fit arithmetic is written against the
wrong stage.

Observed peaks @131k d16, last stage (pp_rank 1, node 1): ~183 GiB nvidia-smi
/ ~164 GiB torch-reserved; +20 GiB cold-pool burst; +60 GiB boot#1 warmup
transient; GPU cap 267.7 GiB. Same-topology d2 anchors from the mission tree
(W1a, wprm693): 138.8 GiB torch-reserved / 162.2 GiB smi (node-1 max), d2 ≈
703 tok/s/GPU — the expectation band for this boot (different tree; small
drift expected, big drift = investigate before proceeding).

**Pre-registered decision rule for gate #1 (banach decision 2 — the fit
question is evaluated PER RANK, against the rank that actually binds):**
For EACH stage leader (rank 0 and rank 8), report:
1. the measured post-plateau peak (poller plateau + torch HWM, labeled);
2. remaining headroom vs the 267.7 GiB cap with the ~20 GiB cold-pool burst
   accounted (effective ceiling ≈ 247.7 GiB);
3. the PROJECTED peak after the plan's change = current post-plateau peak +
   (sets_for_that_rank × measured glue-per-layer-mb) + ~5 GiB prefetch
   reserve. **Glue-ALONE is the primary projection (banach's adopted
   correction, 2026-08-20): under full recompute the 0.19 GiB inputs are
   already stored — that is what S_ckpt means — so the net-new storage from
   the plan's keep-on-GPU bucket is the glue row alone.** Report the
   glue-PLUS-input figure alongside it as a clearly-labelled conservative
   upper bound (+0.19 × sets = +13.3 GiB on rank 0, +7.6 GiB on rank 8) so a
   near-line verdict shows the pessimistic reading in the same table.
Then state plainly WHICH RANK IS TIGHTER and by how much. **If either rank's
projection lands within ~20 GiB of the effective ceiling (≳227.7 GiB), that
is a gate-#1 FAIL — the largest glue tensors move to the offload bucket. Say
so loudly; do not report a number and let the reader infer it.**
- glue ≲ 0.66 (census-mid) and both projections clear: plan proceeds as
  designed.
- glue ≈ 0.89 (census-hi) or a projection near the line: contingency
  activates (design change, banach's call).
- Report the measured glue row either way; the number, not a verdict spin.

---

## 1. Box bring-up

Box (provisioned by banach — do NOT provision another):

```
ssh training-job-qed7z1w-0.ssh.baseten.co   # leader:  node rank 0, global ranks 0-7,  pp_rank 0 (38 layers + embedding)
ssh training-job-qed7z1w-1.ssh.baseten.co   # worker:  node rank 1, global ranks 8-15, pp_rank 1 (40 layers + loss head)
```

Per-node env already set by the platform: BT_LEADER_ADDR, BT_NODE_RANK,
BT_GROUP_SIZE. Logs from Mac: `truss train logs --job-id qed7z1w --tail`.
Do NOT `truss train stop --job-id qed7z1w` without telling banach.

On the leader (all phases idempotent):

```bash
ART=/root/.cache/user_artifacts
PP2=$ART/lps1062_pp2
CLONE=$ART/trainers_main    # provisioner-seeded clone; confirm the path banach reports
mkdir -p $PP2/logs $PP2/traces $PP2/mem

cd $CLONE
git fetch origin jackrao/lps-1062-actplace
# PINNED checkout — the branch tip moves as ladder PRs land; boot the
# RECORDED commit, never the moving tip. Rung-1 pin: see below.
git checkout <PINNED_COMMIT>
git log --oneline -1                          # verify it matches the pin
git submodule update --init --recursive       # jacobi's gotcha: fresh checkouts have
                                              # EMPTY submodules (loops/, megatron-bridge);
                                              # build fails without this
```

**The pin: `405943b6`** (branch tip, 2026-08-20; supersedes the earlier
`0c794d36` pin, which is now four commits behind). The whole ladder — rung 1
baseline through rungs 3a/3b — runs ONE tree. The four commits added since
`0c794d36` are: `a102e596` (drops the `moe_combine` offload group — it
captures no bytes), `f130cad8` (Megatron-Bridge pointer bump carrying the
pinned-buffer pooling, in-allocator NUMA binding, boot page-placement
verification, valve telemetry, the conditional `attn_proj` guard relaxation
and the AbsorbedMLA `attn_proj` hook), `85900ca4` (offload hook-engagement
report from the first MoE layer), and `405943b6` (lint/format fixes). All are
inert for the full-recompute census boot. Record the actual booted commit in
the report.

**Tree note (conway, 2026-08-20):** trainers main was restructured on
2026-08-19 (#1027, "extract backend packages") — `server/` is now
`server-megatron-bridge/` plus `server-main`/`server-interface`/
`server-automodel`. `devbox-up`'s venv step still builds the OLD layout
(`make server-venv` → `server/.venv`), so on a box whose shared clone predates
the restructure the bridge venv must be built by hand:
`make megatron-bridge-venv CUDA_FLAVOR=cu13` (cu13 is the B300 flavor), then
`uv pip install --python server-megatron-bridge/.venv/bin/python --no-deps
nvidia-cudnn-frontend==1.27.0` from OUTSIDE the repo dir. Box-side script:
`$PP2/prep_actplace_tree.sh`.

Venv: the provisioner builds it. Verify, don't rebuild:

```bash
cd $CLONE/server-megatron-bridge && uv run --no-sync python -c \
  "import torch, cudnn_frontend; print(torch.__version__, torch.version.cuda)"
# cudnn-frontend must be 1.27.0 (the fixed-wheel rule):
uv pip list --no-index 2>/dev/null | grep -i cudnn   # or: uv pip show nvidia-cudnn-frontend
```

If the venv is broken/missing, the known-good repair on B300/ali (from
BOX_MECHANICS_HANDOFF): `apt-get install -y cuda-nvcc-13-0 cuda-libraries-dev-13-0`,
`CUDA_HOME=/usr/local/cuda-13.0`, `make megatron-bridge-venv`, then the
cudnn-frontend bump `uv pip install --no-deps nvidia-cudnn-frontend==1.27.0`
run from OUTSIDE the repo dir (repo pyproject pins silently constrain
otherwise). Report to banach before rebuilding — a rebuild is a tree-state
change.

## 2. DSA gate (must be green before any anchor run)

DSA = the model's sparse-attention path. The two regression files (5 tests
total) exercise the cuDNN indexer compile/topk paths on the box's GPUs:

```bash
cd $CLONE/server-megatron-bridge
uv run --no-sync pytest \
  tests/unit/dp_worker/test_cudnn_dsa_indexer_launch_stream.py \
  tests/unit/dp_worker/test_cudnn_dsa_indexer_topk.py -v
```

PASS bar: **5 passed** (campaign record: 5 passed / ~22–33 s). Any failure =
STOP, report verbatim. (Old-tree path was `server/tests/unit/dp_worker/...`;
the server restructure moved them to `server-megatron-bridge/tests/...`.)

**Piggyback (jacobi's ask, ~1 min, pure Python, no GPU):** the bridge config
tests have never executed on this branch (the bridge venv can't build on
macOS — linux-only nvidia-resiliency-ext). Run them during bring-up, before
the trainer boot:

```bash
cd $CLONE/server-megatron-bridge
uv run --no-sync pytest tests/unit/dp_worker/api/test_megatron_config.py -v -m 'not gpu'
```

Report back to jacobi (cc banach): pass/fail counts for the file overall and
specifically for class `TestActivationOffload` (expected 18 tests: 13 plumbing
+ 5 review-follow-up/valve). This is also a free tree-sanity check before the
boot.

## 3. Config (unchanged full-recompute golden config)

Boot artifact (staged from the campaign dir, Mac side):

```bash
# Mac:
cd ~/Documents/trainers/experiment_artefacts/glm/lps_1062_perf
scp pp2cp8ep8/configs/trainer_pp2cp8ep8_131k.json pp2cp8ep8/configs/trainer_server.json \
    training-job-qed7z1w-0.ssh.baseten.co:$PP2/
# (also stage tools: profile_driver_new.py, mfu.py, poll_gpu_mem.sh from tools/)
```

The JSON is the mission config: GLM-5.2-FP8, max_seq_len 131072, TP1/PP2/CP8/
EP8/ETP1, LoRA r32/a32, weight_sync disabled, **no `recompute` key →
granularity defaults to "full"** (verified: `RecomputeConfig.granularity =
"full"` default, models/src/loops_models/control.py:94-104 on the branch).
The same topology is registered on the branch at
`models/src/loops_models/model_configs/trainer_configs.py`
(`Model.GLM_5_2_FP8` / `GpuType.B300` / `SeqLen.S131K`) — that registry entry
is the SDK-facing record; the JSON above is the box boot artifact.

Box side:

```bash
cp $PP2/trainer_pp2cp8ep8_131k.json $ART/trainer_config.json
cp $PP2/trainer_server.json       $ART/trainer_server_config.json
export BT_TRAINER_CONFIG_PATH=$ART/trainer_config.json
export BT_TRAINER_SERVER_CONFIG_PATH=$ART/trainer_server_config.json
# persist for the launch shell:
printf 'export BT_TRAINER_CONFIG_PATH=%s\nexport BT_TRAINER_SERVER_CONFIG_PATH=%s\n' \
  "$ART/trainer_config.json" "$ART/trainer_server_config.json" > $PP2/trainer_env.sh
```

**Staged-file integrity (CPFS close-to-open quirk — files scp'd to the shared
FS can read back as all-NULs from the SIBLING node):** sha256 every staged
file on the leader AND via `srun --overlap -N2 -n2` on both nodes; re-stage
from Mac on any mismatch before booting.

## 4. Environment block (canonical, adapted to this branch honestly)

```bash
# Ship NCCL env — verbatim from the campaign notes (lps_1062_perf/NOTEBOOK.md):
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_SPLIT_DATA_ON_QPS=1
export NCCL_NCHANNELS_PER_NET_PEER=8
# Multi-rank kineto tracing (the cherry-picked 7998c14e): both PP stage leaders.
export BT_PROFILE_RANKS=0,8
# OFFLOAD RUNGS ONLY (2c, 3a/3b/3c — NOT the rung-1/2a baseline boots):
# NVTE_CPU_OFFLOAD_V1=1 must be in the LAUNCHER env. RESOLVED (jacobi):
# absent = fails LOUD (the bridge validator raises at boot before model
# build) — but the silent trap is a TIMING one: TE latches the variable at
# IMPORT time while the validator reads it lazily, so a worker-side set AFTER
# TE imports PASSES validation while TE silently keeps the old code path.
# LAUNCHER ENVIRONMENT ONLY. carnot's boot log line reports BOTH the env
# value and TE's latched value ('env=1 latch=0' catches the trap at boot) —
# reading and judging that line is a 2c pass/fail item (see §8b).
# The env family is verified COMPLETE (jacobi): no other undocumented
# required vars in the offload path; only carnot's two documented
# BT_OFFLOAD_* NUMA controls besides this one.
# export NVTE_CPU_OFFLOAD_V1=1
```

**Boot-time env verification (standing, every boot):** echo the resolved env
into the run log so every artifact carries proof of what was actually set —
a forgotten export that silently disables offload would make every rung-3 arm
read as "offload does not help" when offload never ran (the worst failure
mode available to this ladder):

```bash
{
  echo "=== boot env $(date -u +%FT%TZ) ==="
  env | grep -E '^(NCCL_|BT_|NVTE_)' | sort
} | tee -a $PP2/logs/boot_env.log
```

(jacobi is also checking for OTHER undocumented env vars in the same family —
if there is one, there may be three; fold any findings into this block.)

Deliberately NOT set, with reasons (verified against the branch, not assumed):

- `BT_TF32_LM_HEAD=1` — **CANCELLED, not deferred (Jack, direct, 2026-08-20:
  "TF32. I don't want that.").** No TF32 port anywhere in the ladder; the
  whole ladder runs one consistent tree with no TF32. Rung 5 is the matched
  d16 pair ON OUR OWN TREE (full-recompute control vs phase-1 treatment) and
  the headline is the RATIO between them. **No absolute record-comparable
  throughput number is a deliverable: do not produce one, do not imply one,
  do not compare any figure to the 984 or 1089–1103 historical bands.** Every
  absolute tok/s figure we report is context only, on our own tree,
  explicitly non-comparable. (What was the "no-TF32 caveat" is no longer a
  caveat — it is simply the environment.)
- `BT_SAVE_STATE_SYNC` — **absent on this branch** (the Q1 queue item is
  unmerged; the var is a silent no-op). RULE: never call `/save_state` at all
  (the async save path hangs under CP>1 — finding F1).
- `BT_SKIP_WARMUP` — not used here. The campaign set it on near-ceiling memory
  boots to dodge the +60 GiB warmup transient; rung 1 runs full recompute at
  ~164–183 GiB vs the 267.7 cap, far from the ceiling, and the trainer warmup
  doubles as the kernels-warm pass. The warmup transient is handled by the
  READ RULE below (plateau read, never the whole-run max).

## 5. Boot 1 — launch, config dump, parity floor legs, smoke

**Order matters (the fresh-weight rule):** the parity legs are FORWARD-ONLY
(no optim_step — weights never move) and must run on a FRESH boot, before any
driver window. LoRA B is zero-init, so a fresh boot's forward == the base
model exactly; that is what makes the cross-boot noise-floor cell (boot 2)
comparable. The smoke and census runs DO step the optimizer (weights move) —
they come after the parity legs.

Stale-process sweep first (stop_trainer.sh can leave orphaned workers holding
GPU memory — an orphan OOMs the next boot for a phantom reason), and the
SQUEUE RULE: `squeue` must show ZERO devbox_trainer jobs before dispatching:

```bash
squeue
srun --overlap -N2 -n2 bash -c \
  'pkill -f "[d]p_worker.main" 2>/dev/null; sleep 2; pgrep -af "[d]p_worker" || echo clean; \
   nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
```

Launch (backgrounded waits only — never block the session on a wait/sleep
loop; `wait_trainer_health.sh` exists for this):

```bash
source $PP2/trainer_env.sh   # + the env block from §4
bash $ART/.devbox_up/start_trainer.sh &
bash $ART/.devbox_up/wait_trainer_health.sh &
```

Readiness signal: `/status` reports world_size 16, TP1/PP2/CP8/EP8/DP1, seq
131072. (Under BT_SKIP_WARMUP there is no READY banner — not applicable here,
warmup is on; expect the banner after "kernels warm".) Weight load is 13–20
min on a warm cache.

### 5a. At-boot config dump (banach decision 3 — cheap, read it off the live model)

The DSA indexer is per-layer either 'full' or 'shared' (`indexer_types`), and
the code default (all 'full') would contradict PR #1070's stated 38/40-boundary
rationale, so the real pattern must come from the checkpoint's config. Dump
the RESOLVED lists and report per-stage counts:

```bash
# 1) boot log (mcore prints the resolved model/config at startup):
grep -o "indexer_types[^,]*" $PP2/logs/trainer_srun.log | head -2
grep -o "mlp_layer_types[^,]*" $PP2/logs/trainer_srun.log | head -2
# 2) cross-check against the HF checkpoint config.json:
python3 - <<'EOF'
import glob, json, os
hf = os.environ.get("HF_HOME", "/root/.cache/user_artifacts/team_artifacts/huggingface")
p = glob.glob(os.path.join(hf, "hub", "models--zai-org--GLM-5.2-FP8", "snapshots", "*", "config.json"))[0]
cfg = json.load(open(p))
it = cfg.get("indexer_types"); mt = cfg.get("mlp_layer_types")
print("config:", p)
if it:
    print("indexer_types full/shared: all=%d/%d stage0(0-37)=%d/%d stage1(38-77)=%d/%d"
          % (it.count("full"), it.count("shared"),
             it[:38].count("full"), it[:38].count("shared"),
             it[38:].count("full"), it[38:].count("shared")))
else:
    print("indexer_types: ABSENT (code default applies — flag it)")
if mt:
    print("mlp_layer_types dense/sparse: all=%d/%d stage0 dense=%d stage1 dense=%d"
          % (mt.count("dense"), mt.count("sparse"),
             mt[:38].count("dense"), mt[38:].count("dense")))
else:
    print("mlp_layer_types: ABSENT (flag it)")
EOF
```

Report: per-stage full/shared indexer counts (expected from the topk-sharing
rule: a layer computes its own topk iff 1-based layer ≤ 3 or (layer−3)%4==0)
and the dense/MoE map (expected: 3 dense + 35 MoE on stage 0; 40 MoE on stage
1). Why it matters: core attention is the one bucket kept as recompute, and a
'full' indexer layer saves/recomputes a different amount than a 'shared' one —
a lopsided split across the stage boundary makes the per-layer census average
misleading. If the resolved pattern contradicts the expectation, say so before
the census run.

### 5b. Parity floor legs ×2 (within-boot cell, FRESH weights)

The forward-side noise floor, per the S2 noise-matrix protocol
(`runs/overnight_20260813_overlap_campaign/S2_NOISE_MATRIX_PREREG.md`). The
parity driver sends a FIXED 9-datum mixed-length set (262,032 real tokens,
deterministic seed) and records loss + per-token logprobs; two identical legs
on the same boot give the within-boot floor. Labels are stamped with job-id +
UTC (the parity-collision lesson — the JSON carries the label):

```bash
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
python3 $PP2/parity_driver.py --label rung1A-qed7z1w-p1-$STAMP
python3 $PP2/parity_driver.py --label rung1A-qed7z1w-p2-$STAMP
python3 $PP2/parity_driver.py --compare \
  /root/.cache/user_artifacts/lps1062_bench/parity_rung1A-qed7z1w-p1-$STAMP.json \
  /root/.cache/user_artifacts/lps1062_bench/parity_rung1A-qed7z1w-p2-$STAMP.json
```

Record the compare output's three statistics: per-token max-abs diff, %
tokens >1e-3, loss rel spread. (The driver's built-in 1e-6/1e-3 tolerances
are the OLD absolute bars — we do NOT judge against them; we record the
statistics as the floor. Noise-relative only, per the plan.) If the driver
itself errors on this branch (datum format / API drift), STOP and report —
do not improvise a patch mid-run.

### 5c. Smoke (cheap) BEFORE the census run

**Smoke (rides the SAME boot — do not waste it):** prove the branch builds,
the topology is up, and allocator snapshots land where expected, with ONE
d1-class driver pass:

```bash
python3 $PP2/profile_driver_new.py --label rung1smoke-qed7z1w-d1-$(date -u +%Y%m%dT%H%M%SZ) \
  --datums 1 --control-repeats 1
```

Then verify, on BOTH nodes:

```bash
srun --overlap -N2 -n2 bash -c \
  'ls -la /tmp/checkpoints/profiles/latest/memory/ | head -12'
```

SMOKE PASS BARS (all four, written down before running):
1. Driver completes: warmup + traced + 1 control, loss in the 12.2–12.4
   canary band.
2. `/tmp/checkpoints/profiles/latest/memory/memory.rank{0..7}.pickle` on
   node 0 and `memory.rank{8..15}.pickle` on node 1 — 16 files, nonzero.
3. Kineto traces for ranks 0 AND 8 under
   `/tmp/checkpoints/profiles/torch_trace/` (rank 8's is ON NODE 1).
4. Pickle sanity (leader): nonzero segments + device_traces —
   `python3 -c "import pickle; s=pickle.load(open('/tmp/checkpoints/profiles/latest/memory/memory.rank0.pickle','rb')); print(len(s['segments']), sum(len(t) for t in s['device_traces']))"`

Any miss = STOP: fix the path/config problem HERE, not after the full run.
Smoke pickles are disposable (the census run's memory_profile/start wipes the
dir on each node).

## 6. Census run A + repeat run B (the real measurement, twice)

Start the per-GPU memory pollers (nvidia-smi reserved, the OOM-relevant
metric — NEVER mix it against the torch-reserved metric from /status; they
differ by ~12 GiB of NCCL/driver overhead):

```bash
srun --overlap -N2 -n2 bash $PP2/poll_gpu_mem.sh $PP2/mem &
```

Then the driver. **Run length (the plateau rule):** torch-reserved creeps
~+21 GiB over a run's early steps before flattening; a window-1 read
understates the peak. Campaign evidence: at d16 the creep ran steps 6→9 and
was flat through step 16 (MEMORY_LEG_DECISION, fourth note); the d2-cadence
rule of thumb is 4–6 mains past warmup suffice (BLOCKK_WARMPOOL_PREREG). The
driver's windows are steps: warmup = step 1, traced = step 2, controls = steps
3…. **`--control-repeats 10` → 12 steps total**, so the last two controls are
steps 11–12 — comfortably past the step-9 plateau declaration with margin,
and satisfying the read rule (plateau = poller-max over the LAST TWO mains,
which must agree within ~2 GiB):

```bash
python3 $PP2/profile_driver_new.py --label rung1A-qed7z1w-d2-$(date -u +%Y%m%dT%H%M%SZ) \
  --datums 2 --control-repeats 10
```

(~23 s/step at d2 → ~5 min of controls; the traced step adds the kineto tax
and the stop-dump can take minutes. All normal.)

**Then the repeat — run B, identical config/seed/data order (the driver
re-seeds per invocation, so run B's window k sees the same data as run A's
window k).** Run B is the within-boot repeat: its memory read is a second
plateau sample, and its per-window loss/gn trajectory against run A's is the
stepping-side noise floor:

```bash
python3 $PP2/profile_driver_new.py --label rung1B-qed7z1w-d2-$(date -u +%Y%m%dT%H%M%SZ) \
  --datums 2 --control-repeats 10
```

PASS BARS (pre-registered; apply to BOTH runs):
1. Canary: every window's loss in 12.2–12.4, grad norm comparable to the
   fixed-wheel 0.36–0.49 band. Drift > 5e-3 = STOP + verbatim report.
2. Plateau declared: poller-max over the last two controls agrees within
   ~2 GiB on both nodes. If not, the run hasn't plateaued — extend (re-run
   with more controls) rather than read early.
3. 16 fresh pickles + rank-0/rank-8 traces present per run (same checks as
   §5; pull or rename run A's before run B — the memory_profile/start of a
   new run wipes the pickle dir on each node, and runtime_profile/start
   CLEARS the box trace dir — the campaign's pull-immediately rule).
4. Baseline banked PER RUN: control-window tok/s/GPU (mean + spread), torch
   HWM from the driver's final /status, poller plateau per node. Record all
   three with their metric labels — the torch/smi distinction is a standing
   rule.
5. Repeat-pair floor recorded: per-window |loss A−B| and |gn A−B| at matched
   window positions, plus the parity-leg statistics from §5b. Plausibility
   sanity check ONLY against the campaign's 3.7–5.5 per-token floor — wildly
   outside = report before proceeding, not a bar trip.

Note on the traced step: it is step 2, BEFORE the plateau, and kineto buffers
can inflate that one step's footprint. The plateau read comes from the
post-trace control steps via the poller series; if the whole-run torch HWM
sits above the plateau, say so and report both (trace-step spike vs plateau).
This is the honest handling of the campaign's "no trace tax on a memory read"
caveat against the plan's requirement to trace ranks 0+8 on this boot.

## 6b. Boot 2 — the cross-boot floor cell (same config, fresh boot)

Rung 2 boots SEPARATELY from rung 1, so the floor that adjudicates it must
include boot-to-boot variation — the S2 noise matrix's central lesson is that
within-boot and cross-boot floors are different cells, and both are needed.
Boot 2 is the SAME baseline config, freshly booted. It is NOT a second census
— no pickles needed beyond the driver default, no traces (leave
BT_PROFILE_RANKS unset on this boot to keep it trace-tax-free):

1. Sweep + squeue check + boot (same as §5). READY.
2. One parity leg (fresh weights): `parity_driver.py --label
   rung1B2-qed7z1w-p1-$STAMP`. Cross-boot cell = compare vs boot 1's p1:
   `parity_driver.py --compare parity_rung1A-...-p1-... parity_rung1B2-...-p1-...`.
3. One d1 driver step (mirrors boot 1's smoke, so the driver runs on both
   boots sit at the same weight offset — matched-step comparability for the
   gn trajectory).
4. Driver run A′, d2, 6 control repeats (8 steps): matched-step gn/loss
   trajectory vs run A's first 8 windows = the cross-boot stepping floor.
   Memory reported, not a gate (6+2 steps may not fully plateau — label it).

Boot 2 cost: one weight load (~13–20 min warm cache) + ~5 min stepping. If
box time forces a choice, boot 2's parity leg is the last thing to cut — but
cutting it means rung 2's numerical gate is judged against a within-boot-only
floor, which UNDERSTATES the quantity under test; that trade-off is banach's
call, made explicitly, not by omission.

## 7. Evidence collection (immediately — the box can die; profile dirs are volatile)

```bash
# Mac side — pull from BOTH nodes (paths are pod-local):
STAMP=qed7z1w-$(date -u +%Y%m%dT%H%M%SZ)
for N in 0 1; do
  scp -C training-job-qed7z1w-$N.ssh.baseten.co:'/tmp/checkpoints/profiles/latest/memory/memory.rank*.pickle' \
    ~/perf_profiles/lps-1062/pp2cp8ep8/rung1_census/node$N/
  scp -C training-job-qed7z1w-$N.ssh.baseten.co:'/tmp/checkpoints/profiles/torch_trace/*.pt.trace.json' \
    ~/perf_profiles/lps-1062/pp2cp8ep8/rung1_census/node$N/
done
scp training-job-qed7z1w-0.ssh.baseten.co:'/root/.cache/user_artifacts/lps1062_bench/rung1census-*.json' \
  ~/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/pp2cp8ep8/results/
# poller CSVs:
scp training-job-qed7z1w-0.ssh.baseten.co:'/root/.cache/user_artifacts/lps1062_pp2/mem/mem.*.csv' \
  ~/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/pp2cp8ep8/results/rung1_census_mem/
```

sha256 every pulled file box-side vs Mac-side (the CPFS/relay quirk). Record
the bench JSON label stamp (job-id + UTC) — the parity-collision lesson.

## 8. Analysis — turning pickles into the glue number

Tooling status (checked 2026-08-20): `analysis/notebooks/recompute_cost_analysis.ipynb`
holds the existing evidence as HARDCODED tables from trace reads — it has NO
pickle parser. No pickle-parsing script exists in the campaign tools either.
**I will write `tools/memory_census.py`** (new), using the snapshot format per
the `analyze-torch-traces-loops` skill (PyTorch memory_viz is the fallback
viewer):

1. Per rank, unpickle `memory.rank<N>.pickle`: `segments[].blocks[]` (state,
   size, alloc frames) + `device_traces[]` (alloc/free events with python
   stacks; `context="alloc"` means frees have no stack — attribute on allocs).
2. Replay the device-trace event stream to reconstruct live-bytes-over-time;
   take the composition AT THE PEAK INSTANT of the final (post-plateau) step.
   Attribute every live allocation to a module bucket by its alloc stack's
   file names: `experts.py`/`moe` → MoE block; `token_dispatcher.py` →
   dispatch/combine; `absorbed_mla.py`/`attention.py` → projections/attention;
   `dsa.py` → core-attn internals; norms/residual/router files → glue;
   loss/lm-head files → loss/logits (last stage only); everything else →
   base (params/optimizer/NCCL/buffers).
3. Report the two stage leaders SEPARATELY, labelled by rank, with the
   in-flight count AND layer-type overlay stated: rank 0 = 70 MoE sets + 6
   dense sets; rank 8 = 40 MoE sets + loss/logits. Per-bucket GiB/layer/mb
   uses the RIGHT denominator per bucket — MoE buckets (moe_act,
   core-attn, attn_proj, glue) over the MOE set count only (70 for rank 0,
   40 for rank 8): the 3 dense layers have no expert block or dispatcher, and
   a blended denominator drags the MoE figure down ~8% and corrupts the
   offload sizing (banach decision 4). Rank 8 is uniform-MoE → the CLEANEST
   per-MoE-layer number in the experiment; rank 0 cross-checks it and its
   residual isolates the dense-layer cost. Never average the two ranks.
   THE GLUE ROW IS THE DELIVERABLE — measured, not subtracted.
4. Cross-checks: live-at-peak total vs poller plateau vs torch HWM (three
   metrics, labeled, never mixed); the first-stage vs last-stage peak
   comparison is a headline output (see §0 — the campaign pollers already
   showed node 1 higher; confirm or refute from the allocator snapshots).
5. The gate-#1 projection per §0's decision rule: per rank, current
   post-plateau peak + sets × measured glue + ~5 GiB prefetch reserve, vs the
   ~247.7 GiB effective ceiling. Which rank is tighter, by how much, and the
   FAIL line (≳227.7 GiB projected) if either trips it. Both projections
   (glue-alone primary, glue+input conservative bound) in the same table.
6. **DSA topk-selection stash — its OWN census row, never in glue (banach
   hazard #2):** leader layers write a [1, 16384, 2048] integer selection
   tensor (268,435,456 B in int64; 134,217,728 B if int32) into a plain
   Python dict on the per-microbatch object — OUTSIDE autograd's saved-tensor
   system, never popped within the microbatch. Aggregate ~2.4–5.4 GB/rank;
   left in the residual it inflates glue ~4–16% on rank 0 — material against
   the 20 GiB fail margin. `memory_census.py` isolates these BY EXACT SIZE as
   the `dsa_topk_stash` row, with the count at peak (expected: stage 0 = 11
   leaders × 2 in-flight = 22; stage 1 = 10 × 1 = 10 — the §5a indexer_types
   dump overrides if it disagrees) and the top alloc sites printed for
   eyeball verification (expect dsa.py/indexer call sites; anything else =
   investigate the size-alias before trusting the row). **Report note
   (mandatory):** this cost is paid TODAY under full recompute — the dict's
   lifetime does not depend on recompute granularity — so it is NOT a new
   cost of the plan and must not be counted against the change. If the tool
   cannot isolate them cleanly, say so and report glue with an explicit
   stated uncertainty of that magnitude rather than a clean number.
5. `max_entries` caveat: the driver accepts the server default (100,000
   events). If the final step's event window is not fully covered in
   device_traces, re-run the stop/start with a bigger max_entries via a direct
   `POST /memory_profile/start {"max_entries": ...}` — the API accepts it
   (server.py StartMemoryProfileDetails); note it in the report if needed.

## 8b. Rung 2 PRE-REGISTRATION — RESCOPED (hilbert-approved 2026-08-20)

**The feasibility flag was CONFIRMED (banach re-derived it independently):
selective-only at 131k cannot run — rank 0 projects to ~292 GiB vs the 247.7
effective ceiling (OOM by ~45), rank 8 to ~248 (on the line). Selective
recompute and the offload are a PACKAGE, not two independent wins; there is
no selective-only configuration at 131k and no "just take the recompute win"
fallback. (The block+K dial is THE fallback and a different mechanism.)
Package framing rule for ALL reporting: never phrase anything as "the
recompute change gets us X and offload adds Y".**

### Rung 2a — selective-recompute-only at **32k**, no offload (the rescope)

Both of rung 2's jobs survive at 32k: whether selective recompute changes the
math is a per-layer property that does not care about sequence length, and the
memory gate gets BETTER — combined with rung 1's 131k census it gives two
points on the memory-vs-sequence-length line. 32k (not 64k) exactly matches
the one prior offload trial, so we inherit a comparable reference point.

Configs (staged from the campaign configs dir):
- Baseline: `configs/trainer_pp2cp8ep8_32k.json` (NEW — the 131k base at
  max_seq_len 32768, no recompute key = full recompute).
- Arm: `configs/trainer_pp2cp8ep8_32k_selective_novpp_plain.json` (existing;
  selective, no offload).

Structure mirrors rung 1 at 32k (d2 = 2 × 32,768 = 65,536 tokens/step; same
1F1B in-flight semantics; same 12-step plateau rule; steps are ~4× cheaper):
1. 32k baseline boot 1: config dump → parity legs ×2 (FRESH weights) →
   driver runs A/B (d2, 12 steps each) — the 32k floor + the 32k base memory
   measurement.
2. 32k baseline boot 2: parity leg ×1 + d1 offset step + 8-step driver run —
   the 32k cross-boot cell.
3. Rung-2a boot (selective, 32k): parity leg (fresh) + driver run (d2, 12
   steps, memory_profile on).
Parity legs at 32k use `--max-datum-len 32768` (drops the datum set 9→6,
byte-identical content — the W2 shim precedent).

**Gate 1 — memory:** the band is written HERE from the all-rungs projection
table (§8c) BEFORE the 2a boot — standing rule: band first, boot second.
[BAND: TO BE WRITTEN FROM RUNG-1 CENSUS + 32k BASELINE.] Outside band = STOP
+ report.

**Gate 2 — numerical (noise-relative; NEVER an imported absolute bar):**
- Forward side (the sensitive detector): the 2a boot's parity leg vs the 32k
  baseline legs. Statistics: per-token max-abs, % tokens >1e-3, loss rel.
  Within the 32k floor (BOTH cells) on all three → PASS. Materially above
  every floor cell by a clear margin → FAIL: STOP + escalate (LPS-1063 was
  exactly this failure shape — silent mis-attention + gradient corruption, no
  crash, no memory anomaly). Ambiguous middle → conservative FAIL-leaning
  read per the S2 rule: the claim does not get the benefit of an unresolved
  gap; escalate with the full matrix published.
- Gradient side: AGGREGATE grad-norm only, matched-step trajectory vs the
  32k baseline runs. Within the floor's gn spread → PASS; beyond → STOP.
  (No per-parameter-group norms — they do not exist and hilbert explicitly
  does not want them built; the per-token loss vector dominates.)
- Canary: loss 12.2–12.4 + gn comparable per window. (At 32k the absolute
  values may sit differently than the 131k band — record actuals; a canary
  shift vs the 32k BASELINE is the signal, not the 131k band.)
- Precondition note (carnot, settled): shared-indexer layers construct no
  indexer module at all (attribute literally None) — they CANNOT recompute a
  selection; a missing dict entry raises a loud named error. Crash-not-silent,
  and the dict lifetime is outside autograd, so recompute granularity cannot
  affect it. This gate confirms a mechanism-level-sound path; it is not
  hunting a suspected break.

### Rung 2b — deliberate 131k selective-only OOM confirmation (CONDITIONAL)

Run ONLY if rung 1's census leaves the 131k selective-only projection
ambiguous. If the census predicts OOM as cleanly as the estimate does
(~292/~248 GiB), do not spend a boot proving what was predicted twice. If it
runs: pre-register the predicted OOM number BEFORE booting, boot selective-only
at 131k d2 with the poller on, and record where the OOM lands vs the
prediction.

### Rung 2c — full phase-1 offload config at 32k, pure BRING-UP pass (APPROVED)

The full phase-1 configuration (selective + offload [moe_act,
attn_proj] + valve + pool fix + carnot's hooks) at 32k, where memory is not
binding and boots are cheap. Rationale (banach): rung 3 arm 1 otherwise stacks
eight separate first-ever-hardware-boots (backpressure valve, pinned-pool fix,
in-allocator NUMA binding, placement verification, valve telemetry, the new
module name + relaxed validator, the unpermute hook, the projection-input
hook) onto the configuration where memory is tightest and every attempt is
most expensive — a failure there is over-determined and bisects eight
candidates at 131k prices.

**Preconditions (code, before the 2c boot):** carnot's dispatcher hook +
vendored vocabulary hunk, the projection-input hook, the pinned-pool
freeze-exception patch, and jacobi's two trainers commits (b6894e56,
0c794d36) — all landed on the pinned branch. Config: the 32k full-phase-1
variant (stage from jacobi's 32k offload configs; verify the module list
matches the phase-1 set).

**Pass criteria (ONLY these):** it boots; hooks engage where expected;
placement verification reports NUMA-local; valve counters report sane values;
nothing crashes. **Explicitly NOT a throughput claim** — 32k throughput says
nothing about the 131k win and must not be reported as if it does.

**hilbert addition 1 — the pool-fix comparison is a TRIPWIRE, not a gate:**
2c runs at exactly the sequence length where the prior trial measured a 30%
step-cost penalty from fresh pinned-host-buffer allocation every layer — a
genuinely length-matched, same-allocation-pattern read on whether the pool
fix recovers it. OBSERVE the step-cost delta vs the prior trial; **if the
pool fix does not visibly recover most of the 30%, STOP before any 131k arm
and investigate** — the pool premise is load-bearing for the entire design,
and if fresh-allocation cost is not what the prior trial lost, we do not
understand the overhead we are fixing. Do NOT report a throughput number from
this — it is a tripwire reading, not a measurement.

**hilbert addition 2 — 2c also smoke-runs the INSTRUMENTS** ("rung 3's runs
should never have to debug their own instruments"). Every instrument below is
new and unexercised on real output; at 2c each must PRODUCE its artifact and
get eyeballed once:
1. `memory_census.py` parses the real pickles, and its allocation-site
   printout names the expected sites — the size-signature aliasing concern's
   real-world test (if sites aren't the DSA/indexer call sites, investigate
   before trusting the row).
2. The trace classifier — `analysis/trace_tools/copy_exposure.py` (godel-owned;
   the pinned-copy class was added to hilbert's spec on top of the
   extract.py/analyze_1f1b.py/recompute_cost.py pipeline) — actually finds and
   classifies the pinned device-to-host and host-to-device copy events.
   Self-tested against the fe127_d16_rank0 reference trace BEFORE any box
   boot: reproduces the known classes exactly (a2a 28.55s, p2p 8.55s vs the
   plan's constants), reports the baseline's pinned-DtoH background (66k tiny
   copies, fully hidden — the offload arms read copy numbers as a DELTA
   against this background), and the cross-check branch behaves.
3. Traces land on BOTH ranks (BT_PROFILE_RANKS=0,8 exercised end-to-end).
4. carnot's boot-time placement verification log reports NUMA-local rather
   than nothing.
5. carnot's NVTE boot log line reports BOTH the environment value and TE's
   latched value — I read it and judge pass/fail: `env=1 latch=1` = pass;
   `env=1 latch=0` = the import-time latch trap fired = FAIL the boot (the
   export landed after TE imported; fix the launcher env, do not debug TE).
   carnot was told to format the line so it is unmistakable to someone who
   has never heard of the trap — if it is not unmistakable when I read it,
   that is itself a finding to report.
Any malformed or empty artifact = a 2c finding worth STOPPING on — much
cheaper to fix at 32k than mid-analysis of an expensive run.

**Mandatory caveat in any 2c report (hilbert's phrasing):** 32k de-risks
CRASHES and WIRING, not TUNING. The backpressure valve only really engages
under memory and bandwidth pressure that 32k does not create, and the
memory-placement penalty only appears under all-8-GPU saturation. A green 2c
means "the machinery works", never "the machinery is tuned".

## 8c. THE ALL-RUNGS PROJECTION TABLE (hilbert standing instruction)

**The FIRST derived artifact from rung 1's census is one projection table
covering ALL remaining rungs in a single pass** — 2a at 32k, every rung-3 arm
at 131k, and the d16 anchor. One recomputation from the measured base; every
subsequent gate inherits it. This prevents per-rung arithmetic drift (each
gate's band computed from slightly different assumptions, disagreement
noticed by nobody). The census tool's output feeds this table directly.

Per rank r ∈ {0, 8}, from the measured post-plateau base B_r (which already
includes the stored 0.19 GiB inputs × sets_r) and the measured per-MoE-layer-mb
census rows (glue g, moe_act m, combine c, projections p, core-attn k — all
GiB/layer-mb at 131k; 32k rows ≈ 131k rows × 32/131, the scaling assumption
is stated and checked against the 32k baseline boot's own measurement):

| rung | config | delta vs base, per rank | projected peak |
|---|---|---|---|
| 2a @32k | selective, no offload | sets_r × (S_eager³² − core³² − input³²) where S_eager³² ≈ 2.94×32/131 etc. | B_r(32k, measured) + delta |
| 3a @131k | selective + offload moe_act | sets_r × (g + c + p) [moe_act offloaded, core recomputed] | B_r(131k) + delta |
| 3b @131k | 3a + offload attn_proj | sets_r × g | B_r(131k) + delta |
| 3c @131k | 3b + offload attn_proj | sets_r × g | B_r(131k) + delta |
| d16 anchor | = 3c config, d16 | memory identical to d2 (1F1B caps in-flight at 2) — throughput is the headline, not memory | = 3c projection |

Each projection is checked against the ~247.7 GiB effective ceiling with the
FAIL line at ≳227.7, per rank, per §0 — glue-alone primary, glue+input
conservative bound alongside. The table is filled ONCE from rung 1's census
(+ the 32k baseline for the 2a row) and inherited by every gate.

## 8d. Forward notes for rungs 3a–3c (jacobi's config package + boot-env rules)

Run-configs live in the artefacts repo (JackRao123/experiment-artefacts) under
`glm/lps_1062_perf/pp2cp8ep8/configs/`, parse-checked against
TrainerControllerConfig at trainers 0c794d36:
- 3a (selective + offload moe_act):
  `trainer_pp2cp8ep8_131k_selective_offload_moe_act.json` — the FIRST config
  that runs at 131k at all: treat as a bring-up milestone, expect first-boot
  problems, budget for them.
- 3b (+ attn_proj):
  `trainer_pp2cp8ep8_131k_selective_offload_moe_act_attn_proj.json` — needs
  carnot's dispatcher hook + the vendored vocabulary hunk (trainers b6894e56)
  landed first, else the boot fails at provider.finalize().
- 3c (+ attn_proj):
  `trainer_pp2cp8ep8_131k_selective_offload_moe_act_attn_proj.json` —
  UNBLOCKED (attn_proj rule relaxed under core_attn recompute; trainers
  0c794d36). Same carnot-hunk boot dependency as 3b.

Boot-env rules for the offload rungs (jacobi, verified):
- `NVTE_CPU_OFFLOAD_V1=1` must be in the LAUNCHER env — TE latches it at
  import; a worker-side set is too late.
- carnot's NUMA allocator defaults to NUMA-local; `BT_OFFLOAD_NUMA_BIND=off`
  is the opt-out A/B switch — leave it UNSET for the real arms.

## 9. Standing rules (from the campaign — all apply here)

- No subagent reviews for this stack (Jack's waiver). One variable at a time;
  bars written before the run (done above).
- Never `/save_state` (see §4).
- No foreground blocking waits/sleeps; backgrounded tools only.
- Snapshot logs BEFORE any relaunch (start_trainer.sh clobbers
  trainer_srun.log); pull evidence Mac-side immediately.
- squeue shows zero devbox_trainer before any boot; stale-process sweep both
  nodes; stop_trainer.sh's GPU-state collection races teardown — the srun
  sweep is the real check.
- Plain language in every report; define each label on first use.

## 10. Report contents (to banach at window close)

Framing rules for every number (hilbert + Jack, final): the whole ladder runs
one tree with no TF32 anywhere — that is the environment, not a caveat.
**Absolute tok/s figures are context only, on our own tree, explicitly
non-comparable to any historical band (984 / 1089–1103); never produce or
imply an absolute record-comparable number.** Package framing rule (hilbert):
never phrase anything as "the recompute change gets us X and offload adds Y" —
selective recompute plus offload is ONE package; the offload is load-bearing
at 131k; there is no selective-only fallback; the block+K dial is THE
fallback and a different mechanism.

0. **THE ALL-RUNGS PROJECTION TABLE (§8c) — the first derived artifact,
   filled from this census in one pass** (2a@32k, 3a/3b/3c@131k, d16 anchor),
   per rank, each projection checked against the effective ceiling. Every
   subsequent gate inherits its band from THIS table.
0b. Any 32k rung's report (2a/2c) carries the caveat verbatim: **"32k
   de-risks CRASHES and WIRING, not TUNING — a green 32k means 'the machinery
   works', never 'the machinery is tuned'."** And 2c reports NO throughput
   number (tripwire reading, not a measurement).
1. The glue number (GiB/layer/mb) PER RANK — MoE buckets over MoE sets (70 /
   40), dense residual isolated on rank 0 — plus the full per-bucket census
   table vs §0, each rank separate, never averaged.
2. Per-rank gate-#1 fit read (banach decision 2): measured post-plateau peak,
   headroom vs the ~247.7 GiB effective ceiling, PROJECTED post-plan peak —
   and plainly: which rank is tighter, by how much, and GATE-#1 FAIL loudly
   if either projection is ≳227.7 GiB.
3. Which stage peaks higher today, by how much, on all three metrics — if
   the last stage (rank 8, loss/logits) binds, called out LOUDLY as a
   plan-level finding.
4. The at-boot config dump (decision 3): resolved indexer_types full/shared
   counts per stage, mlp_layer_types dense/MoE map per stage, expectation
   check result.
5. The d2 baseline: control tok/s/GPU mean/spread, torch HWM, poller plateau
   (both nodes), step time — PER RUN (A and B).
6. The noise floor: parity-leg three statistics (within-boot pair + the
   cross-boot cell vs boot 2), per-window |loss A−B| / |gn A−B| matched-step
   table, and the plausibility note vs the campaign's 3.7–5.5 (sanity only).
7. The DSA topk-stash row: GiB at peak, count vs expected, alloc sites, and
   the mandatory "paid today, not a plan cost" note.
8. Canary table (loss/gn per window).
9. Evidence manifest: pulled files + sha256s + box paths.
10. Any deviation from this runbook and why.
