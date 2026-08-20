# Activation placement — state and instructions

Single entry point for this workstream. Read this, then
`RUNG3_PREREG.md` (the bars for the run in flight) and `RUNG1_REPORT.md`
(the baseline measurements everything is judged against).
`ACTIVATION_PLACEMENT_PLAN.md` is the design of record — read its AMENDMENTS
block, not just the body.

## What we are doing

GLM-5.2 at 131k context on 2 nodes x 8 B300, PP2/CP8/EP8. Today the trainer
throws away every layer's intermediate data and recomputes it during
backward, which costs about 35% of each step. We are replacing that with:
keep the small bookkeeping tensors on the GPU, stream the big expert-block
and attention-projection tensors to CPU memory over PCIe, and recompute only
core attention. Target is roughly +40% throughput.

**Selective recompute and the offload are ONE package.** Selective recompute
alone does not fit in memory at 131k. There is no "take the recompute win
without the offload" fallback; the only fallback is the block+K dial, a
different mechanism. Never describe the result as "recompute gets X and
offload adds Y".

## Right now

**Arm 3b is booting on qed7z1w** (dispatched 22:55Z, health expected ~15 min
later). 3b is the phase-1 configuration of record: selective recompute +
offload of `moe_act` and `attn_proj`.

When it is healthy, run in this order:

```bash
bash $PP2/rung3_arm.sh 3b
```

That does the three boot checks first (they are pass/fail — see below), then
a forward-only parity leg, then a timed d2 run matched to the baseline, then
the plateau read and valve counters.

**Then, on the SAME boot** (no reboot needed — d16 has the same peak memory
as d2 because 1F1B caps in-flight microbatches at 2), run the d16 window that
is half the headline:

```bash
$VENV/bin/python $PP2/profile_driver_new.py --label rung5-3b-qed7z1w-d16-$(date -u +%Y%m%dT%H%M%SZ) \
  --datums 16 --control-repeats 6
```

## Then: the headline deliverable

One more boot — the **full-recompute baseline at d16** — and the deliverable
is the **ratio** between that and 3b's d16 number. d16 rather than d2 because
d2's pipeline bubble is ~33% of the step versus ~6% at d16, which understates
the win.

Absolute tokens/sec figures are context only, on our own tree. **Never
compare anything to the historical 984 or 1089-1103 bands** — different tree
and environment. There is no TF32 anywhere in this ladder; that is the
environment, not a caveat.

**Arm 3a is cancelled.** It cannot ship (3b is the phase-1 configuration), it
holds `attn_proj` resident for ~14 GiB more than 3b on rank 0, and it is
projected at or past the memory fail line. It survives only as a conditional
diagnostic: run it if and only if 3b's throughput disappoints AND the
copy-exposure trace implicates `attn_proj` traffic.

## How to work here

- **Test in isolation before spending a boot.** A trainer boot is ~15-20
  minutes of weight load. Both bugs that killed the first offload arms needed
  no model and no 131k context to reproduce — one object construction and one
  function call would have caught them. **Run `offload_preflight.py` in the
  worker venv on an idle GPU before every offload boot**; it exercises the TE
  latch, PCI/NUMA resolution, manager construction, pinned-pool allocation
  and a real device-to-host round trip in seconds. Exit non-zero means do not
  boot. Extend it whenever a new mechanism appears.
- **Commit on the laptop, push, `git pull` on the box. Never scp code.** Jack
  watches the PR evolve. (Configs and box-side tools under `tools/` are
  artefacts, not code — scp for those is fine.)
- **Answer Jack's questions directly.** Do not launch into tool calls when he
  asks a question.
- **No fleet of subagents.** Work directly; a one-off subagent only for a
  genuinely self-contained, context-heavy lookup.
- **Plain language in every report.** Define each label on first use. No
  dense shorthand.
- **Use the `devbox-up` skill for devbox management.**
- Make your own judgment calls; do not stop to ask about routine choices.

## The box

**qed7z1w** — 2 nodes x 8 B300 (`ali`), provisioned by Jack. Do not stop it,
do not provision another without asking. Capacity in this region is scarce.

```
ssh training-job-qed7z1w-0.ssh.baseten.co   # ranks 0-7,  pp_rank 0, 3 dense + 35 MoE layers
ssh training-job-qed7z1w-1.ssh.baseten.co   # ranks 8-15, pp_rank 1, 40 MoE layers + loss head
```

`nvidia-smi` reports these as `NVIDIA L20D` / Ada / cc 8.9. That is
export-compliance masking. CUDA reports the truth: **sm_103, 267.7 GiB, 148
SMs** — B300.

Paths: `ART=/root/.cache/user_artifacts`, `PP2=$ART/lps1062_pp2`,
`CLONE=$ART/trainers_main`,
`VENV=$CLONE/server-megatron-bridge/.venv/bin/python`.

### Box mechanics that are not obvious

- **The trainer runs through pinned scripts in `$PP2`, not the devbox-up
  ones.** `stage_and_boot.sh` (pre-boot sequence + dispatch),
  `run_trainer_node_pp2.sh`, `wait_trainer_health_pp2.sh`. The devbox-up
  copies target the pre-#1027 layout, cap at 1 node, and grep for a renamed
  process; they will not work.
- **Health is checked only with `wait_trainer_health_pp2.sh`**, backgrounded,
  re-invoked at each 3-minute checkpoint. Never a blocking sleep loop.
- **Any `srun` that runs alongside the trainer must attach to its allocation**:
  `srun --jobid=$(squeue -h --name=devbox_trainer -o %A) --overlap ...`.
  A bare `srun --overlap` queues behind the trainer and looks like a hang.
- **The drivers need the worker venv's python** (`$VENV`); nothing else on the
  box has `httpx`.
- **Never run heavy analysis on the box during a timed window.** It cost one
  census run 2.4% throughput and quadrupled its spread.
- **Never call `/save_state`** — the async save path hangs under CP>1.
- Rebuilding the venv: `make megatron-bridge-venv CUDA_FLAVOR=cu13`, and it
  needs the CUDA 13 toolkit apt-installed on BOTH nodes
  (`cuda-nvcc-13-0 cuda-libraries-dev-13-0`, `CUDA_HOME=/usr/local/cuda-13.0`).
  Do NOT install `nvidia-cudnn-frontend` from PyPI — it clobbers the tree's
  vendored pin. The import name is `cudnn`, not `cudnn_frontend`.

## The code

**trainers PR #1074** (draft), branch `jackrao/lps-1062-actplace`, based on
#1070's branch — retarget to `main` when #1070 merges. Vendored changes are
real commits, not patches:

- `basetenlabs/Megatron-LM` @ `jackrao/lps-1062-activation-offload`
- `basetenlabs/Megatron-Bridge` @ `jackrao/lps-1062-activation-offload`

Current tip: **`bb027cc6`**. Boot the recorded commit, never the moving tip,
and record what actually booted in the report.

A fresh checkout has EMPTY submodules — `git submodule update --init
--recursive`. megatron-core is installed editable, so a source change in the
fork takes effect on `git checkout` without a venv rebuild.

## The offload boot environment

```bash
NVTE_CPU_OFFLOAD_V1=1 BT_OFFLOAD_VALVE_TELEMETRY=1 BT_OFFLOAD_VALVE_TELEMETRY_EVERY=1 \
  bash $PP2/stage_and_boot.sh <arm config>
```

- `NVTE_CPU_OFFLOAD_V1=1` **must be in the launcher environment.**
  Transformer Engine latches it at import while the validator reads it
  lazily, so a worker-side export passes validation while TE silently keeps
  its pre-V1 path.
- Both telemetry variables are required: the first defaults off, and the
  second defaults to dumping every 100 training iterations against 12-step
  runs. Without them the valve produces no data at all.
- Leave `BT_OFFLOAD_NUMA_BIND` UNSET (defaults on). `=off` is the A/B control
  arm only.

## The three boot checks — pass/fail, read before any timing

If the offload did not actually engage, every number afterwards is the
baseline wearing the arm's label. That is the worst result this ladder can
produce, because it reads as a real negative.

1. `activation-offload NVTE latch: env=1 latch=True -> OK`. **`latch=False` is
   a failed boot**, not a slow one — fix the launcher env, do not debug TE.
2. `activation-offload engagement:` — `fine_grained_activation_offloading=True`,
   the arm's module list, and the first-MoE-layer flags reading True rather
   than None.
3. `BT_OFFLOAD_NUMA_BIND: GPU ... -> NUMA node N` and
   `BT_OFFLOAD_NUMA_VERIFY: pinned buffer ... page nodes ...` reporting
   NUMA-local. Unbound placement is the process default and runs ~30% under
   the bandwidth this design needs while looking perfectly healthy.

## Numbers you need to judge a result

- **Baseline (full recompute, d2, 131k):** 645.1 tok/s/GPU, 1.4% within-run
  spread, step 25.4 s.
- **Plateau memory (nvidia-smi reserved):** first stage 152.5 GiB, last stage
  162.6 GiB. Card 267.7 GiB; effective ceiling ~247.7 after the ~20 GiB
  cold-allocator burst; gate fail line ~227.7.
- **Read memory only at the plateau.** Reserved memory creeps ~+21 GiB over
  the early steps and flattens by about step 9. The read is the poller max
  over the last two control windows, which must agree within ~2 GiB.
  `mem_plateau.py` does this.
- **Numerical noise floor** (two identical forward-only legs, same weights):
  loss stable to 2.9e-05, but per-token logprobs differ by 0.059 at the
  median and 4.34 at worst, with 95.67% of tokens over 1e-3. **Parity here
  only catches gross breakage; the aggregate loss is the sharp instrument.**
  Judge arms as ratios against `$PP2/floor_withinboot.json` using
  `parity_floor_stats.py`. Never import an absolute 1e-6/1e-3 bar, and never
  reuse the campaign's 3.7-5.5 figure — this box's own pair is the floor.
- **Layer geometry:** 78 layers, `mlp_layer_types` 3 dense + 75 sparse with
  all 3 dense on stage 0; `indexer_types` 11 full on stage 0, 10 on stage 1.
  So rank 0 = 70 MoE sets + 6 dense, rank 8 = 40 MoE sets. The stages bind on
  different things — rank 0 on set count, rank 8 on total peak. **Never
  average the two ranks.**

## What we know that the plan does not

- **The glue row is still unmeasured, and rung 1 could not measure it.** Under
  full recompute those tensors are not resident, so a baseline boot can only
  see today's residency. **3b's peak is what measures it:**
  `glue = (peak_3b - base - prefetch) / sets`. Do that as soon as 3b's
  plateau is read, and put the number in `RUNG3_PREREG.md`.
- **The last stage's loss head is 26.04 GiB**, not the ~5.1 GiB the plan
  assumes — 5x, stable across all eight last-stage ranks, the largest single
  bucket in the census. It is why the last stage peaks higher than the first
  despite holding 40 activation sets to 70. Rank 0 still binds the
  projection, but rank 8's margin is ~21 GiB smaller than the plan thinks.
- **The DSA topk-stash census row is not trustworthy.** It is isolated by
  exact byte size and at this geometry other tensors share that size — the
  printed allocation sites are autograd/RoPE/GEMM, not the indexer. Whatever
  those blocks are, they are paid today and are already inside the base, so
  they are not a cost of this change.
- `memory_census.py` warns that the allocator event ring wrapped. At
  `--max-entries 1000000` that warning is a **false positive**: the window
  covers ~11 steps. It fires because the replay starts from zero and cannot
  see the persistent baseline allocated before profiling began.

## Hard constraints

- `offload_core_attention` stays OFF. Phase 2, and conditionally alive at
  best — it fits the sustained unidirectional bandwidth regime but not the
  phase seams.
- No `qkv_linear` offload: its tensor is shared with the core-attention
  checkpoint, so offloading it either double-stores or puts a host-to-device
  transfer inside the critical path of the one thing we chose to recompute.
- The valve (`max_inflight_offloads`) stays at its default (uncapped) for the
  arm. Tuning it in the same run as the arm change would break
  one-variable-at-a-time. It blocks the compute stream rather than skipping,
  so an undersized valve costs throughput and never memory.
- No per-PR subagent reviews for this stack.

## Tools

All in `lps_1062_perf/tools/`, staged to `$PP2` on the box:

| tool | what it does |
|---|---|
| `offload_preflight.py` | seconds-long pre-boot gate for the offload paths |
| `stage_and_boot.sh` | drain check, sweep, config staging + cross-node sha, env, dispatch |
| `wait_trainer_health_pp2.sh` | the only sanctioned health wait |
| `rung3_arm.sh` | one arm end to end: boot checks, parity, timed run, plateau, valve |
| `rung1_census.sh` | the baseline census pair |
| `profile_driver_new.py` | the timing/memory driver (`--max-entries` defaults to 1M) |
| `parity_driver.py` | forward-only parity legs |
| `parity_floor_stats.py` | parity as a distribution, judged as ratios against a floor |
| `mem_plateau.py` | the plateau read from the poller CSVs |
| `memory_census.py` | per-module attribution from allocator snapshots |
| `projection_table.py` | the memory projection, computed once |
| `analysis/trace_tools/copy_exposure.py` | exposed copy stall; validated against the reference trace |
