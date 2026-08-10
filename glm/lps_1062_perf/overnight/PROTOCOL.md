# LPS-1062 overnight parallelism/MFU sweep — operator protocol v2 (2026-08-09)

You are running throughput experiments on GLM-5.2 LoRA SFT on B300 devboxes.
Orchestrator: **feynman** (message via `~/.agents/scripts/send-message.sh feynman "..."`).
Operators: **gibbs = box A**, **ramanujan = box B**, **laplace = box C**.
Read this fully before doing anything. Ask feynman if anything is unclear.

## Mission

Customer will train GLM-5.2 (CP THD) on ~608K samples: 70% ≤32K tokens,
28.5% 32–64K, 1.4% 64K–131,072 (max 131K). The golden 2×8 B300 config
(EP16/CP16) trains at 629 tok/s/GPU steady ~660 (mfu3x 6.6%, steady 6.9% — LoRA-corrected convention 2026-08-09) at 524K-token
steps with the "ship" env. **Tonight's question: which parallelism config
maximizes MFU in the customer's regime (packed THD buffers, ≤131K max
sample)?** Simplest, highest-return levers first; parallelism is the lever.
NO optimization that could impact correctness — every config change is
canary-checked, and only the two proven loss-neutral env knobs are allowed
(ship NCCL env + TF32 head).

## Why these configs (the hypothesis being tested)

CP16 exists to reach 512K context; the customer max is 131K. Per-rank token
count is identical between 262K@CP16 and 131K@CP8, so compute per rank
matches — but CP8 halves CP communication and frees a DP dimension, and with
attention-only LoRA the DP grad all-reduce is nearly free. EP32 (4 nodes)
additionally halves the expert weight floor (~131 GiB → less), shrinking
per-rank a2a rows. The benches below measure whether that arithmetic converts
to MFU.

## Box assignments & experiment queues

**Every bench is standardized at ~524,288 tokens/step** (seq_len × datums)
so tok/s/GPU and MFU are directly comparable across configs and to the
Aug-7 numbers. All runs: `--repeats` as listed, first bench per (config,L)
is that L's canary reference.

### Box A — gibbs — `devbox-up 2` — config `expA-ep16cp16.json` (golden, DP1)

| order | label | command args (`--num-gpus 16`) |
|---|---|---|
| 1 | A-anchor-262k | `--seq-len 262144 --datums 2 --repeats 3` |
| 2 | A-131k-d4 | `--seq-len 131072 --datums 4 --repeats 2` |
| 3 | A-65k-d8 | `--seq-len 65536 --datums 8 --repeats 2` |
| 4 | A-32k-d16 | `--seq-len 32768 --datums 16 --repeats 2` |
| 5 (stretch) | A-131k-d2 | `--seq-len 131072 --datums 2 --repeats 2` (step-size amortization probe) |

A-anchor-262k must reproduce Aug-7 exp06: ~629–660 tok/s/GPU, window losses
≈12.356 (warmup) / 12.339 / 12.310. If it doesn't (loss drift >5e-3 or
throughput off by >15%), STOP and message feynman — the stack diverged.
All 5 benches run against ONE trainer boot (same config).

### Box B — ramanujan — `devbox-up 2` — config `expB-ep16cp8dp2.json` (EP16/CP8/**DP2**)

**max_seq_len is 131072 — do NOT raise it.** CP8 at 262K ≈ 300 GiB/GPU
predicted (>275 cap); the boot warmup itself would OOM.

**Task 0 — DP semantics verification (GATE, do this before benching):**
with world 16 = EP16 and CP8, dp=2. Verify how `/forward_backward` `data`
datums map to DP groups: read the THD-CP controller loop in
`trainers_main` (megatron_controller / dp_worker) or verify empirically.
Expected: N datums are distributed across the 2 DP replicas (so
`--datums 4` = 2 per replica per step, total tokens = 4×L). If datums are
instead REPLICATED to both groups, or a `--datums 1` call leaves half the
GPUs idle in a way the controller doesn't account, our TPS accounting is
wrong — STOP and message feynman with what you found. Note in your report
how loss is reduced across DP groups (mean?) so canaries are interpretable.

| order | label | command args (`--num-gpus 16`) |
|---|---|---|
| 1 | B-131k-d4 | `--seq-len 131072 --datums 4 --repeats 3` |
| 2 | B-65k-d8 | `--seq-len 65536 --datums 8 --repeats 2` |
| 3 | B-32k-d16 | `--seq-len 32768 --datums 16 --repeats 2` |
| 4 (stretch) | B-131k-d2 | `--seq-len 131072 --datums 2 --repeats 2` |

B-131k-d4 vs A-131k-d4 is the headline A/B of the night.
Datum counts must stay multiples of 2 (DP2).

### Box C — laplace — `devbox-up 4` — three configs in sequence

All benches `--num-gpus 32`. Between configs: `stop_trainer.sh`, verify GPUs
clean, relaunch (~15 min each) — budget accordingly, expD and expE are the
priority, expF only if the night allows.

Boot 1 — `expD-ep32cp32.json` (EP32/CP32, DP1):

| order | label | args |
|---|---|---|
| 1 | C-D-262k-d2 | `--seq-len 262144 --datums 2 --repeats 3` (box C canary anchor) |
| 2 | C-D-131k-d4 | `--seq-len 131072 --datums 4 --repeats 2` |

Boot 2 — `expE-ep32cp16dp2.json` (EP32/CP16/DP2) — run the box-B DP
verification once here too (or reuse ramanujan's finding via feynman):

| order | label | args |
|---|---|---|
| 3 | C-E-131k-d4 | `--seq-len 131072 --datums 4 --repeats 3` |
| 4 | C-E-262k-d2 | `--seq-len 262144 --datums 2 --repeats 2` |
| 5 | C-E-65k-d8 | `--seq-len 65536 --datums 8 --repeats 2` |

Boot 3 (stretch) — `expF-ep32cp8dp4.json` (EP32/CP8/DP4, max_seq_len 131072):

| order | label | args |
|---|---|---|
| 6 | C-F-131k-d4 | `--seq-len 131072 --datums 4 --repeats 2` (datums must be multiples of 4) |

## Provisioning (each operator provisions their OWN box)

Previous boxes tonight (w79ypo3, qe5d22q, q9v7p53) are STOPPED — do not try
to reuse them. `devbox-up` ALWAYS creates a fresh box.

```bash
devbox-up <2|4> 2>&1 | tee ~/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/overnight/provisioning/box<A|B|C>_v2.log
```

~12 min provision + ~13–20 min trainer boot. After ssh works (`ssh tj-<jobid>`):

1. **Shared FS check** — `/root/.cache/user_artifacts` is ONE CPFS shared by
   all three boxes (project jrao123-ali). It should already contain
   `trainers_main` (expected `git log --oneline -1` → `0e0b65a6`, dirty
   working tree = the Aug-7 patches) and the server venv at
   `trainers_main/server/.venv`. Report the commit you see to feynman.
   NEVER run git mutations there, never edit `.devbox_up/`.
2. **TF32 patch check**:
   `grep -c BT_TF32_LM_HEAD /root/.cache/user_artifacts/trainers_main/server/src/trainers_server/dp_worker/api/chunked_lm_head.py`
   must print ≥1. If 0: message feynman BEFORE doing anything (the fix is
   `git apply` of `lps_1062_perf/patches/tf32-lm-head.patch` from the Mac,
   but coordinate first — the checkout is shared).
3. **Kit check/stage** — kit dir `/root/.cache/user_artifacts/lps1062/`
   needs: `bench_driver2.py`, `mfu.py`, `run_bench2.sh`, `poll_gpu_mem.sh`,
   `fold_mem.py`, `configs/*.json`. If missing/stale, scp from the Mac:
   ```bash
   scp ~/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/overnight/{bench_driver2.py,mfu.py,run_bench2.sh} \
       ~/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/{poll_gpu_mem.sh,fold_mem.py} \
       tj-<jobid>:/root/.cache/user_artifacts/lps1062/
   scp ~/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/overnight/configs/*.json \
       tj-<jobid>:/root/.cache/user_artifacts/lps1062/configs/
   ```
   **CPFS gotcha (real, hit before): relay-scp'd files can read as all-NULs
   from a sibling box.** After staging, verify checksums ON YOUR BOX
   (`md5sum` there vs `md5 -q` locally) before trusting any file you didn't
   just write from that box.
4. Gotcha (papercut pc_f38f091cf0e4): on multinode boxes the trainer HTTP
   server lands on Slurm's node 0 (alphabetical hostname), not necessarily
   the k8s leader. After health, `curl -s localhost:8001/health` on
   `tj-<job>` AND `tj-<job>-1` (etc.); run the bench driver on the node
   where it answers.
5. Before every (re)launch, verify nodes are clean:
   `srun --overlap -N $BT_GROUP_SIZE -n $BT_GROUP_SIZE --ntasks-per-node=1 bash -lc 'hostname; pgrep -fc "[d]p_worker.main|[t]orchrun"; nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1'`
6. B300s report as `L20D` in nvidia-smi (rebadge; they are Blackwell Ultra).

## Trainer launch

```bash
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_SPLIT_DATA_ON_QPS=1
export NCCL_NCHANNELS_PER_NET_PEER=8
export BT_TF32_LM_HEAD=1
BT_TRAINER_CONFIG_PATH=/root/.cache/user_artifacts/lps1062/configs/<config>.json \
BT_TRAINER_SERVER_CONFIG_PATH=/root/.cache/user_artifacts/lps1062/configs/server-config.json \
bash /root/.cache/user_artifacts/.devbox_up/start_trainer.sh --num-nodes <N>
```

- The ship env (+51% at 256K, proven loss-neutral 2026-08-07) and TF32 head
  go on EVERY launch. Nothing else in the env changes without feynman's OK.
- **Always pass `--num-nodes` explicitly** — the shared `.devbox_up/` scripts
  were last generated by whichever box provisioned last; the default node
  count may be wrong for you.
- `BT_SKIP_FULL_WARMUP` stays UNSET tonight: the boot warmup at max_seq_len
  is prod-like, and booting at all is part of the result (it proves the
  config's max-length footprint fits).
- Wait via `wait_trainer_health.sh` in the background; boot ~13–20 min.
- Leave `weight_sync` disabled (as configured); checkpoint_dir is /tmp.

## Bench protocol (apples-to-apples)

```bash
bash /root/.cache/user_artifacts/lps1062/run_bench2.sh <label> \
    --seq-len <L> --num-gpus <16|32> --datums <D> --repeats <R> \
    [--canary-json /root/.cache/user_artifacts/lps1062_bench/<anchor>.json]
```

- Run from the node where :8001 is healthy. Results land at
  `/root/.cache/user_artifacts/lps1062_bench/<label>.json` with per-GPU
  max/min mem folded in.
- First main window after boot runs ~15% slow (allocator/autotune settling);
  that's why anchors use `--repeats 3` — steady state = windows 2–3.
- **Canary**: first bench per (config, L) is the reference for that L.
  Later benches at the same L pass `--canary-json`. Loss/grad_norm drift
  ≤2e-3 = fine (run-to-run noise). Drift >5e-3: STOP, message feynman.
  Cross-config comparisons at the same L (e.g. B-131k vs A-131k) should also
  agree on loss to ~5e-3 — DP/CP resharding only changes reduction order.
  Bigger drift = possible correctness problem with that parallelism config:
  that's a FINDING, report it immediately (do not keep benching that config).
- OOM during a bench = a valid result. Record label + error + mem peaks,
  message feynman, move on.
- Synthetic data = single sequences of length L (rng seed 0xB300, fixed
  consumption order), matching all prior LPS-1062 benches.

## MFU reporting (method UPDATED 2026-08-09 by fibonacci, at Jack's direction)

`mfu.py` (in the kit) is now LoRA-corrected: frozen base weights skip the
wgrad GEMM in backward (verified at the bench-commit pins), so a step costs
2×matmul + 3×attention + 3×adapter forward-equivalents (useful), or
3×/4×/4× under full recompute (executed). `bench_driver2.py` reports:

- `mfu3x` = tps/GPU × useful(L, r) / 2.5e15 — useful FLOPs, notebook convention
- `hfu` = tps/GPU × executed(L, r) / 2.5e15 — full recompute (analytic estimate;
  see mfu.py docstring for the empirical measurement route)

Supersedes the prior flat 3×/4×-of-fwd full-FT convention, which read
×1.30–1.39 (mfu3x) / ×1.21–1.27 (hfu) high over 32K–262K; per-label
conversions of every published figure: `overnight/mfu_lora_correction.md`.
Exact for single-sequence synthetic benches; raw tok/s/GPU remains
method-independent.

## Reporting (CHANGED from v1 — do not edit the notebook)

- **Do NOT edit NOTEBOOK.md** — feynman maintains it (single writer, no
  merge collisions).
- After EACH bench: scp the result json to the Mac:
  `scp tj-<job>:/root/.cache/user_artifacts/lps1062_bench/<label>.json ~/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/overnight/results/<label>.json`
  then message feynman ONE line:
  `<label> | <tok/s/GPU> | step <s> | mfu3x <%> | hfu <%> | mem <max>/<min> MiB | canary <ok|dloss=…>`
- Also message feynman: when your box is provisioned+healthy (include jobid
  + trainers_main commit + TF32 grep result), when a boot fails, when your
  queue is done, when you're blocked >20 min on anything.

## Rules

- Correctness first: no experimental knobs beyond the listed configs/env.
  Anything that moves the loss canary >5e-3 gets reported and stopped.
- Do NOT kill or reconfigure another operator's box. `squeue`/`sinfo` are
  shared per-cluster — ignore jobs that aren't yours.
- Do NOT stop your devbox when your queue is done — message feynman first
  (feynman coordinates teardown; idle boxes burn quota, so don't sit on a
  finished queue silently either).
- If a trainer boot fails twice with the same config, message feynman with
  the last ~80 log lines before retrying differently.
- You may spawn disposable subagents for analysis/code-reading, but you run
  the benches yourself (they need your box).
