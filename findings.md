# Findings & pitfalls — Nemotron-3-Super 131k LoRA SFT

Running log of non-obvious findings while finding a 131k golden config for
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` (hybrid Mamba2 + Transformer
MoE) on 2-3 nodes of 8x B200. Newest at the bottom of each section.

## Architecture / parallelism

- **TP is hard-capped at 8 for this model.** The Mamba2 mixer has `ngroups=8`
  and MCore asserts `ngroups % tp_size == 0`, so TP=16 fails at construction
  (`AssertionError: ngroups must be evenly divisible by tp_size`). To shard
  across >8 GPUs you must use CP or PP, not more TP.

- **Context Parallelism (CP) does NOT shard activations in the committed
  codebase.** Our data path (`packing.py`) emits only `bshd` and never calls
  Megatron's `get_batch_on_this_cp_rank` (`megatron/core/utils.py`), and the CE
  forward step feeds the full `(B,S)` to the model. So `context_parallel_size>1`
  builds CP process groups + CP attention but every rank still gets the whole
  sequence: zero per-GPU activation savings, doubled total memory, and
  unverified numerics (cf. hybrid-CP skill pitfall #5: silently wrong
  gradients). CP is a net-negative lever here until the data path is taught to
  slice the sequence. Even if wired, Mamba's recurrent scan doesn't
  context-parallelize like attention. **Use PP, not CP, to fit long context on
  this model.**

  Update: an experimental `_slice_batch_for_cp` was tested (not committed) —
  it DOES shard activations and fits 131k on a single node at TP4/CP2, but has
  a ~1% loss/grad correctness gap (CP-unaware normalization). See
  `super/profiling_memory.md` Experiment 4 for the full investigation and the
  remaining-work list to make CP>1 a correct training config.

- **Why a 1.1T Kimi fits 131k on 16 GPUs but a 120B Nemotron doesn't:** total
  params drive *weight* memory (frozen in LoRA, sharded by TP×EP, fixed wrt
  seq len); seq len drives *activation* memory, sharded mainly by TP. Kimi is a
  pure-transformer MoE → TP=16 legal → 16-way activation sharding + compact MLA
  KV. Nemotron is hybrid Mamba → TP≤8 + heavy SSM per-token state → ~2x the
  per-GPU activation and no working extra-sharding lever (CP unwired).

## Code bugs found (fixed in-repo)

- **PP>1 crashes the Mamba provider:** `mamba_provider.provide()` reads
  `self._pg_collection.pp`, but our controller calls `get_model(...)` directly
  and never sets `provider._pg_collection` (only the deprecated
  `provide_distributed_model` does, at model_provider.py:187). At PP=1 the
  controller forces `pre/post_process=True`, short-circuiting the deref, so it's
  hidden; at PP>1 it raises `AttributeError: 'NoneType' object has no attribute
  'pp'`. Fix: set `model_provider._pg_collection = pg_collection` before
  `get_model` in `megatron_controller._init_training_stack`.

- **Attention backend for CP:** the hybrid provider hardcodes
  `AttnBackend.flash`, but the only installed flash build on B200 is
  FlashAttention-4, and TE 2.16 disables FA4 whenever CP>1
  (`dot_product_attention/utils.py`). With CP>1 the flash-only policy leaves no
  usable backend → `No dot product attention backend is available`. Fix:
  conditionally set `AttnBackend.fused` (cuDNN) when CP>1 in the controller;
  keep FA4 at CP=1. (Note: with the PP-not-CP strategy, CP=1 and this stays on
  FA4.)

## Devbox / ops pitfalls

- **`run_trainer_node.sh` NUM_NODES clobber:** `super_env.sh` exports
  `NUM_NODES=1`; the launcher must capture the desired `NUM_NODES` *before*
  sourcing `super_env.sh` and re-export after. The committed version at the
  branch point sourced first then `export NUM_NODES="${NUM_NODES:-2}"` (no-op
  because it was already 1) → every node launched `--standalone` (world=8) and
  PP×TP failed the divisibility assert. Fixed version captures `_WANT_NUM_NODES`
  first.

- **Detaching multi-node launches:** `ssh host 'setsid ... &'` over a
  ControlMaster-multiplexed connection can hang the ssh client (it waits on the
  backgrounded process's inherited fds), so a launch loop silently only starts
  rank 0. Use `tmux new-session -d -s trainer "..."` per node instead — it fully
  detaches and ssh returns immediately.

- **Node not provisioned for the trainer:** a node may be running something
  else (e.g. a vLLM sampler for a different model) and lack the trainer venv /
  be on a different branch. Reclaim it: kill the other workload, `git fetch`
  the task branch via token, `git submodule update --init --recursive`, then
  `uv sync --extra worker --extra dev` (mamba-ssm/causal-conv1d build from
  source, ~3-5 min). Copy `/root/super_env.sh` from an already-set-up node.

- **git token URL rewrites collide:** bootstrap sets
  `url."git@github.com:".insteadOf "https://github.com/"`. To use a `gh` token
  over HTTPS for private submodules, `--unset-all` that rule and add BOTH
  `url."https://x-access-token:$TOKEN@github.com/".insteadOf` entries (for
  `https://github.com/` and `git@github.com:`) with `--add` (same multi-valued
  key — a second plain `git config` replaces, not appends).

- **SSH connection reuse:** add `ControlMaster auto` / `ControlPath` /
  `ControlPersist 30m` to the `Match host training-job-*.ssh.baseten.co` block
  so repeated `ssh` calls reuse one tunnel instead of re-handshaking each time.
  (Documented in the multi-devbox-management skill.)

- **Multi-node hang vs OOM:** when one node OOMs and dies mid-collective, the
  survivors don't error — they hang in NCCL looking 100% busy / full memory.
  Always check *all* nodes' GPU + process state; a node at 0 MiB while others
  are pinned means it died and the rest are waiting on it.

- **`pybind11/pybind11.h: No such file or directory`** at trainer startup:
  Megatron compiles its dataset `helpers_cpp` C++ extension at import time. It
  needs (a) `pybind11` installed in the venv, and (b) the helpers Makefile calls
  bare `python3 -m pybind11 --includes`, which resolves to the *system* python
  unless the venv `bin` is on PATH. Fix: install `pybind11` in the venv AND
  prepend `<server>/.venv/bin` to PATH at launch (and/or set
  `CPLUS_INCLUDE_PATH` to the venv's pybind11 include dir). Persist by adding
  `pybind11` to `server/pyproject.toml` worker extra.

- **Fresh box without bootstrap:** if you skip `dev_job/bootstrap.sh`, you'll be
  missing apt packages it installs — `python3-dev` (`Python.h` → C-ext builds
  fail) and `tmux` (silently makes `tmux new-session` a no-op). Install both.

- **`mamba-ssm` build `urlopen error [Errno -5] No address associated with
  hostname`:** its setup.py fetches a prebuilt wheel from GitHub; intermittent
  DNS makes it hang/fail. Force the source build (nvcc present):
  `MAMBA_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE uv sync ...`.

- **`pkill -f <pat>` / `pgrep -f <pat>` self-kill:** over SSH, your own remote
  command line contains `<pat>`, so it matches and kills your session (exit
  255). Kill by explicit PID, or run the kill from a script file (its cmdline is
  just `bash /path/script.sh`, no match), or use a bracket pattern AND ensure
  your command doesn't contain the literal substring.

- **Cross-cluster box bring-up (no shared cache):** `/root/.cache/user_artifacts`
  IS shared across the nodes of a single multinode job (download weights ONCE
  there); `/b10/workspace` and `/root` are per-node (clone repo + build venv per
  node). `BT_LEADER_ADDR` is available even in non-interactive ssh. Stage the
  `uv` binary via the shared cache to avoid re-installing per node. Disable Xet
  (`HF_HUB_DISABLE_XET=1`) if the HF download hangs with
  `httpx.ReadError: Bad file descriptor`.

## Pre-provision GPU health gate (catches faulty GPUs `nvidia-smi` misses)

Two consecutive fresh boxes had a dead node: one with the SSH proxy down, one
where **node 1's GPU 6 was faulty** — `nvidia-smi` showed it idle with 0 ECC / 0
Xid, but `torch.cuda.set_device(6)` (and `cudaSetDevice(6)`) failed with `CUDA
error: CUDA-capable device(s) is/are busy or unavailable`. A whole-device
`torch.cuda.init()` / `device_count()` did NOT catch it — only binding that
specific device did. TP=8 needs all 8 GPUs on every node, so one bad GPU kills
the run.

**Always health-gate a new box BEFORE provisioning:** (1) all N nodes reachable
+ 8 GPUs each; (2) 0 uncorrected ECC / 0 Xid; (3) a real per-GPU CUDA
bind/malloc/sync probe on every GPU of every node. Cheap nvcc probe (no torch
needed):

```c
// gpucheck.cu — nvcc -o gpucheck gpucheck.cu
#include <cuda_runtime.h>
#include <stdio.h>
int main(){int n=0;cudaGetDeviceCount(&n);int bad=0;
 for(int i=0;i<n;i++){cudaError_t e=cudaSetDevice(i);
  if(e){printf("gpu%d SETDEV %s\n",i,cudaGetErrorString(e));bad=1;continue;}
  void*p=0;e=cudaMalloc(&p,1<<24);
  if(e){printf("gpu%d MALLOC %s\n",i,cudaGetErrorString(e));bad=1;continue;}
  cudaFree(p);cudaDeviceSynchronize();}
 printf("%s (%d gpus)\n",bad?"BAD":"ALL_OK",n);return bad;}
```

If any GPU reports BAD, stop the job (`truss train stop --job-id <id>`) and spawn
a new box rather than provisioning a doomed cluster.

## VALIDATED golden config (131k)

**TP=8, PP=4, CP=1, EP=8, ETP=1** on 4 nodes / 32× B200, bf16 LoRA rank 16,
full recompute, micro-batch 1: **131072-token step runs, no OOM, peak 100.79
GiB/GPU**, loss decreasing (0.667→0.654), ~11 s/step steady-state (after a
~491 s first-step autotune). 88 layers → PP must divide 88; PP=4 is the
smallest clean split that fits. Added as the `NEMOTRON_3_SUPER S131K` golden
row. (Memory sweep details: `super/profiling_memory.md` Exp 1–8; the
recompute setting was later improved from "full"-inert to real full layer
recompute, which fits 131k on a single node — see Exp 5/6.)

## Pitfall: recompute parity tests must control optimizer state (a self-inflicted false alarm)

While validating that full/Mamba activation recompute is numerically equivalent
to eager (no-recompute), a trajectory-parity test first appeared to **FAIL** —
full recompute looked like it diverged from eager by +0.11 loss over 8 steps.
It was an **experimental artifact**, not a real bug. Recompute IS correct
(re-running it cleanly: full vs eager max deviation 0.0021, *inside* the
no-recompute-vs-no-recompute control band of 0.0037). The mistake is instructive:

1. **`lr=0` freezes weights but NOT the optimizer.** Adam updates its moments
   every step regardless of lr:
   `m ← β1·m+(1−β1)·g`, `v ← β2·v+(1−β2)·g²`, `θ ← θ − lr·m̂/(√v̂+ε)`.
   Only the last line is scaled by lr. So `lr=0` steps leave `θ` at init but
   warm `m`, `v`, and the step counter `t`. "Frozen weights" ≠ "frozen
   optimizer state."

2. **The confound.** The `full` trainer used for the trajectory had already
   taken ~21 `optim_step`s (a 10-step lr=0 loss-parity run + grad_norm probes) —
   visible as `"step":21` in `/status` before the run. The `no-recompute`
   trajectories were each run on a **fresh** restart (optimizer at `m=v=t=0`).
   So full and no-rc started from different optimizer states; that asymmetry —
   not recompute — drove the divergence.

3. **Why bias-correction doesn't save you past step 1.** With weights frozen and
   data fixed, the lr=0 gradient `g` is constant, so `m→g`, `v→g²` and bias
   correction makes the *first* real step identical to a fresh optimizer. But
   from step 2 on the weights move, `g` changes, and the warmed optimizer
   (`t≈22`, no warmup ramp) blends new gradients differently than a fresh one
   (`t=1,2,…`). The trajectories drift — exactly the +0.11 seen.

4. **A frozen-weight (lr=0) loss-average test cannot detect recompute bugs at
   all.** The reported loss comes from the (unchanged) forward pass, so it
   matches to ~1e-5 whether or not the backward/gradients are right. Recompute
   correctness lives in the gradients; only a **real-LR trajectory** (or a direct
   grad comparison) exposes it.

**Rules for recompute / numerics parity going forward:**
- Every arm of a trajectory-parity run must start from a **fresh optimizer**
  (fresh trainer process or explicit optimizer reset). Never reuse a trainer
  that has called `optim_step`. Check `/status` `step` is 0 before starting.
- Always run a **control** (same config twice) to establish the noise floor;
  only call a difference "real" if it exceeds that band.
- Test numerics with a real-LR trajectory, not a frozen-weight loss average.

(See also `super/profiling_memory.md` Experiment 7 for the parity table and
the retraction of the earlier "FAILS" draft.)
