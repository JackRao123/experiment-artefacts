# Trainer startup optimization notebook

Date: 2026-08-25

Status: baseline instrumentation in progress.

## Objective

Reduce trainer launch-to-`/health` latency, using GLM-5.2-FP8 as the primary
measurement model. Iterate on reduced-layer GLM proxies and reserve the full
78-layer model for final end-to-end validation.

## Source and hardware

- Trainers baseline: `74c5f22744873f639bdf8f9ed5811a584a0d00f9`.
- Branch: `jack-optimise-trainer-startup`.
- Local worktree: `/Users/jackrao/Documents/trainers-wt-startup`.
- Devbox: `tj-q4grmdq`.
- Hardware: 2 nodes x 8 B300 GPUs. `nvidia-smi` reports B300 as `NVIDIA L20D`.
- Devbox state at adoption: GPUs idle; no `devbox_trainer` Slurm job.
- The pre-existing devbox trainer checkout is detached at `71a9f3b7` and has
  unrelated modifications. It will not be cleaned or reused. This campaign
  uses a separate remote worktree and run-specific copies of devbox-up's
  generated lifecycle scripts.

## Measurement contract

- Startup begins immediately before dispatching the generated
  `start_trainer.sh` lifecycle script.
- Startup ends at the first successful HTTP 200 response from `/health`.
- Use generated devbox-up `start_trainer.sh`, `wait_trainer_health.sh`, and
  `stop_trainer.sh`; do not manually launch or poll.
- Keep model, snapshot, topology, warmup length, code revision, and cache state
  fixed within each A/B comparison.
- Record warm/cold cache state explicitly.
- Do not disable startup warmup. It is a forward+backward connectivity and
  kernel-compilation gate and is part of time-to-ready.
- Phase timers use host wall time and add no CUDA synchronization or distributed
  barriers. They are diagnostic timings, not device-kernel timings.
- Log every rank so the collective straggler is visible.

## Model ladder

### Iteration proxy

- Snapshot: `/root/.cache/user_artifacts/glm52-debug-1d1m`.
- Shape: one dense layer plus one MoE layer.
- Preserved production dimensions: hidden size, attention geometry, vocabulary,
  256 routed experts, top-k 8, and shared expert count.
- Topology: TP1/PP1/CP1/EP8 on one 8-GPU B300 node. EP8 preserves the
  production expert sharding and all-to-all path without paying for the second
  pipeline node.
- Max sequence length and startup warmup: 8,192 tokens unless an A/B explicitly
  tests warmup behavior.
- Snapshot is random BF16 and therefore does not model production FP8
  checkpoint dequantization or full checkpoint I/O.

### Scaling proxies

Use the existing 0D1M/0D2M/0D4M/0D6M snapshots when a suspected cost needs a
layer-scaling curve. The 0D1M family uses production MoE layer 6, which owns a
complete DSA indexer. The 1D1M snapshot's MoE layer was derived from an
index-sharing layer and emits missing-indexer warnings, so indexer-sensitive
conclusions require the 0D1M family.

### Final validation

- Model: full `zai-org/GLM-5.2-FP8` checkpoint.
- Snapshot currently present on the devbox cluster:
  `/root/.cache/team_artifacts/huggingface/hub/models--zai-org--GLM-5.2-FP8/snapshots/ba978f7d347eaf65d22f1a86833408afdb953541`.
- Production B300 topology: 2 nodes x 8 GPUs, TP1/PP2/CP8/EP8.
- Full model is used only after the winning changes pass the debug ladder.

## Prior evidence

Historical 1D1M warm launches were approximately 70-75 seconds; a cold launch
was 158.1 seconds. One coarse instrumented restart measured:

| phase | seconds |
|---|---:|
| imports and provider setup | 48.6 |
| construct 12.19B-parameter model | 18.4 |
| load 24.3 GB BF16 checkpoint | 8.4 |
| optimizer/checkpoint setup | 0.8 |
| startup forward+backward | 27.5 |
| HTTP server startup | 0.2 |
| total | 103.9 |

This evidence suggests imports, model construction, and warmup are large on the
proxy, but cache variability was high and the categories were too coarse for a
safe optimization decision.

## Scaling interpretation

- Import/runtime initialization is largely fixed per rank.
- Model construction, checkpoint conversion/loading, LoRA module traversal,
  and much of DDP registration are expected to scale approximately with layer
  count or parameter count.
- Startup forward+backward scales with both layer count and warmup sequence
  length, with additional fixed compilation/collective setup costs.
- Any per-layer proxy phase is projected to the full model using 3 dense and 75
  MoE layers. A small proxy duration is not dismissed when its scaling curve is
  linear.
- Production FP8 checkpoint loading/dequantization must be measured on the full
  model because the BF16 proxy intentionally omits that mechanism.

## Session log

### 2026-08-25 10:00 PDT - setup

- Fetched the requested baseline and created/pushed
  `jack-optimise-trainer-startup` at exactly `74c5f2274`.
- Full repository pre-push checks passed after initializing submodules.
- Read the devbox-up lifecycle rules and verified the assigned B300 devbox is
  reachable and idle.
- Located existing debug snapshots and prior timing evidence.
- Started a diagnostic-only code change that records backend import, bridge
  provider, LoRA config, Megatron config, distributed runtime initialization,
  JIT fusion warmup, model build/load/wrap, optimizer, and final stack setup.
- No startup behavior has been optimized yet.

### 2026-08-25 10:25 PDT - instrumentation validation

- Diagnostic trainers commit: `41b0e63335b3ccfb10440ad42b35645d92635d79`.
- Ruff, formatting, and type checking passed across all trainers packages.
- Focused B300 unit suite passed: 20 tests in
  `test_init_trainer_server.py`.
- Staged a clean devbox worktree at the exact commit and initialized its Bridge
  and Megatron-LM submodules. The existing dirty checkout remains untouched.
- The generated devbox-up lifecycle scripts remain unmodified. Their CUDA venv
  is reused while `PYTHONPATH` selects the clean worktree source packages.

### 2026-08-25 10:35 PDT - rejected launch and lifecycle corrections

- Rejected the first launch before recording a baseline. Slurm replaced the
  exported `PYTHONPATH` with `/mnt/baseten-pystartup`, so ranks imported the
  old checkout instead of commit `41b0e6333`.
- The generated health waiter also falsely declared the trainer dead because
  leader-side `pgrep` cannot see processes inside the Slurm job PID namespace.
  The Slurm job and all eight ranks were still running.
- Stopped the rejected run with generated `stop_trainer.sh`; both nodes returned
  to idle GPU state.
- Added a run-specific wrapper around generated `run_trainer_node.sh` that sets
  the clean-worktree `PYTHONPATH` inside the Slurm step.
- Added a run-specific `pgrep` shim for the otherwise-unmodified generated
  waiter. It accepts an active `devbox_trainer` Slurm job as process liveness.
- These are lifecycle corrections only. They do not change trainer startup
  behavior.

### 2026-08-25 10:45 PDT - rejected instrumentation run

- Rejected the next launch because backend phase records were not emitted.
- Root cause: logging setup enables INFO for `trainers_server_interface`, while
  `trainers_server_megatron_bridge` inherits the WARNING root level. Existing
  backend INFO banners are suppressed for the same reason.
- Changed only the timing records to use a dedicated
  `trainers_server_interface.startup` logger. This preserves INFO severity
  without enabling all backend INFO logs.
- The run also exposed that one-node Slurm scheduling can place the trainer on
  the non-SSH node. Generated `wait_trainer_health.sh` supports this via
  `TRAINER_HEALTH_URL`; subsequent runs target the scheduled node address.

### 2026-08-25 10:55 PDT - 8,192-token diagnostic baseline

Commit `31bb52bab`, 1D1M, TP1/PP1/CP1/EP8, warm caches, one 8-GPU B300 node.

- Launch to `/health`: 153 seconds.
- Complete log: `runs/baseline_31bb52ba/trainer_srun.log`.
- Structured result: `runs/baseline_31bb52ba/summary.json`.

Slowest rank per phase:

| phase | seconds |
|---|---:|
| distributed runtime and process groups | 19.535 |
| JIT fusion setup/warmup | 3.807 |
| model build, checkpoint load, LoRA, and DDP wrap | 8.733 |
| optimizer | 2.906 |
| final stack setup | 0.079 |
| startup forward+backward | 92.918 |
| warmup zero-grad and final barrier | 0.002 |

The slowest backend rebuild rank completed in 32.332 seconds. Approximately 28
seconds of launch-to-health elapsed before backend rebuild or between rebuild
and HTTP readiness. At 8,192 tokens, startup forward+backward dominates at
60.7% of total launch time.

This is a diagnostic stress baseline, not the production baseline. The trainer
defaults `BT_WARMUP_SEQ` to 64. The next run removes the explicit 8,192-token
override so optimization decisions reflect the real startup contract.

### 2026-08-25 11:05 PDT - production-default baseline

Commit `31bb52bab`, 1D1M, TP1/PP1/CP1/EP8, warm caches, one 8-GPU B300 node,
default 64-token startup warmup.

- Launch to `/health`: 112 seconds.
- Complete log: `runs/baseline_default64_31bb52ba/trainer_srun.log`.
- Structured result: `runs/baseline_default64_31bb52ba/summary.json`.

Slowest rank per phase:

| phase | seconds |
|---|---:|
| distributed runtime and process groups | 10.659 |
| JIT fusion setup/warmup | 2.848 |
| model build, checkpoint load, LoRA, and DDP wrap | 8.061 |
| optimizer | 3.855 |
| final stack setup | 0.074 |
| startup forward+backward | 68.003 |
| warmup zero-grad and final barrier | 0.001 |

The slowest backend rebuild rank completed in 21.324 seconds. Approximately
22.7 seconds elapsed outside backend rebuild and warmup. Startup warmup remains
the dominant phase at 60.7% of launch-to-health.

The 8,192-token run's warmup was 92.918 seconds. Increasing the token count by
128x added only 24.915 seconds, so most of the default warmup is fixed one-time
kernel compilation and communication initialization rather than token compute.

The warmup cannot be removed without replacing its safety function. Git history
documents the original failure: the first real backward crashed while compiling
backward kernels, and asymmetric cold compilation could exceed the NCCL watchdog
timeout. The optimization target is therefore reusable compilation/cache state,
not deferring warmup until after `/health`.

### 2026-08-25 11:20 PDT - full-model profiling decision

The debug profile identifies fixed costs but cannot rank full-model work. The
following phases can scale differently with 78 layers and the 800B FP8
checkpoint:

- model module construction;
- FP8 checkpoint I/O, dequantization, conversion, and distributed copies;
- LoRA module matching and adapter allocation;
- CUDA placement and DDP registration;
- forward/backward compute after kernel compilation.

Per Jack's correction, the next measurement is the full production-shaped
GLM-5.2 model before any optimization. The debug model remains the iteration
vehicle only after that run identifies the dominant scalable bottleneck.

Full-run contract:

- model: `zai-org/GLM-5.2-FP8` from the warm team HF cache;
- topology: 2 nodes x 8 B300, TP1/PP2/CP8/EP8;
- max sequence length: 131,072;
- startup warmup: production default 64 tokens;
- full forward+backward safety gate retained;
- additional nested timers split model construction, checkpoint
  load/dequantization, LoRA application, and post-hook CUDA/DDP wrapping.

### 2026-08-25 11:55 PDT - full-model baseline result

Commit `f7e8bc9c6`, full GLM-5.2-FP8, TP1/PP2/CP8/EP8, warm team HF cache,
2 nodes x 8 B300, default 64-token startup warmup.

- Approximate launch to `/health`: 783 seconds (13:03).
- Complete log: `runs/full_baseline_f7e8bc9c/trainer_srun.log`.
- Structured result: `runs/full_baseline_f7e8bc9c/summary.json`.
- All 16 ranks reached `/health`; no startup failure or OOM.

Slowest rank per phase:

| phase | seconds | scaling interpretation |
|---|---:|---|
| bridge/provider config | 5.827 | mostly fixed/config-sized |
| distributed runtime/process groups | 14.541 | topology-sized |
| JIT fusion setup | 3.094 | mostly fixed |
| full model construction | 25.522 | layer/module-scaled |
| composed checkpoint load/dequantization + LoRA apply | 449.614 | parameter/layer-scaled |
| post-hook CUDA/DDP wrap | 0.709 | parameter traversal, small here |
| optimizer, including collective skew | 76.441 | mostly waiting on slower load ranks |
| full startup forward+backward | 263.153 | fixed compile + layer compute |
| final warmup cleanup/barrier | 6.211 | rank skew |

The backend rebuild converged at 493.312 seconds on the slowest rank. The
largest phase by far is the composed pre-wrap hook at 449.614 seconds (57.4%
of launch-to-health), followed by warmup forward/backward at 263.153 seconds
(33.6%). Together they account for approximately 91% of startup.

The nested timer saw one composed provider hook because Megatron Bridge's
`pre_wrap_hook` property combines its internal checkpoint-load and LoRA hooks.
The next instrumentation revision wraps the underlying registered hooks so the
debug iteration can separate FP8 load/dequantization from LoRA traversal before
optimizing the 7.5-minute phase.

This validates Jack's concern about debug-only profiling. The debug model's
model-build/load/wrap phase was 8.061 seconds; the full model's equivalent was
471.356 seconds, a 58.5x increase. Fixed initialization stayed near the same
10-20 second range and would have looked disproportionately important on the
debug proxy.

### 2026-08-25 12:10 PDT - faithful iteration proxy correction

The existing 1D1M debug snapshot stores random BF16 tensors and removes
`quantization_config`. It cannot reproduce the full baseline's dominant FP8
checkpoint I/O/dequantization path, so it is not a valid optimization proxy for
the 449.614-second pre-wrap phase.

The next iteration proxy is a real-weight FP8 1D1M snapshot built from the same
cached GLM-5.2-FP8 checkpoint:

- production dense layer 0 remains layer 0;
- production MoE layer 6, which owns a complete DSA indexer, becomes layer 1;
- embeddings, final norm, LM head, tokenizer, FP8 weights, and
  `weight_scale_inv` tensors are preserved byte-for-byte;
- hidden, attention, expert, vocabulary, and quantization dimensions remain
  unchanged;
- MTP is disabled because its transformer layer is not retained.

Builder: `scripts/build_fp8_debug_snapshot.py`.

Build result:

- remote path: `/root/.cache/user_artifacts/glm52-debug-fp8-1d1m`;
- 1,595 tensors;
- 13.12 GiB tensor payload, 14 GiB on disk;
- source snapshot: `ba978f7d347eaf65d22f1a86833408afdb953541`.

### 2026-08-25 12:40 PDT - CPU-thread startup sweep

The assigned B300 node exposes 256 logical CPUs to the Slurm task. Torchrun
defaults to `OMP_NUM_THREADS=1`; eight ranks therefore use only eight CPU
threads during CPU-bound FP8 conversion and CuTe compilation.

Matched faithful-FP8 1D1M results:

| OMP threads/rank | launch to health | FP8 checkpoint load | startup F+B |
|---:|---:|---:|---:|
| 1 | ~192s | 18.008s | 113.380s |
| 8 | 128s | 6.765s | 71.799s |
| 16 | 112s | 4.810s | 63.494s |
| 32 | 128s | 6.899s | 77.038s |

OMP16 is the clear winner. Relative to OMP1:

- launch-to-health improves approximately 42%;
- checkpoint load/dequantization improves 73%;
- cold forward/backward improves 44% because CuTe host compilation also uses
  the additional CPU parallelism.

OMP32 saturates all 256 logical CPUs across eight ranks and regresses, consistent
with SMT/compiler contention. The production change therefore computes physical
cores per local GPU rank (`logical_cpus / (2 * NUM_GPUS)`), caps at 16, floors
at 1, and preserves an explicit `OMP_NUM_THREADS` override.

Structured result: `runs/omp_sweep_summary.json`. Raw logs are under
`runs/debug_fp8_{baseline,omp8,omp16,omp32}_59633080/`.

### 2026-08-25 12:50 PDT - pre-dequantized checkpoint estimate

At OMP16, the existing BF16 1D1M snapshot's checkpoint hook took 4.635 seconds,
versus 4.810 seconds for the faithful FP8 snapshot. The 0.175-second (3.6%)
checkpoint-phase difference is small because the BF16 file is larger and OMP16
already makes FP8 dequantization cheap.

Conclusion: a full pre-dequantized BF16 artifact is not the current priority.
It would consume roughly twice the checkpoint storage and likely save only
seconds or low tens of seconds after the OMP fix. Revisit only if the optimized
full-model run contradicts this proxy.

### 2026-08-25 13:00 PDT - first optimization implementation

Trainer commit `4c159e40c` adds a bounded launcher default for CPU-bound startup
work:

```text
threads_per_rank = clamp(logical_cpus / (2 * local_gpu_ranks), 1, 16)
```

An explicit `OMP_NUM_THREADS` is preserved. The factor of two maps SMT logical
CPUs to physical cores and the cap encodes the measured OMP16 knee.

Validation before commit:

- 7 focused launcher tests passed, including computed bounds and override
  preservation;
- Ruff and formatting passed;
- all trainers package type checks passed;
- pre-push `make check` passed;
- branch pushed to `origin/jack-optimise-trainer-startup`.

The run-specific devbox wrapper mirrors the same calculation because the
generated lifecycle script's shell entrypoint is rooted in the pre-existing
checkout; Python package imports still point at the clean branch worktree.

### 2026-08-25 13:15 PDT - optimized full-model validation

Commit `4c159e40c`, full GLM-5.2-FP8, TP1/PP2/CP8/EP8, warm team HF cache,
2 nodes x 8 B300, default 64-token startup warmup, computed OMP16.

- Approximate launch to `/health`: 418 seconds (6:58).
- Baseline: 783 seconds (13:03).
- Improvement: 365 seconds, 46.6% lower startup latency (1.87x faster).
- Complete log: `runs/full_optimized_4c159e40/trainer_srun.log`.
- Structured result: `runs/full_optimized_4c159e40/summary.json`.
- All 16 ranks reached `/health`; full forward+backward warmup retained; no OOM
  or startup failure.

Phase comparison:

| phase | baseline | optimized | change |
|---|---:|---:|---:|
| checkpoint load/dequant + LoRA | 449.614s | 101.155s | -77.5% |
| model build/load/wrap total | 471.356s | 129.199s | -72.6% |
| backend rebuild cumulative | 493.312s | 160.828s | -67.4% |
| startup forward+backward | 263.153s | 231.263s | -12.1% |
| launch to health | ~783s | ~418s | -46.6% |

The optimized full checkpoint phase confirms that CPU threading, not a
pre-dequantized BF16 artifact, is the highest-return change. The remaining
dominant phase is the 231-second full-model safety warmup. Removing or deferring
it is not acceptable because it protects against real backward-compile and NCCL
watchdog failures. Persisting/AOT-compiling the CuTe DSA kernels is a separate,
larger dependency project; the current branch does not take that risk.

### Final code state

Trainer branch `jack-optimise-trainer-startup` contains:

- `41b0e6333`: structured per-rank startup timing;
- `31bb52bab`: visible startup timing logger;
- `f7e8bc9c6`: model startup sub-phase timing;
- `596330802`: separate registered checkpoint and LoRA hook timing;
- `4c159e40c`: bounded CPU parallelism for startup.

All trainer commits are pushed. The full repository pre-push check passed on
every code commit. The devbox trainer was stopped after evidence capture and no
`devbox_trainer` Slurm job remains.

### Residual opportunity

CuTe diagnostic logging confirmed a 0% cross-process JIT cache hit rate and one
DSA backward `build_ir_module` call of approximately 46.7 seconds per rank. The
raw diagnostic log is retained locally at
`runs/cute_profile_default64_31bb52ba/trainer_srun.log` (475 MiB, intentionally
not committed). A future CuTe AOT/persistent-cache project could target part of
the remaining 231-second warmup, but it requires dependency-level artifact
versioning and correctness validation and is not a safe extension of this patch.
