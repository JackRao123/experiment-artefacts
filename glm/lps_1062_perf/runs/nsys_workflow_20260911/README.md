# Repeatable Nsight workflow

Current machine: **`ssh tj-q9exk9w`**, using the old copied venv and its built-in
Nsight2025.3.1 after installing the missing libdw/libelf system libraries.
The Kubernetes profiler pod has been deleted after checksum-verified migration.
Use `prepare_devbox.py` and `devbox-*` cases for new runs; `v2-*` is the earlier
profiler-pod setup. See `devbox_validation/MIGRATION.md`.

Single entry point, no SQL and no AI required for repeat analyses:

```bash
python3 workflow.py collect devbox-ep1-te --host tj-q9exk9w --analyze
python3 workflow.py collect devbox-ep8-te --host tj-q9exk9w --analyze
python3 workflow.py compare devbox-ep1-te/timing.analysis.json devbox-ep8-te/timing.analysis.json --output comparison.md
```

If the checkpoint was restored under a different cache root, add
`--checkpoint-relocation-note "Exact HF snapshot restored to node-local disk after shared-cache removal"`
to `compare`. This still requires identical HF repository and snapshot hashes;
it does not permit arbitrary model substitutions. Preserve completed configs.
Prepare only pending cases with `prepare_devbox.py --case CASE --base-model SNAPSHOT`.

`analyze` also accepts `.nsys-rep` and exports with `nsys` when SQLite is absent.
SQLite analysis uses only Python's standard library and works on the Mac.
Unchanged inputs/analyzer are cached. First ingestion is not instantaneous.
`collect` checks a finalized capture manifest, resumes transfers with rsync,
verifies SHA256 and sizes, and optionally runs the analysis. Partial failed
captures require explicit `--allow-partial`; only finalized files are retrieved.
For files already local: `python3 workflow.py analyze TRACE.sqlite --benchmark CASE/benchmark.json`.

## Capture contract

- GLM 5.3 full model, 131072 tokens, one identical synthetic datum, seed 0xB300.
- LoRA rank/alpha 32, BF16 expert storage, full one-layer recompute, TP1/PP1/ETP1/CP8.
- Megatron FSDP, persistent double buffers and parameter prefetch on.
- Three warmups, five unprofiled controls, one timing-traced FB request.
- Separate one-request GPU-metrics pass; 10 kHz all-GPU sampling.
- Timing: CUDA/NVTX, CUDA event tracing **off**, no CPU sampling or call stacks.
- Head chunk 4096, memory-efficient head off, CUDA graphs off.
- Controls include optimizer calls, but TPS uses FB duration only, as historically.
- Collection is off during controls, although the nsys launcher is present.
- Trainer implementation and annotation changes are committed in PR1355;
  no run-local source patches. Original `tools/profile_driver.py` and `mfu.py` untouched.

`prepare.py` generates configs and adapts copies of devbox-up's generated lifecycle.
Start with each case's `launch.sh`, run its generated health waiter, then on the pod:

```bash
python workflow.py capture devbox-ep8-te --metrics
```

Before stopping, close only that case's nsys session, then use its generated stop script.
Keep node-local checkpoint copies safe; the team-cache write issue remains waived.

## Preliminary versus v2 captures

`ep1-te` and `ep8-te` are the preliminary, event-tracing-enabled runs (three
timing steps each). Their controls completed. The subsequent EP1 hardware-metrics
request stalled in FSDP recompute; see `ep1-te/FAILED_METRICS.md`. Its failed
metrics pass is not a performance measurement.

`v2-*` uses corrected coalesced-collective and backward CP annotations,
CUDA-event tracing off, one timing step, and a bounded profiled-request deadline.
Event tracing can introduce false cross-stream dependencies according to nsys
help; disabling it is a mitigation, not proof of the stalled run's root cause.
Keep formal comparisons within the same capture settings and source revision.
The revised small debug run completed both timing and all-GPU metrics captures.

The stop script identifies workers by the exact exported trainer-config path
as well as stdout, because nsys replaces stdout with a pipe. Launch refuses to
run while any GPU process is present. Large closed artifacts should be retrieved
with resumable rsync and SHA256 verification; raw `kubectl cp` repeatedly timed out.

## What the report means

The analyzer identifies ranks from NVTX markers, links CUDA launches to enclosing
CPU scopes by thread and process, then attributes their asynchronous GPU work.
All elapsed GPU times use interval unions, not sums across streams. Each rank's
exclusive categories, mixed-category overlap and idle exactly reconcile to FB time.
Forward, actual checkpoint recomputation and real backward are separate.

The report ranks **observed costs**, not guaranteed removable milliseconds.
Communication exposure is an upper bound, not proven blocking communication.
Partial CUDA event dependency reconstruction follows recording instances by
`eventSyncId`, including relayed waits. Ambiguous/missing events and prerequisites
that finish after their supposed consumer starts are excluded. The linked blocking
figure is incomplete, not a claim to resolve every driver or dispatcher dependency.
Unclassified collectives remain unclassified rather than guessing CP/EP/FSDP from
their kernel name. Dispatcher occupancy includes packing/synchronization.

GPU metrics use explicit physical GPU/rank mapping, resolve metric names, and
report both all-resident and category-exclusive samples. Absent/unsampled values
are unknown. Sampling is device-wide, so overlapping operations contaminate
non-exclusive samples. These are not Nsight Compute kernel roofline measurements.

The comparison validates workload/config/input/source compatibility and reports
same-run capture slowdown. An EP1-vs-EP8 comparison alone cannot establish the
causal gain of a different GEMM implementation; the paired TE/grouped-MM cases
are included for that ablation if both full-model configurations run successfully.

## Topology interpretation

On eight GPUs, EP1 has expert-DP size eight: FSDP gathers expert weights.
EP8 has expert-DP size one: 32 experts reside on each rank and there is no
cross-rank expert FSDP gather. Non-expert parameters remain FSDP-sharded across CP8.
EP8 is therefore not merely EP1 with a dispatch switch: expert ownership,
GEMM row counts, weight traffic and imbalance all change.
