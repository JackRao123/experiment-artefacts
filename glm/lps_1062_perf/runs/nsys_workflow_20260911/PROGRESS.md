# Work in progress

Update~21:36 UTC: devbox-ep1-te finally became healthy without restart after
long startup. Now running capture.py (3warmups/5controls/1timing/1metrics).
Native debugging was inconclusive: installing gdb upgraded libc from .7 to .9
while target was alive, initial native symbols mismatched; copied original
mapped libc for diagnosis but do not treat bad frames as root-cause proof.
Target eventually reached Uvicorn. No startup/debug time is in measured windows.
Record that libc was updated during setup if interpreting tiny CPU differences.

Latest (21:13 UTC): `devbox-debug` on tj-q9exk9w completed warmup/control,
timing and metrics. The single validation control2.028s was NOT steady state;
subsequent timing0.653s/metrics0.664s. Do not present its control as benchmark.
Formal full `devbox-ep1-te` has been initializing since~20:49 UTC. Representative
workerPID25892 was sampled in active HF checkpoint loading (`_load_cached_tensors`
reader-cache eviction); no training controls yet. Use generated health waiter,
manually reinvoked at180s checkpoints. Current waiter session4425 may have returned;
read its state before reinvoking. Model/venv/source are correct, GPU memory~202GB.

Analyzer now schema2: expert host-scope idle, allocator/VMM idle, partial event
dependencies, guarded prefetch-readiness, warning summaries, cached results.
`collect.py` + `workflow.py collect` added: finalized manifest, rsync, SHA checks,
optional local analysis. capture.py now writes capture_complete/artifact hashes.
These latest driver/collect/README changes still need committing; last artifacts
push f2d973d (some subsequent changes uncommitted). Trainer PR1355 remains13137ef1a.

## Latest infrastructure override (2026-09-11 ~20:38 UTC)

User provided `ssh tj-q9exk9w` (8 B300, SYS_ADMIN) and instructed us to use
its BUILT-IN profiler version, not pin old Nsight. Migration complete and
old raw Kubernetes profiler pod DELETED (confirmed NotFound/empty listing).
Use ONLY new devbox now. Old localhost2227 port-forward is obsolete.

New devbox direct SSH works; persistent foreground control master is tool
session31376, socket `/tmp/glm53-q9-ssh.N8ccbW/live`. Prefer:
`ssh -S /tmp/glm53-q9-ssh.N8ccbW/live tj-q9exk9w ...`.
The older `mux` socket was from a detached master that did not persist.

Same root paths copied verbatim (8.9GB old venv +92.4GB GLM data/code),
checksum verified. Full native checkpoint remains shared, no re-download.
Source on new node remains13137ef1a; old venv Torch2.11/TE2.16 works.
Built-in nsys3.1 was initially missing OS libdw/libelf, NOT its importer.
Installed those libraries; restored `/usr/local/bin/nsys` to original bundled
3.1 path. Fresh built-in capture verified1000 CUDA kernels and14395269 metric
samples across8 GPUs. NCU verified. Copied old3.2 SDK is NOT active.
See devbox_validation/MIGRATION.md. Temporary migration public key disabled.

`prepare_devbox.py` uses the new box's generated lifecycle templates, retaining
old copied venv/source. Cases now: `devbox-debug` (1d2m, EP8, grouped),
`devbox-ep1-te`, `devbox-ep8-te`, `devbox-ep1-grouped`, `devbox-ep8-grouped`.
All event tracing OFF; 3warmups/5controls/1timing/optional1metrics.
All staged on new devbox. Currently launching `devbox-debug` for full trainer
verification on new box, then resume full matched matrix. Original experiment
request remains unfinished; infrastructure switch did NOT cancel it.

Preliminary old-node EP1 exact result:11.583772790s,1414.392383 TPS/GPU,
248.10GiB. EP8:10.583087509s,1548.130447 TPS/GPU,213.62GiB.
Both raw timing reports and analyses now on Mac, SHA verified.
`preliminary-comparison.md` exists. EP1 expert GPU union2.161s vs EP8 1.279s
(rank0); expert exclusive time is misleading here because EP1 overlaps
FSDP gathers. EP1 attention4.473s vs EP8 4.046s, idle~1.5s vs~0.26s.
Old communication taxonomy incomplete; use new coalesced annotation for formal
breakdown. Old EP1 metrics pass invalid/stalled and intentionally excluded.

Task: repeatable nsys capture/analysis and a full-model FSDP CP8EP1 versus CP8EP8
comparison at 131072 tokens, BF16 experts, LoRA32, full recompute, 5 controls.
Original drivers remain pristine. No tests added or modified.

## Source

- Trainers PR1355 branch `jack/glm-dsa-131k-perf`, worktree
  `/Users/jackrao/Documents/trainers-wt-glm-dsa-131k`.
- Current pushed trainer commit `13137ef1a`: EP enabled for experimental FSDP;
  rank/layer/recompute/expert-GEMM/shape/collective NVTX labels. Coalesced FSDP
  flush and backward CP boundaries included. `make check` passed.
- Bridge unchanged `60b1570fbbcf52961f64df5794c8ed1256279754`;
  Core unchanged `cf81782b23150362b761888277866ec1e758cde3`.
- Pod checkout `/root/glm53-pr1355-repro-20260910/trainers` at current commit.
- Workflow committed to separate experiment-artifacts repo in `0aa4d96`;
  subsequent workflow hardening/results still need the next commit.

## Runs

- `debug-ep8`: initial TE capture validated all ranks and phase accounting.
- `ep8-te`: full preliminary baseline, 5 controls, **1548.130 TPS/GPU**, mean
  FB **10.583088 s**. Three timing FBs complete; separate metrics FB complete.
  All artifacts copied to Mac and archive SHA verified:
  `08c0464cf7ccd6f2198ef9f04ec831d7b5c2c57058dad692af0d923e8913129d`.
  Timing analysis cold took 44 s, metrics 107 s; cached rerun is immediate.
- `ep1-te`: full preliminary controls ~1414 TPS/GPU and 3 timing FBs complete.
  Optional subsequent metrics FB stalled; see `FAILED_METRICS.md`. No valid
  metrics report. Rank4 PID59428 stack in FSDP coalesced gather during recompute.
  Clean archive retrieved via rsync, expected SHA
  `8c6c401cf78bd0247f92fddaae43f2c028ad13323eeae5098a0d3a0be330e2ea`.
- Preliminary full runs used trainer178ff025f and CUDA event tracing on.
  Their FSDP/CP communication split is incomplete due coalesced launch scopes.
- `v2-debug-ep8-grouped`: current annotations, event tracing OFF; both timing
  and all-GPU metrics completed. Analyzer cleanly separates CP and FSDP and
  no rank0 unclassified collective category. Grouped-MM works with EP8.
- **Currently initializing `v2-ep1-te`**, current source/event tracing OFF.
  Planned matched current-source cases: `v2-ep1-te`, `v2-ep8-te`,
  `v2-ep1-grouped`, `v2-ep8-grouped`. All configs staged on pod.
  v2 protocol: 3 warmups, 5 controls, 1 timing FB, optional separate 1 metrics FB.

## Access / lifecycle

Profiler pod remains `jackrao-profiler-b300`, namespace
`org-99340d71961343c28c5c567d705ab0c0`, kubeconfig
`/Users/jackrao/.kube/ali-apse7-prod-1.yaml`. Original devbox remains stopped.
Remote run `/root/glm53-fsdp-nsys-131k-20260911`.

Local port-forward2227→pod22 was restarted; SSH works:

```
ssh -i /Users/jackrao/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ServerAliveInterval=15 -p2227 root@127.0.0.1
```

Use each case's generated `.devbox_up/{start,wait,stop}_trainer.sh` lifecycle.
Launch via `CASE/launch.sh`. Invoke waiters manually at their 180s checkpoints.
Stop scripts now bound nsys shutdown and identify owned workers via the exact
exported trainer-config path (nsys stdout is a pipe). Launch checks GPU idle.
The previous stalled run's processes and GPU contexts were verified gone.

Capture on pod: `/root/.devbox-venvs/server/bin/python -u RUN/capture.py CASE --metrics`.
Capture rejects overwriting existing benchmark.json; save failures separately.
Profiled-request timeout is max(60s,5×control median); warmups retain long timeout.

Use resumable `rsync -a --partial -e 'ssh ... -p2227'` for results. Raw `kubectl cp`
timed out repeatedly and even returned exit0 on a missing source. Validate hashes.
Partial failed directory `ep8-te-remote` is ignored and not used for analysis.

## Analysis caveats

No GPU duration sums are used as wall time. Exclusive category time + mixed
overlap + idle exactly reconciles. Partial CUDA-event dependency reconstruction
excludes ambiguous/inconsistent edges. Runtime-correlation coverage is separate
from source-callstack coverage. Diagnostic warnings must be reported, including
"Not all CUDA/NVTX events might have been collected" on trainer processes.
Nsight warning timestampType=2 is HOST clock, not GPU/target timeline clock.

Per-layer expert forward/recompute/backward accounting was checked: all 75 MoE
layers ×3 preliminary steps present for each phase. EP8 expert Tensor Active
~91–93% during expert kernels; not whole-step utilization. Failed EP1 metric
pass must not be used. Source-change comparisons require explicit audited note.
