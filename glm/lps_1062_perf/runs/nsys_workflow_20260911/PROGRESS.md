# Work in progress

## Current state: six captures complete; EP8 async remains

Latest final-case waiter: tool session27178 (session5013 ended at its180s
checkpoint). EP8 async ranks alive, checkpoint import progressing, no failure
reported. Continue that waiter; do not launch a second trainer or duplicate
capture. All EP1 async artifacts are finalized, verified and analyzed locally.

EP1 async all artifacts copied/SHA verified/analyzed. Offset-upload stream syncs
are ZERO on all8 ranks (old grouped300/rank). TPS flat1379.9 versus1384.1 TE
repeat and1389.8 unfixed. Less GPU idle but longer overlapping GPU/collective
residence; no demonstrated end-to-end gain. `ep1-offset-upload-comparison.md/json`
generated with source-change audit. EP1 async stopped; launched final
devbox-ep8-grouped-async (pgid105385), generated waiter started. Source unchanged
57b9a3ef4 / a3438227 / fe3976282. After last capture: collect/analyze, commit all
task changes, append final table/caveats to latest PR1355 description, verify
the appended text, then pause the heartbeat. Do not delete active devbox.

Async EP1 capture finalized. Controls mean11.873678703 s, SD0.200223714 s,
TPS/GPU1379.858796, peak247.977826 GiB. No measurable improvement versus TE
repeat1384.1 or unfixed grouped1389.8. Collection/analysis started on Mac;
lifecycle stop running. After GPU processes clear, launch already-staged
devbox-ep8-grouped-async on current final source57b9a3ef4. Verify synchronization
removal in EP1 trace after analysis; do not infer it from config alone.


LATEST USER REQUIREMENT: when finished, commit all task work in its appropriate
repository and APPEND the final numbers/caveats to PR1355's existing description
(preserve existing text; no artifacts in trainers). Do not pause the follow-up
until both fixed-path comparisons and that PR-description update are complete.

Async EP1 generated waiter succeeded (116s on latest invocation). Started
capture.py devbox-ep1-grouped-async --metrics; inspect its capture.log before
any new request. Same 3 warmups/5 controls/1 timing/1 metrics protocol.

00:16 UTC status check: async EP1 still loading, ranks alive, ~202617MiB/GPU.
Reinvoked generated health waiter after checkpoint; no new error in latest log.
Do not start controls before health success, and do not restart the live trainer.

00:12 UTC: TE repeat all four artifacts copied, SHA256 verified, both analyses
finished. Fresh comparison files generated. TE repeat allocator coincident
idle max11.6ms (original0.632s pause absent), yet GPU idle1.22–1.75s/rank remains;
timing overhead3.06%. Async EP1 still initializing normally with model memory
allocated. Current generated waiter session70450 (manually reinvoked after
previous180s checkpoint). Next capture.py devbox-ep1-grouped-async --metrics
only after health waiter succeeds; then EP8 async. No other GPU workload.

00:08 UTC: TE repeat stopped. Final source bundles imported with
--no-recurse-submodules; remote repos now exactly trainers57b9a3ef4 /
Bridgea3438227 / Corefe3976282. No tracked diff; preserved existing untracked
venv symlink/compiled helpers. Launched devbox-ep1-grouped-async (pgid98968),
generated waiter running. Do not reapply source updates or duplicate launch.
TE repeat collection still running on Mac (session35120); first3 artifacts
SHA verified, metrics SQLite transfer/analysis pending. All earlier four cases
fully copied/analyzed. Next: capture EP1 async when healthy, then EP8 async,
compare source-changed cases with explicit offset-upload-only audit note.

00:07 UTC: TE repeat capture finalized successfully on old source. Five controls
mean11.837257410 s, SD0.192306351 s, TPS/GPU1384.104395. Collection/analysis
started on Mac, lifecycle stop running. After it exits and GPU inventory clears,
update remote repos from FINAL bundles to trainers57b9a3ef4 / Bridgea3438227 /
Corefe3976282, then launch EP1 grouped async. Preserve all existing artifacts.

Final async source pins are PUSHED: trainers57b9a3ef48f0346cc00cd15eac1819b3e0ed47f0,
Bridgea3438227e442944503df367726e7a3892966ce89,
Corefe39762823e1bfef18f730879169d11b8c396eb0. Root make check passed, no tests
changed. Three incremental *-source-async.bundle files regenerated with these
heads and staged under remote run root. AFTER TE repeat finalizes/stops, fetch
each bundle HEAD in its respective repo and checkout those exact SHAs. Verify
clean trees and all three HEADs before launching devbox-ep1-grouped-async then
devbox-ep8-grouped-async. Configs/lifecycle already staged; same old venv.
Current remote source STILL13137ef1a while TE repeat capture runs (session79076).
New analyzer expert CUDA API breakdown found300 stream synchronizations per
GPU/FB in unfixed EP8 grouped, 1.095–1.247s host API residence across ranks.

00:00 UTC Sep12: TE repeat healthy and capture.py devbox-ep1-te-repeat --metrics
running on OLD remote source13137ef1a (tool session79076). Do not update remote
source before that capture finishes. First async fix propagated to PR1355
5a53a08c021cd3f7e28b506d6d30b1a097990a4f / Bridge8643fe08 / Corecd5dbc9ae;
root make check passed, worktree clean. Core follow-up makes pinned staging
explicitly device=cpu: fe39762823e1bfef18f730879169d11b8c396eb0, pushed PR76.
Need propagate that final Core pin through Bridge PR84 and trainers PR1355,
then REGENERATE/RESTAGE the three *-source-async.bundle files (currently first
fix only). Async case configs/lifecycle already staged. Bridge all-file
read-only lint reports pre-existing errors in unrelated model/test files; no
source/test auto-fixes were made. Core changed-file checks + numerical offset
probe passed. Only a gitlink changes in Bridge/trainers.

23:55 UTC: concrete grouped-MM wiring issue found: rank0 EP8 grouped has300
cudaStreamSynchronize calls totaling1094.694ms in expert_shape forward scopes;
TE has none there. CPU-list-to-CUDA offsets in frozen_grouped_mm.py force a
synchronous H2D transfer before every original/recompute projection. This is
not all recoverable wall time, but prevents enqueue-ahead. Core fix uses pinned
CPU staging plus nonblocking H2D, preserving GPU cumsum and routing semantics.
Committed/signed/pushed to Core PR76: cd5dbc9ae8162b55cda278d4b274b6719743cec6.
Manual GPU offset equality passed four zero/nonuniform/EP1/EP8 shapes; no tests
added/modified. Core Black/isort/Ruff/syntax checks passed. New Bridge worktree
`/Users/jackrao/Documents/mbridge-wt-fsdp-offset-upload` stages new Core gitlink;
read-only pre-commit checks running before Bridge PR84 + trainers PR1355 pins.
DO NOT update remote trainer/source until old-revision TE repeat completes.
Current TE repeat initializing, latest generated waiter session78855.
After repeat: validate and run EP1/EP8 grouped async-offset candidates on new
committed source, same5-control protocol, confirm300 synchronizations disappear.

23:46 UTC: EP8 grouped capture completed successfully. Five controls:
mean11.328007527 s, SD0.297089942 s, TPS/GPU1446.326723,
peak allocated213.491493 GiB. ~6.7% TPS regression versus EP8 TE1550.1.
Collect/analyze started on Mac; lifecycle stop running. Next launch prepared
devbox-ep1-te-repeat only after GPU processes clear, then use generated waiter.
Do not re-run or overwrite the four finalized captures.

23:40 UTC: generated waiter confirms EP8 grouped healthy; capture.py
devbox-ep8-grouped --metrics started. No duplicate requests. After finalization,
collect/analyze, stop via lifecycle, then launch prepared devbox-ep1-te-repeat.
The repeat config is staged remotely and preserves existing baseline artifacts.

23:36 UTC: EP1 grouped analysis complete. Kernel inventory confirms actual
PyTorch/CUTLASS grouped-MM, not fallback. Rank0 expert GPU union2.296s vs TE2.173s;
forward+recompute1.278 vs1.294s; real dgrad1.018 vs0.878s. Host time outside CUDA
APIs inside expert scopes0.163s vs2.394s (not critical-path time). Tensor Active
exclusive77–87% vsTE54–60%; no meaningful end-to-end gain. No late steady expert
prefetches. Add one TE EP1 repeat after EP8 grouped to remove the original TE
allocator/profile anomaly and old mapped-libc confound. New case
`devbox-ep1-te-repeat`, same settings, three warmups/five controls, timing/metrics.
Current EP8 grouped waiter session54941; inspect and manually reinvoke if needed.

23:32 UTC: EP1 grouped-MM completed all captures and five controls.
Mean11.788758565 s, SD0.079326541 s, TPS/GPU1389.798587 (~0.9% above EP1 TE,
within observed variability, not a compelling end-to-end gain). All four artifacts
copied to Mac and SHA256 verified; local analysis running (session71766).
Analyzer now also records expert host time before/outside recorded CUDA APIs.
Re-analyzed both TE timing reports with this additive schema2 extension.
EP1 grouped stopped. Launched EP8 grouped (pgid73526), generated waiter started.
Inspect waiter/process state; do not duplicate launches. Still need full EP8
grouped controls/captures, comparisons and final qualified conclusions.

23:26 UTC: EP1 grouped-MM healthy via generated waiter. Started capture.py
devbox-ep1-grouped --metrics (3 warmups/5 controls/1 timing/1 metrics).
Inspect remote `devbox-ep1-grouped/capture.log`; do not duplicate requests.

23:12 UTC: all EP8 artifacts copied, SHA256 verified, timing/metrics analyses
finished. `devbox-te-comparison.md/json` generated with exact-revision relocation
note. New EP8 expert Tensor Active91.2–92.4%, all category-exclusive samples;
EP1 53.7–59.7%, exclusive fraction~51–57%. Full baseline report updated.
EP1 grouped is alive and initializing. Its first waiter falsely declared death
before nsys/torchrun spawned rank processes (no actual restart needed). Added
30s startup grace to generated waiter copies, preserving180s checkpoints.
Current generated waiter tool session90315; inspect logs before re-invoking.

23:09 UTC: `devbox-ep8-te` completed all windows and finalized all four artifacts.
Five controls: mean10.569394289 s, SD0.218789902 s, TPS/GPU1550.136134,
peak allocated213.616497 GiB. Timing10.6s and metrics10.6s completed normally.
Collect/analyze running on Mac (tool session34520); first three artifacts verified.
EP8 stopped; subsequent GPU-process inventory empty. Launched `devbox-ep1-grouped`
(pgid67207), generated waiter running. Do not duplicate launches/captures.
EP8 and EP1 TE use identical trainer/Bridge/Core SHAs and nsys2025.3.1.
Only HF cache-root relocation needs an audited compare note (exact same revision).
Next: finish collection/analysis, compare baseline TE runs; capture EP1 grouped
once healthy, then EP8 grouped. Full grouped-MM performance remains unmeasured.

23:03 UTC: generated waiter confirms EP8 healthy. `capture.py devbox-ep8-te
--metrics` started (three warmups, five controls, timing and GPU metrics).
Remote output is `devbox-ep8-te/capture.log`. Do not issue duplicate requests.

22:49 UTC: initial EP8 restart failed before Python because copied console
launchers had shebangs targeting the removed old shared devbox worktree.
`relocate_venv_launchers.py` normalized 98 shebangs to the existing node-local
venv, with original launchers backed up. Packages and trainer source unchanged.
`torchrun --help` passed. EP8 relaunched (pgid48006); health waiter tool session
49780. Inspect its output and logs before manually reinvoking at its checkpoint.

22:36 UTC: restoration COMPLETE, 154 files / 755663688736 bytes, exact revision
and all manifest file sizes validated. EP8 launched with the generated lifecycle
(pgid45833); health waiter started. Resume that waiter/check logs before capture.
The restored snapshot is node-local. Do not restart the download or launch a
second trainer. After EP8 is healthy, run `capture.py devbox-ep8-te --metrics`.

`devbox-ep1-te` completed 3 warmups, 5 controls, timing and GPU-metrics captures.
Mean FB 11.900133 s, SD 0.196359 s, 1376.791 TPS/GPU, peak allocated 248.10 GiB.
All four raw artifacts are on the Mac with manifest SHA256 verification.
Latest source is still trainer13137ef1a, built-in nsys2025.3.1 on tj-q9exk9w.

EP8 startup failed because the entire shared `models--zai-org--GLM-5.3` directory
disappeared after EP1. No checkpoint deletion was performed by this workflow.
The shared mount remains healthy; no alternate exact checkpoint was found.
The user does not know where it went. Exact HF revision remains accessible:
`187fb9fff6319062325ff825627ef6db084d9bc6`, 154 files, 755663688736 bytes.
`restore_checkpoint.py` restored it with HF/Xet to **node-local**
`/root/glm53-checkpoints-local/hub`, not refilling the shared cache.
The script checks all file sizes and atomically writes `hub/glm53-ready.json`
only after completion. Do not start a reader before that sentinel exists.
Remote log: `devbox_validation/restore-checkpoint.log`.
EP8 is initializing. Full grouped-MM variants remain pending.

Latest files were committed by another local session in artifact commit08eb757;
do not assume the historical uncommitted-file notes below are current.
See `REPORT.md` for qualified completed measurements. Check git status before edits.

## Historical progress log

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
