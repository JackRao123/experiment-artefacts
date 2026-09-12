# EP1 optimization follow-up

Active, requested after the prior Nsight comparison was completed.

## Current user override — supersedes every longer validation plan below

The user explicitly stopped the55-step validation approach. No more stability
loops or optimizer steps. Use `single_fb.py`: one untimed FB warmup, then one
measured FB. `--profile` captures that same measured FB, not an extra request.
EP8 V2 had already finished before this instruction. EP1 V2 is still loading;
its upcoming capture must use the new single-FB driver, not `capture.py`.

Update: EP1 V2 completed that single-FB protocol, optimizer step stayed0.
Warmup69.6007s; measured profiled FB12.4625s /1314.66TPS/GPU. Collecting and
analyzing its one trace now. This changed warmup/optimizer-reset/profiling
contract is NOT a directly comparable delta versus prior unprofiled controls.
No more benchmark calls have been queued. EP1 remains loaded while inspecting.

Next targeted candidate after inspecting that trace: disable expandable CUDA
allocator segments. The single-FB trace contains449ms in cuMemSetAccess on
rank2 and substantial cuMemCreate/cuMemMap stalls. The analyzer incorrectly
omitted cuMemSetAccess; its taxonomy is now fixed. Stop completed EP1 V2 and
launch `devbox-ep1-te-metadata-v2-noexpand` with the same source and
`single_fb.py` only (one warmup + one measured/profiled FB, zero optimizer).
This is a matched allocator ablation of the new single-FB contract, not a
comparison against the old five-control/optimizer-reset runs.

## Measurement contract

Full GLM5.3, 131072 tokens, 8 HGX B300, CP8 with EP1 or EP8, BF16
expert storage, TE expert kernels, LoRA32, full one-layer recompute.
Same restored checkpoint, input hash, old venv and built-in Nsight as the
previous `nsys_workflow_20260911` sweep. Controls exclude optimizer time.
Do not modify tests. Implementation belongs in PR1355 and its dependencies;
scripts, profiles and findings belong in this separate artifact repository.

## Evidence and candidates

- `idle_breakdown.py` reads the existing all-rank timing SQLites. EP1 idle
  1.22–1.75s/rank; EP8 0.23–0.29s. EP1 has 0.62–0.78s in material gaps
  preceding collective launches, often while `fsdp_gather:expert` host
  scopes are active, and 0.32–0.37s around its final host-only tail.
  These are scope/next-launch associations, not complete causal attribution.
- Source rebuilds per-parameter views in `DataParallelBuffer.fetch_bucket`
  and resolves full dotted module paths in raw/distributed parameter swaps.
- CPU-only replay of exact committed swapping methods (`metadata_microbench.py`):
  38400 scalar expert parameters, median raw->dist 176.51->41.36ms,
  dist->raw 179.67->41.52ms. Tied aliases and identities verified. Not TPS.
- Trainer commit327dff77c adds GC and FSDP host-method NVTX scopes.
- Core commit6851f7895 adds opt-in static parameter owner and persistent
  frozen BF16 view caching. Bridge33ccd9cb4 pins it. Full-model validation pending.
- GC-freeze-only case: `nsys_workflow_20260911/devbox-ep1-te-gcfreeze`.
  Completed on source57b9a3ef4/Corefe3976282, metadata cache off:
  11.7847s mean FB, SD0.2569s,1390.28 TPS/GPU. Not a demonstrated gain.
  Both raw traces/SQLite exports are SHA256 verified on the Mac.
- Completed screen: `nsys_workflow_20260911/devbox-ep1-te-metadata`,
  on trainer734081da3/Bridge33ccd9cb4/Core6851f7895. GC freezing off;
  metadata cache on. Mean11.3441s, SD0.2218s,1444.27 TPS/GPU (5controls).
  Timing/metrics finalized; collecting and analyzing locally.
- Completed continuation: `devbox-ep1-te-metadata-validation`,50 further
  controls and one timing capture on the same loaded model. This starts at
  optimizer step10, so do not treat it as a fresh-model matched A/B.
  All50 finite, peak248.10GiB unchanged. Last10 validation mean10.9647s,
  1494.25TPS/GPU; these later training-state numbers are not the initial headline.
- V2: Core963a47acd deduplicates per-parameter waits/releases into per-bucket
  operations under the same opt-in flag. Recorder replay preserves first-use
  bucket transitions and FP8 processing order;768params/3buckets requires3
  rather than768 wait/release calls. Trainer c699f8990, Bridge28622cfb7.
- Completed `devbox-ep8-te-metadata-v2`, now switch to EP1 on exactly the same
  source. Each gets3warmups,5headline controls, timing+metrics,45additional
  validation controls. This preserves comparable initial profiling windows and
  completes55optimizer steps per layout. GC freezing stays off.
  EP8 V2 initial controls: mean10.6553s, SD0.3703s,1537.64TPS/GPU.
  All55optimizer steps finished; both traces are finalized. Refreshing the
  previously partial local manifest and analysis with the completed validation.
  The EP8 source is unchanged and its lifecycle is stopping before EP1 launches.
  Do not mutate remote HEAD while a case runs: manifests read HEAD.

## Infrastructure

`ssh tj-q9exk9w`, SSH master `/tmp/glm53-q9-ssh.N8ccbW/live`.
Remote runner `/root/glm53-fsdp-nsys-131k-20260911`.
Remote trainer `/root/glm53-pr1355-repro-20260910/trainers`.
Use generated lifecycle copies and manually inspect each180s waiter checkpoint.
No new pods or package changes. Previous follow-up automation remains paused.

## Validation scope

No production or convergence claim. Full-model finite-loss controls and profiles
are useful before recommending the candidate. The user explicitly rejected the
skill's long-run validation gate here. Focus on single-FB TPS; do not restart it.
