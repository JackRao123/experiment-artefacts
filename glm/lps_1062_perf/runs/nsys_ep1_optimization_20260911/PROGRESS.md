# EP1 optimization follow-up

Active, requested after the prior Nsight comparison was completed.

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
- Active case: `nsys_workflow_20260911/devbox-ep1-te-metadata`, starting
  on trainer734081da3/Bridge33ccd9cb4/Core6851f7895. GC freezing off;
  metadata cache on. Generated health waiter running. Capture still pending.
  Do not mutate remote HEAD while a case runs: manifests read HEAD.

## Infrastructure

`ssh tj-q9exk9w`, SSH master `/tmp/glm53-q9-ssh.N8ccbW/live`.
Remote runner `/root/glm53-fsdp-nsys-131k-20260911`.
Remote trainer `/root/glm53-pr1355-repro-20260910/trainers`.
Use generated lifecycle copies and manually inspect each180s waiter checkpoint.
No new pods or package changes. Previous follow-up automation remains paused.

## Validation scope

No production or convergence claim. Full-model finite-loss controls and profiles
are required before recommending the candidate. A durable winner should receive
the repository skill's50-step validation, reporting a declared steady window.
