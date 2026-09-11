# Migration to tj-q9exk9w

User requested the SYS_ADMIN devbox instead of the raw Kubernetes profiler pod.

- New job: `tj-q9exk9w`, 8 HGX B300 / NVIDIA L20D, driver580.105.08.
- Effective capability mask `0xa82425fb` includes CAP_SYS_ADMIN.
- Same Ubuntu24.04 / Python3.12.3 ABI.
- Old venv copied verbatim: Torch2.11.0+cu130, Transformer Engine2.16.0.
- Copied 8.90 GB venv and 92.40 GB GLM experiments/checkpoints/code directly
  pod→devbox via rsync. No full checkpoint re-download; the full GLM snapshot
  is already readable from the shared team cache.
- Checksum dry-run over all material source paths found no file-content
  differences (only `/root` directory timestamp differed).
- Generated 1d2m/1d3m/2d2m checkpoints remain at the same `/root/glm53-*` paths.
- `/root/trainers`, PR1355 checkout, old run artifacts, kernel caches and
  `lmhead-candidate.py` preserved. Profiler setup preserved; default old NCU
  Documents sections archived under `/root/glm53-preserved-profiler-docs-20260911`.
- Qwen scratch was not transferred, per the user's earlier instruction that
  Qwen is no longer needed. Shared cache mounts were not deleted or modified.

## Installed profiler version

Use the devbox's **built-in Nsight2025.3.1.0**, per the user's instruction.
`/usr/local/bin/nsys` is restored to its original target under the Nsight Compute
installation. The initial diagnosis of a missing importer was too broad:
QdstrmImporter existed but lacked system `libdw.so.1`. Installed `libdw1t64` and
its `libelf1t64` dependency; fresh built-in captures now import/export correctly.

A complete old Nsight2025.3.2 SDK was copied during investigation but is NOT
the active profiler and is not required by the workflow.

Validation of `builtin-fixed.sqlite`: **1000 CUDA kernels, 14395269 metric
samples, all8 physical GPUs represented**. NCU successfully collected a kernel
report over10 passes. BF16 matmul result finite.

The temporary personal public-key authorization used for direct migration was
disabled afterwards; normal `ssh tj-q9exk9w` certificate access remains intact.
Old profiler pod deletion was requested only after data and profiler validation.
