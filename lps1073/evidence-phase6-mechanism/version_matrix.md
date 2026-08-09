# LPS-1073 version matrix — cuDNN x TE, CP4 standalone repro (work item C)

> (Originally LPS-1063, pre-split — see the banner at the top of `../NOTEBOOK.md`.)

- lane: cu13 | torch: torch==2.11.0+cu130 | GPUs: 4,5,6,7 (CUDA_VISIBLE_DEVICES) | box: tj-w6xx15w, 1-node B300-class sm103 (ali) | started: 2026-08-08 13:40:38 UTC
- per cell: CP4, seeds 16,17,18, 60 iters/seed (= 180 forwards/rank/cell)
- repro: standalone_te_cp_repro_v2.py (thd, padding_causal, MQA 8q/1kv/d128, bf16, 698-token seq padded to 704, p2p CP); rc=1 by design when nondeterministic
- cuDNN axis PROVEN per cell: loaded libcudnn resolved via /proc/self/maps + cudnnGetVersion() (see version_matrix.json probe.loaded_cudnn); cells where the loaded lib != requested wheel are marked invalid, never counted

| TE \ cuDNN | 9.19.0.56 | 9.21.1.3 | 9.24.0.43 | 9.25.0.15 |
|---|---|---|---|---|
| 2.16.0 | **FIRES** (3/3 seeds) | **FIRES** (3/3 seeds) | **FIRES** (3/3 seeds) | **FIRES** (3/3 seeds) |
| 2.16.0 (direct_seqlens) | — | — | **FIRES** (3/3 seeds) | **FIRES** (3/3 seeds) |
| 2.17.1 | **FIRES** (3/3 seeds) | **FIRES** (3/3 seeds) | **FIRES** (3/3 seeds) | **FIRES** (3/3 seeds) |
| 2.17.1 (direct_seqlens) | — | — | **FIRES** (3/3 seeds) | **FIRES** (3/3 seeds) |
| main | quiet (0/3 seeds) | quiet (0/3 seeds) | quiet (0/3 seeds) | quiet (0/3 seeds) |
| main (direct_seqlens) | — | — | quiet (0/3 seeds) | quiet (0/3 seeds) |

## Per-cell detail (firing cells)

### te=2.16.0|cudnn=9.19.0.56|arm=default
- seed 16: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r2:8distinct rows[174], r1:7distinct rows[174], r0:6distinct rows[174]
- seed 17: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r0:5distinct rows[174], r2:5distinct rows[174], r1:4distinct rows[174]
- seed 18: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r2:9distinct rows[174], r0:4distinct rows[174], r1:4distinct rows[174]

### te=2.16.0|cudnn=9.21.1.3|arm=default
- seed 16: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r0:6distinct rows[174], r1:7distinct rows[174], r2:8distinct rows[174]
- seed 17: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r2:4distinct rows[174], r1:4distinct rows[174], r0:5distinct rows[174]
- seed 18: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r2:4distinct rows[174], r1:4distinct rows[174], r0:3distinct rows[174]

### te=2.16.0|cudnn=9.24.0.43|arm=default
- seed 16: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r2:6distinct rows[174], r1:7distinct rows[174], r0:6distinct rows[174]
- seed 17: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r0:5distinct rows[174], r1:4distinct rows[174], r2:4distinct rows[174]
- seed 18: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r0:4distinct rows[174], r1:4distinct rows[174], r2:3distinct rows[174]

### te=2.16.0|cudnn=9.24.0.43|arm=direct_seqlens
- seed 16: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r0:6distinct rows[174], r2:5distinct rows[174], r1:7distinct rows[174]
- seed 17: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r0:5distinct rows[174], r2:4distinct rows[174], r1:4distinct rows[174]
- seed 18: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r0:4distinct rows[174], r2:3distinct rows[174], r1:4distinct rows[174]

### te=2.16.0|cudnn=9.25.0.15|arm=default
- seed 16: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r2:5distinct rows[174], r1:7distinct rows[174], r0:6distinct rows[174]
- seed 17: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r0:4distinct rows[174], r2:4distinct rows[174], r1:3distinct rows[174]
- seed 18: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r1:3distinct rows[174], r2:3distinct rows[174], r0:3distinct rows[174]

### te=2.16.0|cudnn=9.25.0.15|arm=direct_seqlens
- seed 16: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r1:7distinct rows[174], r2:7distinct rows[174], r0:6distinct rows[174]
- seed 17: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r1:4distinct rows[174], r0:5distinct rows[174], r2:4distinct rows[174]
- seed 18: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r1:4distinct rows[174], r0:3distinct rows[174], r2:4distinct rows[174]

### te=2.17.1|cudnn=9.19.0.56|arm=default
- seed 16: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r1:6distinct rows[174], r2:7distinct rows[174], r0:6distinct rows[174]
- seed 17: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r2:6distinct rows[174], r0:5distinct rows[174], r1:3distinct rows[174]
- seed 18: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r0:4distinct rows[174], r1:3distinct rows[174], r2:6distinct rows[174]

### te=2.17.1|cudnn=9.21.1.3|arm=default
- seed 16: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r0:6distinct rows[174], r1:7distinct rows[174], r2:5distinct rows[174]
- seed 17: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r0:4distinct rows[174], r1:5distinct rows[174], r2:4distinct rows[174]
- seed 18: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r0:3distinct rows[174], r2:3distinct rows[174], r1:4distinct rows[174]

### te=2.17.1|cudnn=9.24.0.43|arm=default
- seed 16: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r2:8distinct rows[174], r1:7distinct rows[174], r0:6distinct rows[174]
- seed 17: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r2:4distinct rows[174], r1:4distinct rows[174], r0:5distinct rows[174]
- seed 18: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r2:4distinct rows[174], r1:4distinct rows[174], r0:4distinct rows[174]

### te=2.17.1|cudnn=9.24.0.43|arm=direct_seqlens
- seed 16: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r1:7distinct rows[174], r0:6distinct rows[174], r2:7distinct rows[174]
- seed 17: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r2:6distinct rows[174], r1:5distinct rows[174], r0:4distinct rows[174]
- seed 18: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r2:6distinct rows[174], r1:4distinct rows[174], r0:3distinct rows[174]

### te=2.17.1|cudnn=9.25.0.15|arm=default
- seed 16: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r2:5distinct rows[174], r1:7distinct rows[174], r0:6distinct rows[174]
- seed 17: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r0:5distinct rows[174], r1:3distinct rows[174], r2:4distinct rows[174]
- seed 18: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r0:4distinct rows[174], r1:3distinct rows[174], r2:3distinct rows[174]

### te=2.17.1|cudnn=9.25.0.15|arm=direct_seqlens
- seed 16: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r0:6distinct rows[174], r2:6distinct rows[174], r1:7distinct rows[174]
- seed 17: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r1:5distinct rows[174], r0:5distinct rows[174], r2:6distinct rows[174]
- seed 18: NONDETERMINISTIC (3/4) — r3:1distinct rows[], r2:6distinct rows[174], r0:4distinct rows[174], r1:4distinct rows[174]

## Venv builds

- TE 2.16.0: OK (61s)
- TE 2.17.1: OK (290s)
- TE main: OK (1391s)
## Notes

- TE 'main' = 2.19.0.dev0+8260f49 (built from source, NVTE_CUDA_ARCHS=100, NVTE_WITH_NCCL_EP=0, cuda-toolkit 13.0).
- NVTE_FUSED_ATTN_DIRECT_SEQLENS is ABSENT (python + compiled binary) in every tested TE build (2.16.0, 2.17.1, main) — the direct_seqlens arm cells are inert duplicates of default, kept for the record.
- Wobble-row histogram over all firing cells/seeds/ranks: {174: 108} — 100% row 174 (T_LOCAL-2, tail of the rank's second load-balanced CP chunk), the session-3 signature, invariant across cuDNN versions.
- nvidia-cublas pinned to 13.6.0.2 in all venvs (torch cu130's default 13.1.0.3 lacks cublasLtGroupedMatrixLayout* needed by TE cu13 wheels).
