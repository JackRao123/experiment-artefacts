# Timeline: GLM-5.2 IMA Bug, DPxCP, Fixes, and Affected Runs

## Overview

A CUDA illegal memory access (IMA) bug in nvidia-cudnn-frontend 1.26.0's CuTe-DSL DSA
indexer top-k kernel (`indexer_top_k_decode_varlen`) affects GLM-5.2 training runs on
B200 and B300 GPUs. The bug is **deterministically data-dependent**: same seed/data
fails at the same iteration. The crash requires a row where fewer than 2048 values are
above ~-65504 (the top-2048 threshold lands in the fp16 `-inf` coarse bin), combined
with >148 rows (large_occupancy compile). The flood condition is data-marginal and
flips on run-to-run numeric wobble.

GLM-5.2 only runs on B200 or B300.

---

## Fixes

### B200 fix — PR #814, commit `aa5d05e`

- **Type**: GitHub PR (basetenlabs/trainers)
- **Author**: Jack Rao
- **Created**: 2026-07-28 04:50 UTC
- **Merged**: 2026-07-28 19:49 UTC
- **What**: Vendored patched wheel (`1.26.0+dsatopk1`) that skips OOB lanes in the
  three vectorized histogram/collection loops of the DSA indexer top-k kernel.
  Shipped as a pristine PyPI 1.26.0 wheel with the patch applied, repacked and
  sourced via `[tool.uv.sources]` (marker-gated to linux/x86_64/cp312).
- **Root cause**: nvidia-cudnn-frontend 1.26.0's CuTe-DSL radix top-k kernel fills
  out-of-bounds lanes of a row's final vector tile with `-inf` and counts them as
  real elements in its radix histogram and candidate-collection passes. When a row's
  top-2048 threshold lands in the fp16 `-inf` coarse bin (fewer than 2048 values above
  ~-65504), the phantom lanes flood the per-row candidate buffers (512-entry smem +
  `num_cols` gmem), overflowing them → out-of-bounds shared/global writes →
  `cudaErrorIllegalAddress` (Xid 43).
- **Validation**: A/B/A causality proof on both B200 and B300 — pristine 1.26.0 →
  `cudaErrorIllegalAddress`; patched → PASS with exact `torch.topk` parity.
  12/12 randomized parity trials. 100/100 iterations on the GLM-5.2 full-indexer repro.

### B300 fix — PR #829, commit `1fd152a`

- **Type**: GitHub PR (basetenlabs/trainers)
- **Author**: Jack Rao
- **Created**: 2026-07-29 02:40 UTC
- **Merged**: 2026-07-29 17:18 UTC
- **What**: Pins GLM-5.2 B300 trainer config to patched DSA image
  `trainer-cuda13-sm103-0e0b65a` (previous: `trainer-cuda13-sm103-5a4ae4d`).
  B300 uses CUDA 13.0 / sm103 — a different image build from B200. The #814 patch
  is carried in the `0e0b65a` build (config-only repin on top of `98cd395`).
- **Validation**: Dev-box A/B on 2×8 B300 on ali (one node is an incident node):
  pristine 1.26.0 → 5/5 IMA; #814 patch → 11/11 PASS. Production validation: job
  `wpdvvpw` (trainer deployment, 2×8 B300), TRAINING_JOB_COMPLETED / VALIDATION_PASS.
  `nvidia-cudnn-frontend==1.26.0+dsatopk1` verified inside ranks 0 and 1.

### DPxCP implementation — PR #801, commit `4647c37`

- **Type**: GitHub PR (basetenlabs/trainers)
- **Author**: XiaohanZhangCMU
- **Committed**: 2026-07-27 18:48 UTC
- **What**: Enables `data_parallel_size>1` for THD context parallelism (GLM-5.2
  cp32, Nemotron cp4). Before this, `truss loops push --replicas N>1` was a hard
  crash-loop for GLM-5.2 because the trainer rejected `data_parallel_size>1` for
  THD context parallelism.
- **Key changes**: Per-partition grad finalize once (not per-partition), then
  refactored to CP-only reduce + single DP fold (no phantom partitions needed).
  Removed the `data_parallel_size>1` raise (kept the PP>1 raise).

---

## Chronological Timeline

### Early July (Jul 5–9): Crash-heavy runs — Parallel AI | Oppty DL

- **Org**: Parallel AI | Oppty DL (`org-2aa32a7b49234db2990d1908f4a43c24`)
- **Note**: These are Qwen3.5-VL-MoE (Qwen35VLMoEBridge) on H200, NOT GLM-5.2.
  Included for context as they exhibit the same IMA bug class.

| Trainer Deployment ID | GPU | Model | IMA Crashes | Step Logs | Period |
|---|---|---|---|---|---|
| `5qe7vrw` | H200 | Qwen3.5-VL-MoE | 43 | 720 | Jul 5–7 |
| `7qr9p5q` | H200 | Qwen3.5-VL-MoE | 42 | 1368 | Jul 6–9 |
| `5woo6nw` | H200 | Qwen3.5-VL-MoE | 16 | 600 | Jul 5–6 |
| `4w57dr3` | H200 | Qwen3.5-VL-MoE | 16 | 288 | Jul 6–7 |

### Jul 23–25: Mudith's runs — Parsed | Oppty LM | EMEA

- **Trainer deployment**: `e3m97kq`
- **Org**: Parsed | Oppty LM | EMEA (`org-99340d71961343c28c5c567d705ab0c0`)
- **Cluster**: `ali-apse7-prod-1`
- **GPU**: B200 (confirmed via DCGM: modelName = "NVIDIA B200", 56 unique UUIDs)
- **Model**: GLM-5.2 (GLM5Bridge, MimoModelConfig)
- **Image**: Pre-fix (before PR #814)
- **Three-run lineage**:

| Loops Run ID | Type | GPU | Image | Steps | Crashes | Period (UTC) |
|---|---|---|---|---|---|---|
| `6wg87jq` | Loops run (on `e3m97kq`) | B200 | pre-fix | 18 | **1 IMA crash** (the #814 bug) | Jul 23 ~21:03 – 21:57 |
| `4w5jk73` | Loops run (on `e3m97kq`) | B200 | pre-fix | 58 | 0 | Jul 23 ~22:05 – ~23:41 |
| `4q9ex6w` | Loops run (on `e3m97kq`) | B200 | pre-fix | 1529 | **0** | Jul 24 ~00:16 – Jul 25 ~13:40 |

**Run 4q9ex6w loss curve**: 1529 steps, all contiguous (no gaps). Loss decreases
smoothly from ~0.002 to ~0.0012. Only 3 spikes (>3x running median), all in the
first 109 steps:

| Step | Loss | Ratio vs local median | Grad Norm | Timestamp (UTC) |
|---|---|---|---|---|
| 4 | 0.00533 | 3.35x | 0.0015 (normal) | Jul 24 00:23 |
| 96 | 0.00619 | 3.35x | 0.0242 (16x typical) | Jul 24 02:37 |
| 109 | 0.00743 | 4.21x | 0.0287 (19x typical) | Jul 24 02:57 |

After step 109: zero spikes for the remaining 1420 steps. Zero crashes, zero
restarts, zero IMA errors across all 121,099 log lines.

### Jul 24: LPS-995 incident — Trainloop

- **Trainer deployment**: `e3m916q`
- **Type**: Trainer deployment
- **Org**: Trainloop (`org-f1671ec43d9c40c8aa88c3c727d971e1`)
- **Cluster**: `hyd-euis1-prod-1`
- **GPU**: B200
- **Model**: GLM-5.2, CP32
- **Image**: `baseten/trainers-server:main-1d2de6f` (commit 1d2de6f, predates b300 work)
- **Crash**: Jul 24 00:37:32 UTC — IMA on rank 4, mid-`forward_backward` on a 55,111-token batch
- **Linear issue**: LPS-995 (created Jul 28, completed Jul 29)
- **Fix PR**: #814

### Jul 24: baseten org GLM runs — B200

| Trainer Deployment ID | Type | Org | Cluster | GPU | Image | IMA Crashes | Period |
|---|---|---|---|---|---|---|---|
| `rwnrn0q` | Trainer deployment | baseten | hyd-euis1-prod-1 | B200 | pre-fix | 3 | Jul 24 |
| `4q99x6q` | Trainer deployment | baseten | hyd-euis1-prod-1 | B200 | pre-fix | 3 | Jul 24 |
| `yqvp5gq` | Trainer deployment | Parsed | hyd-euis1-prod-1 | B200 | pre-fix | 2 | Jul 24 |

### Jul 27: DPxCP implementation merged

- PR #801, commit `4647c37`, committed Jul 27 18:48 UTC

### Jul 27–29: B300 incidents — Parsed org, ali-apse7-prod-1

These are the B300 incidents that led to the B300 fix (PR #829).

| Trainer Deployment ID | Type | Org | Cluster | GPU | Image | IMA Crashes | First Crash (UTC) | Last Crash (UTC) |
|---|---|---|---|---|---|---|---|---|
| `dq48213` | Trainer deployment | Parsed | ali-apse7-prod-1 | B300 | `5a4ae4d` (old) | 2 | Jul 27 22:02 | Jul 27 22:02 |
| `4w79o03` | Trainer deployment | Parsed | ali-apse7-prod-1 | B300 | `5a4ae4d` (old) | 2 | Jul 28 00:35 | Jul 28 00:35 |
| `rwn24dw` | Trainer deployment | Parsed | ali-apse7-prod-1 | B300 | `5a4ae4d` (old) | 5 | Jul 28 21:38 | Jul 28 22:00 |
| `4w5y6r3` | Trainer deployment | Parsed | ali-apse7-prod-1 | B300 | `5a4ae4d` (old) | 39 | Jul 29 08:43 | Jul 29 22:23 |
| `7qkplew` | Trainer deployment | Parsed | ali-apse7-prod-1 | B300 | `5a4ae4d` (old) | 13 | Jul 28 04:48 | Jul 28 05:08 |

Fleet fingerprint: ≥4 identical IMA crashes in ~26h across ≥3 deployments on
different nodes/GPUs. Xid differential: Xid 13 "Out Of Range Address" + "Multiple
Warp Errors" + Xid 43 with byte-identical ESR words — the SM/shared-memory
out-of-range signature of the top-k candidate flood.

### Jul 28: B200 fix merged

- PR #814, commit `aa5d05e`, merged Jul 28 19:49 UTC

### Jul 29: B300 fix merged

- PR #829, commit `1fd152a`, merged Jul 29 17:18 UTC

### Jul 30: Post-fix status

All deployments listed above have stopped logging. All were running the OLD
(pre-fix) image — none were running the fixed image. No deployment running the
fixed image has been observed crashing on GLM.

---

## Complete List of GLM-5.2 Deployments with IMA Crashes

| Trainer Deployment ID | Type | Org | Cluster | GPU | Image | IMA Crash Logs | First Crash (UTC) | Last Crash (UTC) | Pre/Post Fix |
|---|---|---|---|---|---|---|---|---|---|
| `e3m97kq` | Trainer deployment | Parsed | ali-apse7-prod-1 | B200 | pre-fix | 2 | Jul 23 21:57 | Jul 23 21:57 | PRE-fix |
| `e3m916q` | Trainer deployment | Trainloop | hyd-euis1-prod-1 | B200 | `main-1d2de6f` | 2 | Jul 25 | Jul 25 | PRE-fix |
| `yqvp5gq` | Trainer deployment | Parsed | hyd-euis1-prod-1 | B200 | pre-fix | 2 | Jul 24 | Jul 24 | PRE-fix |
| `rwnrn0q` | Trainer deployment | baseten | hyd-euis1-prod-1 | B200 | pre-fix | 3 | Jul 24 | Jul 24 | PRE-fix |
| `4q99x6q` | Trainer deployment | baseten | hyd-euis1-prod-1 | B200 | pre-fix | 3 | Jul 24 | Jul 24 | PRE-fix |
| `dq48213` | Trainer deployment | Parsed | ali-apse7-prod-1 | B300 | `5a4ae4d` (old) | 2 | Jul 27 22:02 | Jul 27 22:02 | PRE-fix |
| `4w79o03` | Trainer deployment | Parsed | ali-apse7-prod-1 | B300 | `5a4ae4d` (old) | 2 | Jul 28 00:35 | Jul 28 00:35 | PRE-fix |
| `7qkplew` | Trainer deployment | Parsed | ali-apse7-prod-1 | B300 | `5a4ae4d` (old) | 13 | Jul 28 04:48 | Jul 28 05:08 | PRE-fix |
| `rwn24dw` | Trainer deployment | Parsed | ali-apse7-prod-1 | B300 | `5a4ae4d` (old) | 5 | Jul 28 21:38 | Jul 28 22:00 | PRE-fix |
| `4w5y6r3` | Trainer deployment | Parsed | ali-apse7-prod-1 | B300 | `5a4ae4d` (old) | 39 | Jul 29 08:43 | Jul 29 22:23 | PRE-fix |
| `yqv2or3` | Trainer deployment | baseten | hyd-euis1-prod-1 | B200 | pre-fix | 3 | Jul 25 08:09 | Jul 25 08:09 | PRE-fix |

All deployments were running the OLD (pre-fix) image. No deployment running the
fixed image has been observed crashing on GLM.

---

## Why Mudith's Run 4q9ex6w Didn't Crash (B200, Pre-fix)

The crash is **deterministically data-dependent**: it requires a specific data
pattern — a row where fewer than 2048 values are above ~-65504 (the top-2048
threshold lands in the fp16 `-inf` coarse bin), combined with >148 rows
(large_occupancy compile). Not every batch triggers it.

Run 1 (`6wg87jq`) **did** crash at step 18 — same #814 bug. Runs 2 and 3 used
different data (fresh LoRA re-init / warm continuation) that didn't hit the
triggering pattern in 1529 steps. Per PR #829's A/B evidence: "the flood condition
is data-marginal and flips on run-to-run numeric wobble" — the crash is right at
the boundary of triggering, and minor numeric differences (different initialization,
different shuffling) can flip it from crash to clean.

---

## Key Commits and PRs

| Commit | PR | Title | Merged (UTC) |
|---|---|---|---|
| `4647c37` | #801 | feat(dp_worker): support data_parallel_size>1 for THD context parallelism | Jul 27 18:48 |
| `aa5d05e` | #814 | fix(server): carry cuDNN-frontend DSA indexer top-k OOB patch (B200) | Jul 28 19:49 |
| `1fd152a` | #829 | fix(models): scope patched DSA image to GLM-5.2 B300 | Jul 29 17:18 |

## Linear Issues

| Issue | Title | Created | Completed |
|---|---|---|---|
| LPS-995 | GLM trainer restart: CUDA illegal memory access on rank 4 (4x B200) | Jul 28 | Jul 29 |
