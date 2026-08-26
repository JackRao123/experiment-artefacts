# Progress checkpoint: main 95fe75e9, 131k d4/d8

This is the current GLM-5.2 B300 performance checkpoint as of 2026-08-25.
It is intended to be the easy-to-find baseline for subsequent optimization
work.

## Headline

| shape | tokens/step | control FB time | tok/s/GPU | total tok/s | mfu3x | HFU |
|---|---:|---:|---:|---:|---:|---:|
| d4 | 524,288 | 43.36 s | **756** | 12,092 | **6.88%** | 9.99% |
| d8 | 1,048,576 | 77.38 s | **847** | 13,551 | **7.71%** | 11.19% |

Moving from d4 to d8 improves per-GPU throughput by **12.1%**. It processes
2x the tokens in 1.785x the forward/backward time. The useful-FLOPs MFU gain is
0.83 percentage points.

## Control windows

| shape | tok/s/GPU controls | range | approximate CV |
|---|---|---:|---:|
| d4 | 747.9, 748.4, 771.5 | 747.9-771.5 | 1.8% |
| d8 | 860.1, 862.5, 819.5 | 819.5-862.5 | 2.9% |

The headline uses the driver's convention: tokens divided by mean untraced
forward/backward elapsed time. Optimizer time is excluded. d8 has a slower
third control, so its driver headline is 847 tok/s/GPU even though its first
two controls are 860-863.

## Setup

- Trainers commit: `95fe75e9f893c635616ba150586979e16946740e` (exact detached checkout of main)
- Devbox: `q4grmdq`, 2 nodes x 8 NVIDIA B300 GPUs
- Model: `zai-org/GLM-5.2-FP8`
- Topology: TP1 / PP2 / EP8 / CP8 / ETP1 / DP1
- Sequence length: 131,072 tokens per datum
- LoRA: rank 32, alpha 32
- Attention backend: flash
- Weight sync: disabled
- Driver: `tools/profile_driver_new.py`, SHA-256 `5702c0dd0ae5ec78d74e78dd582ce79f31b95112789fa3801abb515bd49adad7`
- MFU calculator: `tools/mfu.py`, SHA-256 `5513fd08381629e4331b5ec2c79619b23d1e1be6865f414fbc4e8838d701c21b`
- Protocol: warmup, one Kineto-traced window, then three untraced control windows per shape
- Run order: d4 first, then d8 on the same hot server

MFU is the driver's LoRA-corrected useful-FLOPs estimate using the LPS-1062
B300 dense-BF16 peak convention of 2.5 PFLOP/s/GPU. HFU is the analytic
full-recompute estimate. These values should only be compared with rows using
the same convention.

## Memory and canaries

- Driver rank-0 final live memory: 142.5 GiB at d4, 150.3 GiB at d8.
- All-rank session maximum from the external poller: 170.8 GiB on pipeline stage 1.
- Peak node-0 GPU: 162.6 GiB; peak node-1 GPU: 170.8 GiB.
- Loss stayed finite and decreased from 12.319 to 12.269 across the measured steps.
- Gradient norm stayed finite in the 0.35-0.46 range.
- No OOM or trainer failure occurred. The allocation was stopped after capture and all GPUs returned to zero usage.

## Artifacts

- `results.json`: exact window measurements, aggregates, trace identifiers, and all-rank memory maxima.
- `trainer_config.json`: trainer configuration used for both shapes.
- Box-side original driver JSONs:
  `/root/.cache/user_artifacts/lps1062_bench/main-95fe75e9-131k-d4.json` and
  `/root/.cache/user_artifacts/lps1062_bench/main-95fe75e9-131k-d8.json`.
- The captured rank-0 Kineto traces were 740,071,541 bytes at d4 and
  1,485,858,688 bytes at d8. Their exact filenames are recorded in
  `results.json`.

The large trace and memory-profile files remain box-side and are not committed.
