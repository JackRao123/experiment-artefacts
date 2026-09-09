# GLM-5.3 PR 1327: 262K CP8/EP8 on 8xB300

Run date: 2026-09-04

## Code and configuration

- Trainers PR: [#1327](https://github.com/basetenlabs/trainers/pull/1327), `ec205be06a0cfe90916244f91ca135b882308e47`
- Megatron-Bridge PR: [#81](https://github.com/basetenlabs/Megatron-Bridge/pull/81), `00c8c2be165586d46da0b1f76fd900d70c2860d1`
- Megatron-LM PR: [#75](https://github.com/basetenlabs/Megatron-LM/pull/75), `e732d64d414cfe94dc19f883b8067a79942cd2d3`
- Model: `zai-org/GLM-5.3`, snapshot `187fb9fff6319062325ff825627ef6db084d9bc6`
- Hardware: one node, 8xB300 (`NVIDIA L20D` device name), 275040 MiB reported per GPU
- Topology: TP1 / PP1 / CP8 / EP8 / ETP1 / DP1
- Sequence: one 262,144-token synthetic datum, seed `0xB300`
- Runtime: FlashAttention, HybridEP, native-FP8 experts, LoRA rank 32, full uniform one-layer recompute
- Exact input: `trainer-config.json`

## Fixed-input rerun

After updating `profile_driver.py` to reuse one datum, one warmup plus ten control steps produced:

- Mean: **24.917 s, 1,315 tok/s/GPU, 10,521 tok/s aggregate**
- Median: **24.791 s**
- Standard deviation: **0.278 s**
- Range: **24.755-25.654 s**
- Peak reserved: **208.398 GiB**
- Peak allocated: **201.514 GiB**

Controls 6-8 ran with flat reserved memory and averaged 24.803 s, or 1,321 tok/s/GPU. Control 9 increased reserved memory again and took 25.654 s.

Raw results: `fixed-input-control10.json`.

## Original unprofiled benchmark

The headline run used the stock trainer launcher, one warmup, then three control steps. Nsight and the PR 1298 NVTX overlay were not present.

| Window | Forward/backward | tok/s/GPU | Optimizer |
|---|---:|---:|---:|
| control 0 | 28.111 s | 1,166 | 0.304 s |
| control 1 | 26.907 s | 1,218 | 0.106 s |
| control 2 | 25.933 s | 1,264 | 0.046 s |
| mean | 26.984 s | 1,214 | 0.152 s |

Headline throughput is **1,214 tok/s/GPU**, or **9,715 tok/s aggregate**. The control spread is material: 1,166-1,264 tok/s/GPU. The single control immediately before capture, under an idle Nsight-launched process tree, was 26.551 s or 1,234 tok/s/GPU.

Peak memory across all ranks and all benchmark windows:

- Reserved: **222,130,339,840 bytes = 206.875 GiB** (77.3% of CUDA-reported capacity)
- Allocated: **214,141,129,216 bytes = 199.434 GiB** (74.5% of CUDA-reported capacity)
- CUDA-reported capacity: 287,428,771,840 bytes = 267.689 GiB

The server computes these peaks with a distributed `MAX`, so they are the maximum rank values rather than rank-0-only measurements.

## Nsight capture

Trace: `glm53-pr1327-b300-262k-cp8ep8-nvtx-all-ranks-gpu-metrics.nsys-rep`

- Size: 287,684,026 bytes (274 MiB)
- SHA-256: `0de131c42d7198155f3ead537f09196f32d7f4661911f4fa870eb9a6c81a2133`
- Nsight Systems: 2025.3.2
- Profiled forward/backward: 26.247 s, 1,248 tok/s/GPU
- Profiled optimizer: 0.112 s
- Captured peak: 213.164 GiB reserved, 205.155 GiB allocated
- Tracing: CUDA, NVTX, cuBLAS, cuDNN, OS runtime, CUDA events, CUDA memory usage, CPU sampling/context switches, GPU context switches
- GPU metrics: all eight devices at 10 kHz
- NVTX: PR 1298 instrumentation adapted to the current PR 1327 tree; this overlay was used only for the trace

The 5,549,268,992-byte SQLite export is beside the trace with SHA-256 `ac4ac8dd8b577808e6c632f97039e05277ab5c88419336e14a43d1353a5f8f05`.

## Artifacts

- `fixed-input-control10.json`: fixed-input warmup and ten-control rerun
- `control.json`: stock-launcher benchmark windows and memory peaks
- `glm53-pr1327-b300-262k-cp8ep8-pre-nsys.json`: warmup and control under the idle Nsight launcher
- `glm53-pr1327-b300-262k-cp8ep8-nvtx-all-ranks-gpu-metrics.json`: captured-step timing and memory
- `glm53-pr1327-b300-262k-cp8ep8-nvtx-all-ranks-gpu-metrics.nsys-rep`: self-contained Nsight report
- `glm53-pr1327-b300-262k-cp8ep8-nvtx-all-ranks-gpu-metrics.sqlite`: SQLite export of the replacement trace
- `nvtx_ranges_pr1298_current.patch`: the current-tree PR 1298 trace overlay
- `trainer_srun.log`: trace-run trainer log
