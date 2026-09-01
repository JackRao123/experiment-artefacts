# GLM-5.3 full-model B300 memory profiles

## Configuration

- Devbox: `tj-qjke943`, one 8xB300 node (`NVIDIA L20D`, 267.69 GiB usable CUDA memory per GPU).
- Code: trainers `ceacc74fc` (`feat(glm): compile native FP8 expert materialization`).
- Checkpoint: complete base `zai-org/GLM-5.3`, snapshot `187fb9fff6319062325ff825627ef6db084d9bc6` (141 FP8 safetensors shards).
- Parallelism: TP1, PP1, CP8, EP8, ETP1, DP1.
- Training: LoRA rank/alpha 32, HybridEP, native FP8 routed-expert storage, BF16 expert compute, full uniform one-layer recompute.
- Workload: one synthetic document per step; 131072 and 262144 tokens.
- Protocol: one shape warmup, three unprofiled control steps, then one memory-profiled step. Runtime profiling was disabled.
- Allocator: `expandable_segments:True,garbage_collection_threshold:0.95`.

## Throughput

| Sequence | Control FB times | Control tok/s/GPU | Driver mean | Stabilized last-two | Memory-profile FB | Profile tok/s/GPU | Profile overhead | MFU | HFU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 131K | 14.529 / 13.754 / 13.543 s | 1128 / 1191 / 1210 | 1175 | 1200 | 14.843 s | 1104 | 6.5% | 10.7% | 15.5% |
| 262K | 28.364 / 27.727 / 27.508 s | 1155 / 1182 / 1191 | 1176 | 1187 | 29.230 s | 1121 | 4.9% | 12.3% | 17.7% |

All ten steps completed with finite losses and gradient norms. There were no OOM events. Against the GLM-5.2 run on the same devbox, stabilized GLM-5.3 throughput was +0.5% at 131K and -0.7% at 262K, which is within run-to-run noise.

## Memory

Allocator-event reconstruction was performed independently for every rank. The hottest rank was rank 0, but rank spread was only about 0.05 GiB at peak.

| Sequence | Active baseline/rank | Peak allocated/rank | Transient growth | Final reserved/rank | Hottest allocated headroom | Hottest reserved headroom |
|---|---:|---:|---:|---:|---:|---:|
| 131K | 123.83-123.84 GiB | 171.00-171.06 GiB | 47.17-47.21 GiB | 174.10-174.57 GiB | 96.63 GiB | 93.12 GiB |
| 262K | 124.83-124.84 GiB | 213.26-213.32 GiB | 88.43-88.48 GiB | 218.01-218.75 GiB | 54.37 GiB | 48.94 GiB |

GLM-5.3's active-memory baseline, peak, transient growth, and dominant allocation sizes match GLM-5.2 exactly. This is expected from their shared architecture and confirms that the native-FP8 expert path introduces no GLM-5.3-specific memory regression.

### Hottest-rank allocations active at peak

| Allocation site | 131K | 262K | Scaling |
|---|---:|---:|---:|
| `_bias_dropout_add_func` | 14.62 GiB | 29.25 GiB | 2.00x |
| `_project_logits` line 183 | 14.18 GiB | 28.36 GiB | 2.00x |
| `_project_logits` line 197 | 11.82 GiB | 21.27 GiB | 1.80x |
| `sort_topk_by_index` | 2.62 GiB | 5.25 GiB | 2.00x |
| `_linear_forward` | 2.36 GiB | 2.36 GiB | constant |
| RoPE `get_emb` | 1.00 GiB | 2.00 GiB | 2.00x |
| `forward@utils.py:1312` | 1.18 GiB | 1.18 GiB | constant |

## Live trainer

The trainer was intentionally left running after profiling.

- Endpoint: `http://127.0.0.1:8001` on `ssh tj-qjke943`.
- Current step: 10.
- Maximum configured sequence length: 262144.
- Model path: `/root/.cache/team_artifacts/huggingface/hub/models--zai-org--GLM-5.3/snapshots/187fb9fff6319062325ff825627ef6db084d9bc6`.
- Lifecycle stop command, when eventually wanted: `bash /root/.cache/user_artifacts/devboxes/qjke943/.devbox_up/stop_trainer.sh`.

## Artifacts

- `131k_result.json`, `262k_result.json`: driver timings and metrics.
- `131k-memory/memory.rank*.pickle`, `262k-memory/memory.rank*.pickle`: all-rank allocator snapshots.
- `memory_summary.json`: machine-readable per-rank memory analysis.
- `analyze_memory.py`: allocator-event reconstruction used for the summary.
- `trainer-config.json`, `trainer-server-config.json`, `trainer_srun.log`: exact run configuration and server evidence.
