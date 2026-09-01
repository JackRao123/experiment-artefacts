# GLM-5.2 full-model B300 memory profiles

## Configuration

- Devbox: `tj-qjke943`, one 8xB300 node (`NVIDIA L20D`, 267.69 GiB usable CUDA memory per GPU).
- Code: trainers `ceacc74fc` (`feat(glm): compile native FP8 expert materialization`).
- Checkpoint: full 78-layer `zai-org/GLM-5.2-FP8`, snapshot `ba978f7d347eaf65d22f1a86833408afdb953541`.
- Parallelism: TP1, PP1, CP8, EP8, ETP1, DP1.
- Training: LoRA rank/alpha 32, HybridEP, native FP8 routed-expert storage, BF16 expert compute, full uniform one-layer recompute.
- Workload: one synthetic document per step; 131072 and 262144 tokens.
- Protocol: one shape warmup, three unprofiled control steps, then one memory-profiled step. Runtime profiling was disabled.
- Allocator: `expandable_segments:True,garbage_collection_threshold:0.95`.

## Throughput

| Sequence | Control FB times | Control tok/s/GPU | Driver mean | Stabilized last-two | Memory-profile FB | Profile tok/s/GPU | Profile overhead | MFU | HFU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 131K | 15.396 / 13.924 / 13.520 s | 1064 / 1177 / 1212 | 1147 | 1194 | 14.577 s | 1124 | 2.1% | 10.4% | 15.2% |
| 262K | 28.062 / 27.433 / 27.422 s | 1168 / 1194 / 1195 | 1186 | 1195 | 28.587 s | 1146 | 3.4% | 12.4% | 17.8% |

The first 131K control still had residual shape warmup. The last two controls are stable and show essentially identical token throughput at 131K and 262K: 1194.0 versus 1194.7 tok/s/GPU.

All ten steps completed with finite losses and gradient norms. There were no OOM events.

## Memory

Allocator-event reconstruction was performed independently for every rank. The hottest rank was rank 0, but rank spread was only about 0.05 GiB at peak.

| Sequence | Active baseline/rank | Peak allocated/rank | Transient growth | Final reserved/rank | Hottest allocated headroom | Hottest reserved headroom |
|---|---:|---:|---:|---:|---:|---:|
| 131K | 123.83-123.84 GiB | 171.00-171.06 GiB | 47.17-47.21 GiB | 174.16-174.92 GiB | 96.63 GiB | 92.77 GiB |
| 262K | 124.83-124.84 GiB | 213.26-213.32 GiB | 88.43-88.48 GiB | 217.95-218.73 GiB | 54.37 GiB | 48.96 GiB |

Peak allocation used 63.9% of device memory at 131K and 79.7% at 262K. Peak reserved usage was 65.3% and 81.7%, respectively. The 262K configuration therefore fits with substantial margin on this 8xB300 shape.

Active memory returned to its pre-step baseline after each profiled step. The allocator retained freed blocks in its cache: rank 0 ended with 51.08 GiB inactive at 131K and 93.58 GiB inactive at 262K. This is caching, not a live-tensor leak; no monotonic active-memory growth occurred inside either recording window.

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

The peak is dominated by sequence-scaled residual/logit materialization, not routed-expert storage. Doubling sequence length adds about 42.26 GiB to peak allocation; the largest contributors approximately double as expected.

## GLM-5.3 checkpoint

The pre-existing GLM-5.3 cache entry was metadata-only. A delegated Xet high-performance download completed in 2m18s with 64 concurrent range gets.

- Full FP8 snapshot: `/root/.cache/team_artifacts/huggingface/hub/models--zai-org--GLM-5.3/snapshots/187fb9fff6319062325ff825627ef6db084d9bc6`
- Resolved size: 755,663,688,736 bytes (705 GiB).
- Contents: 154 files, 141 safetensors shards, 118,629 indexed tensors.
- Verification: zero missing indexed shards, zero extra shards, zero broken symlinks, zero incomplete cache files.
- Quantization: FP8 E4M3, dynamic activation, 128x128 weight blocks.

## Artifacts

- `131k_result.json`, `262k_result.json`: driver timings and metrics.
- `131k-memory/memory.rank*.pickle`, `262k-memory/memory.rank*.pickle`: all-rank allocator snapshots.
- `memory_summary.json`: machine-readable per-rank memory analysis.
- `analyze_memory.py`: allocator-event reconstruction used for the summary.
- `trainer-config.json`, `trainer-server-config.json`, `trainer_srun.log`: exact run configuration and server evidence.
