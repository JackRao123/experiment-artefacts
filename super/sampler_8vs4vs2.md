# Nemotron-3-Super BF16 sampler — 8×B200 vs 4×B200 vs 2×B200, 256k

Quick check (TRN-1488 follow-up to `ultra/profiling_4vs8_gpus.md`), run
2026-07-10: TP=4/TP=2 on box `q9o1pjq`, TP=8 on box `wdgmxmw` (each 1×8 B200,
178.35 GiB visible/GPU, Birch/Weka).

Golden Super recipe mirrored exactly (`sampler_configs.py` NEMOTRON_3_SUPER
B200/S256K, minus `gpu_count`): BF16 checkpoint
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`, bf16 KV (no fp8 scheme, no
CUTLASS/autotune/custom-AR overrides — those are Ultra-NVFP4-only),
`enable_lora` r64 / `max_loras=4` with the pirate-v1 Super adapter on every
request, `max_num_seqs=1000`, 16-size capture set, fastsafetensors,
`gpu_memory_utilization=0.90`, `max_model_len=262144`, vLLM 0.22.0. One
saturating synthetic 256k bucket per TP (prompt 261 104 + 1 024 gen,
`ignore_eos`, N ≈ 1.05× pool-implied concurrency). Harness:
`ultra/sampler_4vs8/run_super.sh`; raw JSONs in `ultra/sampler_4vs8/out/super_*`.

## Results

| | TP=8 (8×B200) | TP=4 (4×B200) | TP=2 (2×B200) |
| --- | ---: | ---: | ---: |
| boots @256k | yes | yes | **no — OOM at KV alloc** |
| "Model loading took" /GPU | 56.49 GiB | 91.23 GiB | 160.66 GiB |
| available KV/GPU | 97.88 GiB | 62.74 GiB | ~0 (1.14 GiB free at OOM) |
| KV pool | 4,268,271 tok | 2,728,680 tok | — |
| max concurrency @256k | **16.28×** | **10.41×** | — |
| total tok/s @256k saturating | 35,870 (4,484/GPU) | **23,338 (5,834/GPU)** | — |
| boot wall, warm / cold (s) | 329.9 / 674.6 | 246.8 / 606.1 | — |
| preemptions / OOM in bucket | 0 / none | 0 / none | — |
| per-GPU peak (nvidia-smi) | 163.8 ×8 | 163.8 ×4, idle 0 ×4 | — |

- Buckets: TP=8 ran 19 requests × 256k (4.96M prompt tokens) in 138.8 s;
  TP=4 ran 12 (3.13M) in 134.8 s. Weight load itself: 37.8 s vs 30.1 s warm.
- **TP=4 is +30% throughput per GPU vs TP=8** (5,834 vs 4,484 tok/s/GPU) and
  +28% KV-pool per GPU (682k vs 534k tok/GPU). A single TP=4 replica is
  0.65× a TP=8 replica; two TP=4 replicas on one node ≈ 46.7k tok/s and
  20.8× combined 256k concurrency vs 35.9k / 16.3× for one TP=8
  (colocation itself not measured). TP=8 keeps the higher *single-replica*
  ceiling.
- **TP=2 is weight-bound, not context-bound**: vLLM's model-load footprint is
  160.66 GiB/GPU — already over the 0.90×178.35 = 160.5 GiB budget before any
  KV. The engine dies in `_allocate_kv_cache_tensors` (2.02 GiB request,
  1.14 GiB free). A smaller `max_model_len` cannot fix this; no fallback run
  was worth the box time.
- **The load footprint fits `total ≈ 278 GiB sharded + 21.8 GiB × ranks`**
  (all three points: TP=2 321, TP=4 365, TP=8 452 GiB — the TP=8 value was
  *predicted* from the other two before it was measured, 452.1 vs 451.9).
  The ~22 GiB/rank replicated tail (buffers/workspaces that don't shard) is
  what kills TP=2 and taxes every extra rank; an FP8/NVFP4 Super checkpoint
  would change this arithmetic entirely.
- Same per-rank KV page-padding tax as Ultra: per-token cost ≈ 98.6 KB at
  TP=4 vs 197 KB at TP=8 (all-GPU), so doubling GPUs only buys 1.56× pool.

## Verdict

**TP=4 is the sweet spot for Super BF16 at 256k**: 10.4× full-256k
concurrency, ~23.3k tok/s saturated (+30% per GPU vs the current TP=8
golden), zero preemptions, warm boot ~4 min, half the footprint. **TP=2
cannot boot** the 256k golden config (or any context) — the per-GPU load
footprint exceeds the memory budget on its own; a Super footprint below 4
GPUs needs a quantized checkpoint, not a TP knob. Keep TP=8 only where one
replica must hold >10 concurrent full-256k rollouts or >23k tok/s.

A `gpu_count` 8→4 flip for Super is supported by these numbers (same shape
as the Ultra TRN-1488 result, +30% instead of +60%); not PR'd in this pass.
