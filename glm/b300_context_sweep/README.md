# GLM-5.2 B300 context-length memory sweep — 1-node EP8+CP8 vs golden 2-node EP16+CP16

Profiled 2026-08-08/09 (UTC) on devboxes `wxll97w` (1 node, 8 B300) and
`3188jrw` (2 nodes, 16 B300), ali cluster, 275,040 MiB usable per GPU.
Question: **at what context length can we train GLM-5.2 on B300, and how does
peak GPU memory scale with context length?**

Stack:

- trainers `0e0b65a6` (`trainers_main` devbox clone)
- model `zai-org/GLM-5.2-FP8` (bridge dequantizes FP8→bf16 on load; 78 layers,
  hidden 6144, 256 routed experts top-8 + 1 shared, MLA + DSA indexer topk
  2048, vocab 154,880, max_position 1,048,576)

Both runs: attention-only LoRA r=32/α=32 (experts frozen — the supported
GLM-5.2 path), full recompute (uniform, 1 layer), alltoall dispatcher, no
`moe_capacity` (as golden), `attention_backend="flash"`,
`max_seq_len=1,048,576` (cap only), `weight_sync` disabled,
`BT_SKIP_FULL_WARMUP=1` (so the prod boot's max_seq_len-token warmup doesn't
pre-reserve the ceiling and poison per-L peaks).

Method: one synthetic random-token sequence of length L per `forward_backward`
(CE loss, labels auto-shifted) + one `optim_step`, per L; warmup step first so
lazy optimizer allocations are in the baseline. Peak = `nvidia-smi` reserved
high-water, max across all GPUs, sampled ~0.3–0.4s around each step. Sweep
ascends a 16K grid until the trainer OOM-terminates.

## TL;DR

| | Run 1: EP8+CP8, 1×8 | Run 2: golden EP16+CP16, 2×8 |
|---|---:|---:|
| weight floor (idle) | 216.4 GiB/GPU | 130.7 GiB/GPU |
| slope m | 0.70 MiB/token/GPU | 0.33 MiB/token/GPU |
| comfortable zone | ≤64K | ≤~384K |
| **max proven context** | **128K** | **512K** |
| first failure | 144K — LM-head fp32 chunk (1.18 GiB) | 640K — MoE sort buffer (15.0 GiB) |
| wall at 128K | 218s (thrashing) | 32s (healthy) |

2× GPUs bought 4× context: half from the halved weight floor, half from CP16
splitting each sequence twice as thin.

## Fits (y = peak max GPU MiB, x = context tokens)

- **Run 1** (linear regime 16K–64K, n=4): `y = 0.697·x + 221,147`, R²=0.988.
  Intercept within 0.2% of the measured idle floor (221,599 MiB).
- **Run 2** (unsaturated, n=14): `y = 0.327·x + 145,746`, R²=0.946. Aggregate
  variants (median/min-rank, ≤256K-only) land in m ∈ [0.25, 0.38],
  b ∈ [140K, 152K].

Both curves **saturate** once reserved hits ~99.6% of the device: the caching
allocator pins at the ceiling and survives via flush-and-realloc cycles
(`empty_cache` + `cudaMalloc`, device-syncing each time) — peaks stop tracking
demand and wall time degrades ~10×. Fit lines are only valid below saturation.
Naive all-point fits (run 1: R²=0.82, run 2: R²=0.93) are misleading; don't
use them.

## Run 1 — EP8+CP8, 1×8 B300 (job wxll97w)

Config: golden B300 recipe halved onto one node — `tp=1 pp=1 ep=8 cp=8 etp=1`
(dp=1). Idle 221,599 MiB/GPU.

| L | peak max MiB/GPU | GiB | spread MiB | wall s |
|---:|---:|---:|---:|---:|
| 16,384 | 233,387 | 227.9 | 382 | 3.1 |
| 32,768 | 243,883 | 238.2 | 380 | 19.4 |
| 49,152 | 253,223 | 247.3 | 620 | 19.5 |
| 65,536 | 268,363 | 262.1 | 758 | 21.4 |
| 81,920 | 274,075 | 267.7 | 360 | 78.8 |
| 98,304 | 274,013 | 267.6 | 1,098 | 104.2 |
| 114,688 | 273,937 | 267.5 | 940 | 125.7 |
| **131,072** | 274,099 | 267.7 | 288 | 217.8 |
| 147,456 | — | OOM | — | died 154s |

144K OOM: `Tried to allocate 1.18 GiB` in the chunked LM head
(`_project_logits`: `base_logits + delta.float()` — fixed 2048-token ×
154,880-vocab fp32 chunk). 258.33 GiB live, 565 MiB device-free, 758 MiB
trapped in partially-used segments — fragmentation at the margin after genuine
exhaustion. EP8 is the *minimum* expert sharding on 8 GPUs (~725B expert
params ⇒ ~181 GiB/rank experts alone).

## Run 2 — golden EP16+CP16, 2×8 B300 (job 3188jrw)

Config: the current golden B300 recipe verbatim — `tp=1 pp=1 ep=16 cp=16
etp=1`, 2 nodes (dp=1). Idle 133,833 MiB/GPU.

| L | peak max MiB/GPU | GiB | spread MiB | wall s |
|---:|---:|---:|---:|---:|
| 16,384 | 141,247 | 137.9 | 100 | 12.3 |
| 32,768 | 146,847 | 143.4 | 400 | 13.1 |
| 49,152 | 153,227 | 149.6 | 120 | 16.0 |
| 65,536 | 160,847 | 157.1 | 420 | 16.3 |
| 81,920 | 168,689 | 164.7 | 1,660 | 25.2 |
| 98,304 | 187,929 | 183.5 | 12,720 | 33.2 |
| 114,688 | 195,769 | 191.2 | 6,520 | 37.3 |
| 131,072 | 201,229 | 196.5 | 7,600 | 32.2 |
| 163,840 | 209,509 | 204.6 | 9,940 | 36.6 |
| 196,608 | 216,149 | 211.1 | 9,960 | 43.4 |
| 229,376 | 224,249 | 219.0 | 13,060 | 40.2 |
| 262,144 | 230,409 | 225.0 | 12,660 | 44.4 |
| 327,680 | 248,549 | 242.7 | 18,920 | 60.0 |
| 393,216 | 263,229 | 257.1 | 21,740 | 68.4 |
| 458,752 | 273,681 | 267.3 | 14,372 | 89.0 |
| **524,288** | 273,909 | 267.5 | 3,368 | 167.4 |
| 655,360 | 274,041 | 267.6 | 5,060 | OOM (150s) |

640K OOM: `Tried to allocate 15.02 GiB` in the **MoE token permutation**
(`transformer_engine …/triton/permutation.py:419 sort_chunks_by_map` via
`moe_chunk_sort_forward`) — the ragged alltoall sort buffer (640K/16 ranks ×
top-8 = 327,680 routed rows/rank). GPU 2 had 255.5 GiB live, 5.2 GiB free.
This is the `moe_capacity`-unset path biting: the Kimi golden configs set
`capacity_factor=1.0, pad_to_capacity=true` precisely to bound this buffer.
That knob (or `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, suggested by
both OOM messages) is the lever if >512K is ever needed on this mesh.

## Caveats

- CE loss, one sequence per step, random tokens (no routing realism; MoE
  buffers ragged since `moe_capacity` unset, as golden).
- Peak = reserved high-water via `nvidia-smi`; per-GPU spread ≤1.1 GiB in run
  1, up to ~21 GiB mid-sweep in run 2 (allocator state divergence across 16
  ranks; max-rank is the binding constraint and is what the fits use).
- LoRA r=32; full FT is not a supported GLM-5.2 path.
- 1-node boxes ship without Slurm; run 1 used direct-exec wrappers around the
  generated `run_trainer_node.sh` chain (papercut `pc_0f02832bd716`).
- On 2-node boxes the trainer HTTP server lands on Slurm's node 0
  (alphabetical hostname), not the k8s leader — run 2's client ran on
  `tj-3188jrw-1` (papercut `pc_f38f091cf0e4`).

## Related

- `glm/profiling_256k.md` — 2026-07-17, 4×8 B200 EP32/CP32: 256K fits at
  159.5 GiB steady-state (LoRA r16). Consistent with this study: the B200 mesh
  spreads experts 32 ways, so its floor is lower still.

## Artifacts (this folder)

- `results_run1_ep8cp8_1node.jsonl` — run 1 measurements (per-L per-GPU peaks, wall)
- `results_run2_ep16cp16_2node.jsonl` — run 2 measurements (per-L per-GPU
  peaks joined from the per-node poller logs, wall)
- `sweep.py` / `sweep2.py` — sweep clients; `mem_poller.sh` — per-node
  sampler; `analyze.py` / `analyze2.py` — join + fit
- `trainer_ep8cp8_1node.json`, `trainer_ep16cp16_2node.json`,
  `trainer_server.json` — trainer configs used
