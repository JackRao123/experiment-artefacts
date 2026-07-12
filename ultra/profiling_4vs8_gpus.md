# Nemotron-3-Ultra NVFP4 sampler — 4×B200 (TP=4) vs 8×B200 (TP=8), single node

Linear: [TRN-1488](https://linear.app/baseten/issue/TRN-1488/test-nemotron-3-ultra-sampler-on-4xb200-vs-8xb200).
Run 2026-07-10 on box `qk4yz2w` (1×8 B200, 183,359 MiB ≈ 179.1 GiB/GPU,
Birch/Weka cluster).

Compares the **committed golden NVFP4 sampler recipe** for
`Model.NEMOTRON_3_ULTRA` (`sampler_configs.py` B200/S256K: `gpu_count=8`,
`enable_lora`, `max_lora_rank=64`, `max_loras=4`, `max_num_seqs=1000`,
`disable_custom_all_reduce`) running on **8 GPUs (TP=8)** vs the same recipe at
**4 GPUs (TP=4, `CUDA_VISIBLE_DEVICES=0-3`)**. The TRN-1488 question: is
4×B200 good enough for RL sampling so the other 4 GPUs can be freed, or does
8×B200 earn its keep?

**TL;DR: switch to 4×B200.** 256k serves with zero preemptions, TP=4 is
**+59–64% throughput per GPU** at every context length, and the KV pool only
drops to 0.625× (not the naive ≤0.5×) because the per-token page cost halves
at TP=4. The trade is the per-replica concurrency ceiling: 17 vs 27.5
concurrent full-256k sequences. PR flips `gpu_count` 8→4.

Serving stack (identical in both configs, per the validated NVFP4 recipe in
`tests/comparison_studies/nvfp4_compare/findings.md`):

- Checkpoint `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` (ModelOpt
  MIXED_PRECISION: routed experts NVFP4, Mamba/shared-experts FP8,
  attention/embeddings BF16), **fp8 KV cache**.
- vLLM 0.22.0 (`default.env` stack, the `trainers_main/sampler/.venv`),
  `_BASETEN_SERVED_MODEL=nemotronhforcausallm` + `MODEL_PATH` → gated
  `_vllm_nemotron_patches.py` applied (verified per cell: "Applied
  Nemotron-3-Ultra NVFP4+LoRA vLLM patches", `VLLM_CUTLASS` NvFp4 MoE),
  `moe_backend=cutlass`, `enable_flashinfer_autotune=false`,
  `disable_custom_all_reduce=true`, `gpu_memory_utilization=0.90`,
  fastsafetensors, the 16-size production cudagraph capture set (1…1000).
- LoRA on in both configs (`enable_lora`, `max_lora_rank=64`, `max_loras=4`);
  the GSM8K-RL Ultra adapter `step-199` (r32) attached to **every** request.

Mamba `n_groups=8` and the attention heads divide evenly at both TP=8 and
TP=4, so TP=4 is architecturally legal; everything below is measurement.

## Method

**One boot per TP** at the golden `max_model_len=262144` (what production
serves), then four data-length buckets (8k/32k/131k/256k) flow through the
same engine, ascending so first-call warmup lands in the cheapest bucket.
Harness: `sampler_4vs8/` (`bench.py` one TP cell per process; `run_sweep.sh`
both cells; raw outputs in `sampler_4vs8/out/*.json`). Per TP cell:

- **Boot / weight-load**: wall clock to engine-ready; weights/GPU and load
  seconds from vLLM's `Model loading took` line. Cold vs warm noted by boot
  order (the box page-caches the 329 GB snapshot after the first load).
- **KV pool**: vLLM's authoritative `GPU KV cache size: N tokens` /
  `Maximum concurrency for 262144 tokens…` boot lines. Per-context max
  concurrency is **derived**: pool tokens ÷ ctx (the spare-KV estimate).
- **Throughput buckets**: synthetic random-token prompts (prompt = ctx −
  1 040, `max_new_tokens=1024`, `ignore_eos`, LoRA on every request), N ≈
  1.05× the pool-implied concurrency for that ctx (capped at
  `max_num_seqs=1000`), so the pool is saturated by construction. Aggregate
  tok/s over each bucket's wall clock; per-GPU = ÷8 or ÷4.
- **Max memory**: 1 Hz `nvidia-smi memory.used` over all 8 GPUs for the whole
  cell (vLLM pre-fills to 0.90 utilization at boot, so this is ~flat across
  buckets; the TP=4 rows double as proof GPUs 4–7 stay free).
- vLLM's offline engine does not emit the periodic `Running:` scheduler
  lines (server-mode feature), so scheduler-observed peak concurrency is not
  reported; saturation comes from the 1.05× sizing + all requests completing.

## E1 — model fit & boot

| config | boots @256k? | weights/GPU (GiB) | weight load (s) | boot wall, warm (s) |
| --- | --- | ---: | ---: | ---: |
| TP=8 (8×B200) | yes | 44.02 | 82.7 | 426.9 |
| TP=4 (4×B200) | **yes** | 82.23 | 101.4 | 370.1 |

- Cold boot (first-ever load on the box, TP=8 validation cell, 8k engine):
  772.5 s wall — page-cache fill + Triton/inductor JIT dominate; warm boots
  are the steady-state number.
- TP=4 loads a 2× weight shard per GPU for only +23% load time (101.4 vs
  82.7 s) and its total warm boot is *faster* (370 vs 427 s) — cudagraph
  capture and init scale with rank count.
- Reliability: every boot attempted after the box was correctly provisioned
  came up clean (TP=8 ×2, TP=4 ×1); no flakes to report, but N is small.
  (One pre-provisioning failure was environmental: missing `python3.12-dev`
  kills Triton JIT — see `devbox-up` notes in the memory index.)

## E2 — KV pool & derived max concurrency per context

One pool per TP (boot at 262 144, LoRA buffers on); per-ctx concurrency =
pool ÷ ctx. Matches vLLM's own `Maximum concurrency` line at 256k.

| config | avail KV/GPU (GiB) | KV pool (tok) | conc @8k | @32k | @131k | @256k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP=8 | 104.81 | 7,216,865 | 881 | 220 | 55.1 | **27.5** |
| TP=4 | 65.54 | 4,510,515 | 551 | 138 | 34.4 | **17.2** |
| TP4/TP8 | — | **0.625** | 0.625 | 0.625 | 0.625 | 0.625 |

- **The pool ratio is 0.625, well above the naive ≤0.5.** Total KV+Mamba
  bytes are 262.2 GiB (TP=4) vs 838.5 GiB (TP=8), a 0.313 byte ratio — yet
  the token ratio is 0.625, because the effective cost per token is **~62
  KB at TP=4 vs ~125 KB at TP=8** (all-GPU total). The hybrid layout pads
  attention pages to match Mamba pages (block size 4224 tokens, "Padding
  mamba page size by 1.78%" at boot); that page-granularity overhead is
  per-rank, so halving the rank count halves the per-token padding cost.
  (Observed arithmetic + boot-log mechanism; not isolated further.)
- Config context: this production config measured 7.10M tok / 867× on an 8k
  TP=8 boot (validation cell) — the pool barely moves with `max_model_len`.
  The `nvfp4_compare` spike's 12.07M @8k / 34.08M @256k came from a leaner
  config (`max_loras=1`, r32, default capture set); pools are not comparable
  across configs, and these production-config numbers are the
  decision-relevant ones.

## E3 — throughput at saturating concurrency (synthetic)

Prompt = ctx − 1 040 random tokens, gen = 1 024, `ignore_eos`, LoRA on every
request; N ≈ 1.05× the pool-implied concurrency (capped at 1000); all buckets
through the single 262 144 engine. Aggregate = (prompt+gen tokens)/wall.

### TP=8 (8×B200)

| ctx | N | prompt tok | gen tok | wall (s) | total tok/s | gen tok/s | tok/s/GPU | preempt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 192   | 927 | 6,629,904 | 949,248 | 482.2 | 15,718 | 1,969 | 1,965 | 0 |
| 32 768  | 233 | 7,392,624 | 238,592 | 458.7 | 16,638 | 520 | 2,080 | 0 |
| 131 072 | 59  | 7,671,888 | 60,416  | 477.4 | 16,196 | 127 | 2,025 | 0 |
| 262 144 | 30  | 7,833,120 | 30,720  | 493.8 | 15,924 | 62  | 1,991 | 0 |

### TP=4 (4×B200)

| ctx | N | prompt tok | gen tok | wall (s) | total tok/s | gen tok/s | tok/s/GPU | preempt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 192   | 580 | 4,148,160 | 593,920 | 380.1 | 12,475 | 1,563 | **3,119** | 0 |
| 32 768  | 146 | 4,632,288 | 149,504 | 350.8 | 13,631 | 426 | **3,408** | 0 |
| 131 072 | 38  | 4,941,216 | 38,912  | 374.1 | 13,311 | 104 | **3,328** | 0 |
| 262 144 | 20  | 5,222,080 | 20,480  | 408.6 | 12,830 | 50  | **3,208** | 0 |

### Max `memory.used` (GiB) per GPU over each TP cell, order g0–g7

```
TP=8 : 166.8 166.8 166.9 166.8 166.9 166.8 166.8 166.9
TP=4 : 165.7 165.7 165.7 165.7 | idle: 0.4  0.0* 1.4  0.2
```
\* g5 showed 62.6 GiB in exactly the first 1 Hz sample of the TP=4 cell —
teardown lag of the previous TP=8 cell's exiting worker; 0 from t=1 s on.

## E4 — does 256k work on 4×B200?

**Yes, comfortably.**

- **Boots at `max_model_len=262144`**: yes (82.2 GiB weights + 65.5 GiB KV +
  2.7 GiB graphs per GPU; peak 165.7 / 179.1 GiB).
- **KV pool @256k**: 4,510,515 tokens ⇒ **17 concurrent full-256k
  sequences** (vLLM's own line: 17.21×).
- **Saturating 256k run**: 20/20 requests completed, **0 preemptions**, no
  OOM.
- **Throughput vs TP=8 @256k**: 12,830 vs 15,924 total tok/s — **0.81× per
  replica on half the GPUs = 1.61× per GPU**.

## Findings

- **TP=4 wins throughput/GPU by +59–64% at every context length**
  (3.1–3.4k vs 2.0–2.1k tok/s/GPU). A single TP=4 replica still delivers
  0.79–0.82× the absolute throughput of TP=8. Two TP=4 replicas on one node
  would deliver ~1.6× a TP=8 replica's tokens — expected near-linear since
  each stays inside its NVLink domain, but **colocation was not measured**.
- **Capacity costs less than expected**: 0.625× pool (17.2× vs 27.5× @256k)
  because per-token page overhead halves at TP=4 (see E2). Per-sequence
  decode speed at full saturation is *better* on TP=4 (e.g. @8k: 1563/580 =
  2.7 tok/s/seq vs 1969/927 = 2.1) — fewer sequences share the aggregate.
- **Boot favors TP=4** (370 vs 427 s warm) despite the 2× per-GPU shard.
- **What TP=8 still buys**: a higher per-replica ceiling — >17 concurrent
  full-256k rollouts in one engine, and ~23% faster absolute prefill for
  latency-critical single bursts. Neither matters for throughput-oriented RL
  sampling with ≤17 concurrent 256k streams per replica.
- Caveats: offline engine (no HTTP serving layer), synthetic random tokens
  (no prefix-cache hits), logprobs not requested, single run per cell.

## Verdict

**Recommend 4×B200 (TP=4) as the default Nemotron-3-Ultra RL sampler
footprint.** It serves the full 256k context with 17× concurrency headroom
and zero preemptions, is ~1.6× more GPU-efficient than TP=8 on every context
length, boots faster, and frees 4 B200s per node (or doubles sampler
capacity via two replicas). Flip `gpu_count` 8→4 in `sampler_configs.py`
`NEMOTRON_3_ULTRA` B200/S256K. Revisit only if a workload needs >17
concurrent full-256k rollouts in a single replica. Note the flip changes the
compile/AOT cache key (gpu_count is part of the S3 path), so the first
deploy at TP=4 rebuilds its cudagraph/inductor artifact.

## Reproduce

```bash
# 1-node 8×B200 devbox (needs python3.12-dev!), NVFP4 snapshot + step-199
# adapter on the shared FS (Birch/Weka has both), vLLM 0.22 sampler venv.
cd /root/.cache/user_artifacts/sampler_4vs8   # copy of experiment_artefacts/ultra/sampler_4vs8
./run_sweep.sh          # TP=8 cell then TP=4 cell; ~70 + ~40 min
./run_cell.sh 4 262144  # just the E4 cell
# per-cell outputs: out/tp{8,4}_boot262144.{log,smi.csv,result.json,summary.json}
```

Raw results archived in `sampler_4vs8/out/`. Box `qk4yz2w` stopped after the
run; its logs persist on the Birch/Weka shared FS.
