# Nemotron-3-Ultra-550B-A55B — vLLM LoRA serving / 256k-context profile

Goal: stand up the vLLM sampler for **`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`**
on a single 8× B200 node, confirm it serves at the model's full **262,144-token**
context, and profile how many sequences / how big a KV-cache pool fit.

**Verdict: YES — Ultra serves at the full 262k context on a single 8× B200 node**,
but **only after fixing the committed golden sampler config**, which does not boot
as written (see §1). The load-bearing levers are `max_loras` (the 512-expert LoRA
buffers, not the weights, are what OOMs), `max_num_seqs` (capped by Mamba state
blocks, not attention KV), and disabling vLLM's custom all-reduce (CUDA-graph
capture crash at TP=8).

Hardware: single node, 8× NVIDIA B200 (183,359 MiB total ≈ 178.35 GiB usable/GPU).
Box `w57o7m3` worker node `…-multinode-0-1` (leader node 0 was busy running
`make test-server-gpu`, so the sampler was placed on an idle worker via `srun`).
Stack: vLLM 0.22.0, torch 2.11.0+cu129, transformers 5.12.0 (the same venv built by
`setup_sampler_venv.sh` and validated in `SAMPLER_REPORT.md`).

Model shape (from the HF `config.json`): `NemotronHForCausalLM`, hybrid
Mamba2 + Transformer-MoE. **108 layers = 48 mamba / 48 moe / 12 attention**;
hidden 8192; **512 routed experts, top-22**, 1 shared expert; attention
`head_dim=128`, `num_key_value_heads=2` (so attention KV is tiny — 12 layers ×
2 KV heads); Mamba `num_heads=256`, `head_dim=64`, `ssm_state_size=128`,
`n_groups=8`; vocab 131072; `max_position_embeddings=262144`. ~1.1 TB bf16
weights → **~137 GiB/GPU at TP=8**.

---

## 1. The committed golden config does NOT boot (three sequential failures)

`SAMPLER_CONFIGS[NEMOTRON_3_ULTRA][B200][S256K]` is currently
`gpu_count=8, enable_lora=True, max_lora_rank=64, max_loras=4, max_num_seqs=1000`
— copied from Super. On Ultra every one of those last two values is wrong, and a
TP=8 vLLM default also has to be overridden:

| # | Symptom (single 8× B200, TP=8, max_model_len=262144) | Root cause | Fix |
|---|---|---|---|
| 1 | `torch.OutOfMemoryError` in `lora/layers/fused_moe.py::_create_lora_b_weights` — GPU at **178.18 / 178.35 GiB** during model load, before KV profiling | The per-expert MoE-LoRA B buffers scale with `max_loras`. With **512 experts × max_loras=4 × rank 64** they add **~38 GiB/GPU** on top of the ~137 GiB weights → OOM | `max_loras=1` (drops the LoRA buffers to ~10 GiB/GPU). This is exactly why the other huge MoE golden config, `QWEN3_5_397B_A17B`, already uses `max_loras=1`. |
| 2 | `ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (712). Each decode sequence requires one Mamba cache block` (crash at CUDA-graph resolve) | For a hybrid model, **each in-flight sequence needs one Mamba state block**. After the weights+LoRA, only **712 Mamba blocks** fit, so `max_num_seqs` must be ≤ 712. The golden `max_num_seqs=1000` (and vLLM's default 1024) exceed it. | `max_num_seqs ≤ 712` (used 512). |
| 3 | `Failed: Cuda error custom_all_reduce.cuh:455 'invalid argument'` during CUDA-graph capture (~94/102), worker `exit()`s, engine init fails | vLLM's custom all-reduce CUDA-graph-capture path issues a CUDA call that returns `cudaErrorInvalidValue` at TP=8 on this model. **Reproduces in both FULL and PIECEWISE cudagraph modes; not memory-pressure** (see §6). | `--disable-custom-all-reduce` (NCCL all-reduce; costs ~0.9 GiB of KV pool, negligible perf on a 550B model) |

**Working launch (boots to "Application startup complete" and serves):**

```bash
cd /b10/workspace/baseten/trainers/sampler
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HOME=/root/.cache/user_artifacts/huggingface HF_HUB_OFFLINE=1 \
MODEL_ID=nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
TENSOR_PARALLEL_SIZE=8 MAX_SEQ_LENGTH=262144 \
ENABLE_LORA=true MAX_LORA_RANK=64 MAX_LORAS=1 MAX_NUM_SEQS=512 \
HOST=0.0.0.0 PORT=8001 \
VLLM_EXTRA_ARGS="--trust-remote-code --disable-custom-all-reduce" \
.venv/bin/python -m sampler.vllm_server
```

Cold boot: weights load ~3.5 min; `init engine (profile, create kv cache, warmup
model)` 117 s (compilation 9.8 s); total to serving ~5.5 min.

---

## 2. Memory + KV-cache profile (the boot config above, gpu_memory_utilization=0.92)

Per-GPU at steady state: **171,244 / 183,359 MiB used (~167 GiB)** on all 8 GPUs
(= the 0.92 util reservation), ~12 GiB untouched. Rough per-GPU split:

| component | ~GiB/GPU |
|---|---|
| model weights (1.1 TB / 8) | ~137 |
| MoE-LoRA buffers (max_loras=1, rank 64, 512 experts) | ~10 |
| activations + CUDA graphs + NCCL/Triton/Mamba workspaces | ~11 |
| **KV cache pool (attention + Mamba, unified)** | **8.49** |

vLLM's reported KV-cache profile:

```
Available KV cache memory: 8.49 GiB
GPU KV cache size: 1,426,783 tokens
Maximum concurrency for 262,144 tokens per request: 5.44x
```

Hybrid KV-manager detail: vLLM unifies the attention and Mamba page sizes —
`Setting attention block size to 2080 tokens to ensure attention page size >=
mamba page size`, `Padding mamba page size by 0.24%`. The 1,426,783-token pool is
that unified cache.

### How many sequences / how much pool — the two binding limits

- **Attention/unified KV pool: 1,426,783 tokens.** At a full 262,144-token request
  that's **5.44 concurrent full-length sequences** (≈ what you can serve at max
  context).
- **Mamba state: ~712 blocks**, one per in-flight sequence regardless of length →
  **≤ ~712 concurrent sequences** for short requests, and `max_num_seqs` must be
  set ≤ 712 at boot or the engine refuses to start (failure #2).
- Net: long-context serving is **KV-pool bound (~5–6 concurrent at 262k)**;
  many-short-request serving is **Mamba-block bound (~712)**.

These are small because the 137 GiB/GPU of weights dominate an 8× B200 node. Raising
`gpu_memory_utilization` (0.92 → ~0.95) buys roughly +5 GiB/GPU of KV (~+0.9M
tokens, ~9× concurrency at 262k); going past ~0.96 risks the load-time peak OOM.
More headroom needs more GPUs (PP=2 / 2 nodes halves weights/GPU to ~69 GiB,
freeing ~70 GiB/GPU for KV + a larger Mamba pool).

> Contrast with Super (120B, `SAMPLER_REPORT.md`): there KV was effectively free
> (~25 M tokens, 98× concurrency at 262k) because weights were only ~30 GiB/GPU.
> Ultra is the same arch but ~4.6× bigger, so the weights crowd out the KV pool —
> concurrency, not the context length, is the scarce resource.

---

## 3. Generation evidence (full-context serving works)

`/v1/completions`, temperature 0, against the running server:

| prompt_tokens | max_tokens | latency | result |
|---|---|---|---|
| 240,001 | 16 | 13.1 s | OK, coherent continuation |
| 261,001 | 16 | 10.1 s | OK (within the 262,144 ceiling) |
| 4 × 240,001 concurrent | 16 each | 35.5 s wall (per-req 13.5–35.4 s) | all 4 OK — concurrent full-context batching confirmed |

(`prompt_tokens` is vLLM's real count; the phrase tokenizes ~10.9 tok/repeat.)

---

## 4. Recommended fix to the golden sampler config

`models/src/loops_models/model_configs/sampler_configs.py`,
`SAMPLER_CONFIGS[NEMOTRON_3_ULTRA][B200][S256K]`:

```python
S(
    gpu_count=8,
    enable_lora=True,
    max_lora_rank=64,
    max_loras=1,        # was 4 — 512-expert MoE-LoRA buffers OOM at 4
    max_num_seqs=512,   # was 1000 — must be <= ~712 Mamba cache blocks
)
```

Plus the launch needs `--disable-custom-all-reduce` (failure #3). That flag is a
launch/engine-arg concern, not a field on `GoldenSamplerConfig` — wire it through
the sampler launch path for NemotronH-at-TP8 (or add it to `VLLM_EXTRA_ARGS` in the
deploy), the same way `--gdn-prefill-backend triton` is hardcoded in
`vllm_server.py`. NOT YET APPLIED — flagged for review.

---

## 5. Root-cause investigation of failure #3 (custom all-reduce crash)

`custom_all_reduce.cuh:455` is inside vLLM's `CUDACHECK(...)` macro, which on **any**
CUDA error prints `Failed: Cuda error <file>:<line> '<msg>'` and then calls
`exit(EXIT_FAILURE)` — a hard process kill, **no NCCL fallback**. So the all-reduce
that fails takes the whole worker (and thus engine init) down. The `'invalid
argument'` is `cudaErrorInvalidValue` raised by a CUDA call on the custom-AR
**CUDA-graph-capture path**.

Why that path is fragile (from the v0.22.0 source comments): during graph capture
the peer GPU pointers aren't known yet, so custom-AR defers them — it pushes the
input into `graph_unreg_buffers_` and only resolves the cross-rank IPC handles
afterward (`get_graph_buffer_ipc_meta` → all-gather → `register_graph_buffers` →
`cudaIpcOpenMemHandle`). The all-reduce is used whenever the tensor is `< max_size`
(8 MiB); the decode all-reduce is `num_seqs × hidden(8192) × 2B = num_seqs × 16 KiB`,
so every captured decode batch `< 512` takes the custom-AR path.

### Experiments run (idle worker nodes, identical model/TP/LoRA, custom-AR ENABLED)

| node | variable | cudagraph_mode | gpu_mem_util | outcome |
|---|---|---|---|---|
| 0-1 | reproduction | FULL_AND_PIECEWISE (default) | 0.92 | **crash** at `.cuh:455`, in the `decode, FULL` phase (~94/102) |
| 0-1 | `cudagraph_mode=PIECEWISE` | PIECEWISE only | 0.92 | **crash** at `.cuh:455`, in the `mixed prefill-decode, PIECEWISE` phase (~94/102) |
| 0-2 | util=0.80 (intended memory control) | FULL_AND_PIECEWISE | 0.80 | inconclusive — died *earlier* with `ValueError: No available memory for the cache blocks` (0.80 budget ≈146.6 GiB barely covers ~147 GiB weights+LoRA → 0 KV), never reached capture. Mis-designed control: lower util = *less* room, not more. |

Key results:
- **My first hypothesis ("only the FULL decode graph triggers it") was wrong.**
  PIECEWISE-only crashes too, at the *same* capture iteration. In the original
  FULL_AND_PIECEWISE run the PIECEWISE phase happened to pass and FULL failed; in
  isolation PIECEWISE fails as well. The trigger is **custom-AR during cudagraph
  capture in general**, not the FULL graph specifically.
- All 8 ranks logged `custom_all_reduce.py:215 Registering 0 cuda graph addresses`
  *before* the crash, i.e. the IPC buffer pool allocated fine — **the failure is the
  per-capture `allreduce()` call, not buffer-pool allocation / memory pressure.**
  Corroborated: the PIECEWISE run at 0.92 had *more* KV headroom (11.07 GiB vs the
  8.49 GiB of the FULL run) and crashed identically, so freeing memory does not help.
  (The util=0.80 control was mis-designed — lower util shrinks the budget, so it OOM'd
  on KV-config *before* capture rather than relieving custom-AR; ignore it.)
- The crash lands at the **same capture index (~94/102)** every time → a **specific
  (larger) batch size** in the **LoRA-specialized** capture set
  (`cudagraph_specialize_lora=True` doubles 51 sizes → 102). The LoRA-specialized
  graph + custom-AR registered-buffer path is the prime suspect; exact size not
  isolated further.

### Why Super didn't need the flag but Ultra does — UNKNOWN (honest gap)

Super (`SAMPLER_REPORT.md`) booted on the same 8× B200 / vLLM 0.22 stack without
`--disable-custom-all-reduce`. I did **not** root-cause the Super-vs-Ultra
difference; it would require re-running Super under matched settings. Candidate
differences (untested): Ultra runs LoRA-specialized cudagraphs under far tighter
memory, different capture-size sets, and a much larger per-layer all-reduce.

### Recommended fix (and why not chase a custom-AR-preserving config)

**Disable custom all-reduce for NemotronH at TP=8.** Rationale:
1. It crashes in every cudagraph mode tried, so there's no free "keep custom-AR via
   PIECEWISE" win.
2. `CUDACHECK` `exit()`s on failure with no fallback — custom-AR is a latent footgun
   on this path even if some config dodged it today.
3. On a 550B model the TP all-reduce is a rounding error of step time; NCCL is fine.

Untested alternative if one *insists* on keeping custom-AR:
`VLLM_CUSTOM_ALLREDUCE_ALGO=1stage` (the source exposes a 1stage/2stage override;
the crash is likely on the 2stage path used for the larger batches). Not worth it
here vs. just disabling.
