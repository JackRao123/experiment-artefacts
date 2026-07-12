# Nemotron-3-Ultra NVFP4 sampler spike — profiling & findings

Goal: decide **Option A** — serve the Ultra base in **NVFP4** on the sampler (1×8
B200) while the trainer stays **bf16**, freeing weight memory for a larger
KV/Mamba pool. Two questions: (1) does unmerged LoRA even run on the NVFP4
checkpoint's **FP8 Mamba/shared-expert** layers, and (2) is the NVFP4↔bf16
per-token logprob drift small enough that the RL `cispo` correction stays
healthy.

**Verdict: GO.** NVFP4 base + unmerged LoRA serves on the pinned vLLM 0.22.0
(no version bump) with a small, well-understood patch set, gives ~6x the KV
pool of bf16 (decomposed below), and adds only ~0.005 nats of mean logprob
drift above the measurement floor.

---

## Environment

- Boxes: `w6le12w` (5×8 B200) then `wnmgv43` (1×8 B200) — same project
  `jrao123-hyd`, so `user_artifacts` (the staged checkpoints/outputs) persisted
  across the swap. 183,359 MiB/GPU.
- Stack: **vLLM 0.22.0**, torch 2.11.0+cu129 (the `trainers_mn/sampler/.venv`).
- NVFP4 checkpoint: `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` (ModelOpt
  `MIXED_PRECISION`): routed experts **NVFP4** (group 16), Mamba `in_proj`/
  `out_proj` + shared experts **FP8**, attention/embeddings **BF16**, **FP8 KV**.
- bf16 checkpoint: `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` (~1.1 TB,
  ~137 GiB/GPU at TP=8).
- Adapter: GSM8K-RL `step-199`, LoRA r=32 / α=64. Targets `in_proj`, `out_proj`,
  `q/k/v/o_proj`, `shared_experts.{up,down}_proj`. **No routed-expert LoRA.**
  (So 2 of 3 target groups land on FP8 base layers — the highest-risk piece.)
- **Where the adapters live:**
  - On the devbox (full PEFT dirs vLLM can load: `adapter_config.json` +
    `adapter_model.safetensors`): all GSM8K-RL checkpoints under
    `/root/.cache/user_artifacts/rl/gsm8k-ultra-weights/sampler_weights/`
    (`step-4` … `step-199`); the spike's behavior adapter is
    `.../sampler_weights/step-199`. These are **Ultra** adapters
    (`base_model_name_or_path: nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`).
  - On the laptop (outside trainers repo):
    `~/Documents/nemo3ultra_rl_overnight/adapters/step-199` and
    `…/step-60` — **safetensors only, no `adapter_config.json`** (drop one in
    before pointing vLLM at these, per README).
  - **Not usable for Ultra:** `/root/.cache/user_artifacts/nemotron_lora_test/sampler_weights/pirate-v{1,2}`
    are **Super-120B** adapters (`…Nemotron-3-Super-120B…`) and will not load on
    the Ultra base — don't reach for them by accident.
- Spike harness: `tests/comparison_studies/nvfp4_compare/` (`score.py` = vLLM
  generate/rescore, `compare.py` = CPU divergence metrics).

---

## The validated serving recipe (vLLM 0.22.0, this checkpoint, B200)

**Engine args:**
- `tensor_parallel_size=8`, `kv_cache_dtype=fp8`
- `disable_custom_all_reduce=true` (custom AR crashes during CUDA-graph capture at TP=8 NemotronH)
- `enable_flashinfer_autotune=false` (FlashInfer fp8_gemm autotuner segfaults — see below)
- `moe_backend=cutlass` (avoid Marlin NVFP4-MoE — see below)
- `enable_lora=true`, `max_lora_rank>=32`

**Patches** (currently in `sampler/sampler/_vllm_patches.py::apply_patches`;
to be productionized as a gated `_vllm_nemotron_patches.py`):
1. **Force `FusedMoEConfig.is_lora_enabled=False`** at construction
   (`fused_moe/layer.py` sets it True whenever global LoRA is on). Lets the
   NVFP4 MoE backend selector accept CUTLASS.
2. **Drop the routed `FusedMoE` from LoRA-targetable modules**
   (`get_supported_lora_modules`), so vLLM never builds a `FusedMoEWithLoRA`
   wrapper (which asserts the experts kernel supports LoRA — only Marlin does).
3. (`_patch_disable_moe_lora_kernels`, the Kimi MoE flag — belt-and-suspenders.)

All three are correct because **we never LoRA the routed experts**; attention /
Mamba / shared-expert LoRA go through the normal LoRA layer system, unaffected.

---

## Failure chain on vLLM 0.22.0 (and the fix for each)

The pinned stack does not serve this NVFP4 checkpoint out of the box. The
failures, in order, were all about the **NVFP4 MoE + LoRA** coupling and
FlashInfer kernels — none were the FP8-Mamba-LoRA risk we feared.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `generate` w/ LoRA: crash in `prepare_nvfp4_moe_layer_for_marlin` (`static_assert`/compile) | `--enable-lora` forces the **Marlin** NVFP4-MoE backend; Marlin kernel prep won't compile on this CUDA wheel/driver | `moe_backend=cutlass` |
| 2 | base-only (no LoRA): **segfault** in `[AutoTuner] Tuning fp8_gemm` | FlashInfer FP8 GEMM autotuner segfaults on this box | `enable_flashinfer_autotune=false` |
| 3 | LoRA: `ValueError: NvFp4 MoE backend 'VLLM_CUTLASS' does not support ... kernel does not support LoRA` | selector reads `FusedMoEConfig.is_lora_enabled` (True under global LoRA); only Marlin "supports LoRA" | force `is_lora_enabled=False` (patch 1) |
| 4 | LoRA: `AssertionError: CutlassExpertsFp4 does not support LoRA` | global LoRA on ⇒ vLLM wraps the routed FusedMoE in `FusedMoEWithLoRA`, which asserts LoRA-aware experts | skip LoRA-wrapping the FusedMoE (patch 2) |

After patches 1–4 + the flags: **base NVFP4 and NVFP4+LoRA both serve and
generate, 0 segfaults.** `VLLM_CUTLASS` NVFP4 MoE selected; `FlashInferFP8ScaledMM`
for FP8 linears runs fine with autotune off.

> Interpretation: this is vLLM 0.22.0's "LoRA ⇒ MoE must be LoRA-aware (Marlin)"
> coupling colliding with broken Marlin/FlashInfer kernels on this box. A newer
> vLLM (with `LoRAExpertsMixin` for CUTLASS experts) would likely remove the need
> for patches 1/3/4 — worth revisiting at the next stack bump, but **not required**.

---

## Headroom (the point of Option A)

Same node / TP=8 / `max_model_len=8192` / `gpu_memory_utilization=0.92`, **MoE-LoRA
buffers disabled by the patches in both runs**:

| | weights/GPU | KV dtype | GPU KV pool | concurrency @ 8k |
|---|---|---|---|---|
| bf16 | ~137 GiB | bf16 | 2,061,312 tok | 251.6x |
| NVFP4 (base) | ~43 GiB | fp8 | 12,434,090 tok | 1517x |
| NVFP4 (+LoRA) | ~43 GiB | fp8 | 12,069,546 tok | 1473x |

**~6x KV pool / concurrency, NVFP4-recipe vs bf16-default.**

### Honest decomposition of the 6x (do not attribute it all to weights)

- **~3x** from NVFP4 weights freeing KV *memory* (~43 vs ~137 GiB/GPU), times
- **~2x** from **fp8 KV cache** (the NVFP4 recipe; 2x tokens/GiB), 
- = **~6x tokens**.

So: ~6x is the real **deployment** delta (NVFP4 ships with fp8 KV); the **pure
weight-isolation** number (same KV dtype) is **~3x**. A bf16 sampler run *also*
with fp8 KV would narrow the gap to ~3x. (Not yet measured — bf16+fp8-KV needs
attention-scale handling; TODO.)

### Why this bf16 number differs from `ULTRA_SAMPLER_REPORT.md` (PR #465)

That report (bf16, `max_model_len=262144`, `max_loras=1`) measured **8.49 GiB /
1,426,783 tok / 5.44x @ 262k**. Two reasons it differs from this spike's bf16
(2.06M / 252x @ 8k):
1. **Concurrency is the context denominator.** 1.43M ÷ 262k = 5.44x; 2.06M ÷ 8k =
   252x. The *pools* are the comparable quantity, not the concurrency.
2. **Pool size:** this spike's patches **disable the ~10 GiB/GPU 512-expert
   MoE-LoRA buffers** that the report's `max_loras=1` config allocated; that freed
   memory became KV pool (2.06M vs ~1.43M, adjusting for context overhead).
   I.e. the MoE-LoRA-skip patch is itself a memory win vs the committed golden
   config, independent of quantization.

---

## Rebase re-verification (2026-06-27) — committed gated patches, 8k + 256k

After rebasing this branch onto `main` (17 commits, incl. the GLM-5.2 / vLLM-0.23
work), the spike's ad-hoc patches were productionized into the **gated**
`sampler/sampler/_vllm_nemotron_patches.py` (commit `ab9d19f9`) and re-validated.

**What the rebase actually did to vLLM:** it did **not** bump vLLM for everyone.
vLLM is selected per-config via `GoldenSamplerConfig.stack` → `sampler/stacks/<stack>.env`.
The GLM commit *added* `vllm-0.23.env` (opt-in, GLM-5.2 only) and left
`default.env` = **vLLM 0.22.0** untouched. `stack` defaults to `"default"`, and
`Model.NEMOTRON_3_ULTRA` does not override it, so **our NVFP4 sampler still ships
on vLLM 0.22.0** — the exact stack these patches target. (The `/root/trainers`
venv happens to be 0.23, but that is not the default-stack wheel; the
production-representative env is the 0.22 `trainers_mn/sampler/.venv`, used below.)

**Patches apply cleanly on 0.22 via the committed gate:** with
`_BASETEN_SERVED_MODEL=nemotronhforcausallm` set, the plugin logs `Applied
Nemotron NVFP4+LoRA vLLM patches`, `ModelOptNvFp4FusedMoE.__init__(self,
quant_config, moe_config)` matches the patch signature, and vLLM selects
`VLLM_CUTLASS` NvFp4 MoE on every run. The +LoRA generate produced real tokens
(8 seqs, finite logprobs, mean −0.377) with **no** `CutlassExpertsFp4 does not
support LoRA` / `NvFp4 ... does not support LoRA` assertion → NVFP4 + unmerged
LoRA still serves.

Headroom (1×8 B200, TP=8, `gpu_memory_utilization=0.90`, fp8 KV, CUTLASS MoE;
vLLM's own boot-log numbers):

| ctx | variant | GPU KV pool | max concurrency |
|---|---|---|---|
| 8k | NVFP4 base | 12,434,090 tok | 1517.83x |
| 8k | NVFP4 (+LoRA) | 12,069,546 tok | 1473.33x |
| 256k | NVFP4 base | 35,108,020 tok | 133.93x |
| 256k | NVFP4 (+LoRA) | 34,078,720 tok | 130.00x |

- **8k is byte-for-byte identical to the pre-rebase numbers above** (12,434,090 /
  12,069,546) → the rebase + the refactor of the patches into the gated module
  changed nothing for the served path.
- **+LoRA costs ~3% of the KV pool** at both contexts (8k: 1473 vs 1517; 256k:
  130.0 vs 133.9) — the `enable_lora` buffers, not the routed experts.
- **256k fits with no OOM** and still leaves a 34–35M-token pool. The pool is
  *larger* in tokens at 256k than at 8k (35M vs 12M) even though concurrency is
  ~11x lower: at 8k vLLM captures the full cudagraph batch-size set (1…512),
  which reserves more non-KV memory; at 256k that capture set is smaller, freeing
  memory for KV. Concurrency is still far lower at 256k because each request
  reserves 32× more tokens (262,144 vs 8,192). (cudagraph-footprint explanation
  is the likely cause, inferred from the configs — not separately isolated.)

Repro: `nvfp4_spike/run_rebase_nvfp4.sh <boot|generate> <max_model_len> <tag>
<use_lora>` on the box (0.22 `trainers_mn` venv, harness from `/root/trainers`,
`_BASETEN_SERVED_MODEL=nemotronhforcausallm`).

---

## Logprob fidelity (900 completion tokens, 8 GSM8K-style prompts, 8k ctx)

Convention: behavior = NVFP4 (sampler); target = bf16. `w = π_target/π_behavior`.

| comparison | mean \|Δlogp\| | k3 KL | ESS/N | clip% (0.8–1.2) | verdict |
|---|---|---|---|---|---|
| **floor** — NVFP4 gen vs NVFP4 rescore (same model) | 0.0136 | 0.0009 | 0.998 | 1.3% | HEALTHY |
| **pure quant** — NVFP4 rescore vs bf16 rescore | 0.0190 | 0.0019 | 0.995 | 1.6% | HEALTHY |
| **RL-relevant** — NVFP4 gen vs bf16 rescore | 0.0188 | 0.0019 | 0.995 | 2.1% | HEALTHY |

- NVFP4 quantization adds only **~0.005 nats** of mean drift **above the
  generate-vs-rescore floor** — negligible.
- **ESS/N ≈ 0.995**: token-level importance-sampling correction discards
  almost nothing. `cispo` is comfortable.
- A handful of outlier tokens (max |Δlogp| ~0.8–0.9, w up to ~2.4) — expected,
  and exactly what `cispo` clipping is for.

---

## Caveats (do not over-read)

1. **vLLM-on-vLLM.** The bf16 side is the vLLM engine, **not the Megatron
   trainer**. These numbers prove the *quantization* is benign; they exclude the
   pre-existing **Megatron↔vLLM engine-mismatch** term (which exists today with
   or without NVFP4). The true production gap = this + that mismatch; measuring it
   needs trainer logprobs (the 4-node leg, deferred).
2. **Small sample**: 8 prompts / 900 tokens / GSM8K-ish / **8k context** (not
   131k). Directionally strong; re-confirm on more + longer sequences before a
   long RL run.
3. **6x ≠ pure weight savings** (see decomposition: ~3x weights × ~2x fp8 KV).
4. **MoE-LoRA-buffer confound**: both spike runs disabled those buffers, which
   alone frees ~10 GiB/GPU vs the committed golden config.

---

## Next steps

- [ ] Productionize the patch stack as a **gated** `_vllm_nemotron_patches.py`
      (fire only on NemotronH + NVFP4; must not touch the Super bf16 path),
      wired into `apply_patches`, with a unit test for the gate.
- [ ] Add the **NVFP4 Ultra sampler golden config** (`sampler_configs.py`):
      `moe_backend=cutlass`, `enable_flashinfer_autotune=false`, `kv_cache_dtype=fp8`,
      `disable_custom_all_reduce`, and retuned `max_num_seqs` / `max_loras`
      (the ~6x headroom lets these go well above the bf16-constrained 512 / 1).
- [ ] (Optional) bf16 **+ fp8 KV** run to isolate the pure weight-savings multiple (~3x).
- [ ] (4-node) Megatron bf16 **trainer logprobs** vs NVFP4 sampler for the true
      train/infer gap.
- [ ] Larger/longer eval (more prompts, 131k context) before a production RL run.

## Reproduce

```bash
# on the devbox, venv = trainers_mn/sampler/.venv, cwd = tests/comparison_studies/nvfp4_compare
# generate (NVFP4 + LoRA), rescore (NVFP4 and bf16), then compare — see README.md.
python compare.py --behavior out/rescore_nvfp4.json --target out/rescore_bf16.json   # pure quant
python compare.py --behavior out/gen_nvfp4.json     --target out/rescore_bf16.json   # RL-relevant
```
