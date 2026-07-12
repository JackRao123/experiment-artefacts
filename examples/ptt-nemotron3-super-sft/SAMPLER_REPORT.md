# Nemotron-3-Super-120B-A12B — vLLM LoRA serving report

Goal: verify the **NEMOTRON_3_SUPER LoRA serving path works end-to-end** —
train + save a LoRA adapter from the trainer, stand up the vLLM sampler on an
idle 8× B200 node, load the adapter, and confirm generation.

**Verdict: YES — NemotronH LoRA serving works on vLLM 0.22.0.** A LoRA trained
on the trainer (TP=8) was exported, loaded into the running sampler via the
runtime endpoint, and produced **distinctly different behavior** on neutral
prompts (see §4). One real fix was required to the sampler venv build (§2).

Hardware: single node, 8× NVIDIA B200 (179 GiB usable HBM/GPU). Model is a
hybrid Mamba2 + Transformer-MoE (`NemotronHForCausalLM`, A12B active).

---

## 1. Stack versions

From `sampler/vllm_stack.env` (the single source of truth shared with the
sampler Dockerfile):

| component    | version                      |
| ------------ | ---------------------------- |
| vLLM         | `0.22.0` (cu129 release wheel) |
| torch        | `2.11.0+cu129`               |
| transformers | `5.12.0`                     |
| fastapi      | `0.115.14` (pinned — see §2) |
| starlette    | `0.46.2`  (pinned — see §2)  |

Resolved engine args at launch (from the boot log):
`enable_lora=True, max_loras=4, max_lora_rank=64, enable_mixed_moe_lora_format=True,
trust_remote_code=True, max_model_len=…, tensor_parallel_size=8,
gdn_prefill_backend='triton'`.

Boot cost (8× B200, cold): model load ~56.5 GiB/GPU in ~93 s; full engine init
(torch.compile + CUDA-graph capture) ~405 s on the first cold start.

---

## 2. Sampler venv build fix (uv-sync / launch)

**The box has no sampler venv by default, and `uv sync` alone does NOT install
vLLM/torch** — those are commit-pinned wheels (`vllm_stack.env`), installed the
same way the Dockerfile does.

**Bug found:** installing the vLLM wheel *after* `uv sync` upgrades vLLM's
unpinned web stack — `fastapi 0.115.14 -> 0.137.2` and
`starlette 0.46.2 -> 1.3.1` — silently clobbering sampler's pins. The server
then boots fine ("Application startup complete") but **every HTTP request 500s**
with:

```
{"error":{"message":"'_IncludedRouter' object has no attribute 'path'",
          "type":"InternalServerError","code":500}}
```

Root cause: Starlette 1.x changed `include_router` internals (the
`_IncludedRouter` placeholder) that vLLM's per-request route iteration assumes
has a `.path`; fastapi 0.137 also breaks prometheus-fastapi-instrumentator
(already noted in `sampler/pyproject.toml`). The Dockerfile dodges this by
installing the editable sampler (which carries the pins) *last*.

**Fix:** re-pin as the final venv step:
`uv pip install "fastapi==0.115.14" "starlette==0.46.2"`. After re-pinning + a
sampler restart, `/v1/models`, `/v1/completions`, `/v1/chat/completions` all
return 200. This ordering is now encoded in `setup_sampler_venv.sh`.

---

## 3. Working launch command

Built the venv with `setup_sampler_venv.sh`, then launched with
`run_sampler_node.sh`. The effective command (LoRA enabled, validated):

```bash
cd /b10/workspace/baseten/trainers/sampler
HF_HOME=/root/.cache/user_artifacts/huggingface HF_HUB_OFFLINE=1 \
MODEL_ID=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
TENSOR_PARALLEL_SIZE=8 MAX_SEQ_LENGTH=16384 \
ENABLE_LORA=true MAX_LORA_RANK=64 MAX_LORAS=4 \
HOST=0.0.0.0 PORT=8001 VLLM_EXTRA_ARGS=--trust-remote-code \
.venv/bin/python -m sampler.vllm_server
```

Notes:
- `ENABLE_LORA=true` is what `enable_lora_from_env()` keys off; the server then
  auto-sets `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`, enabling runtime
  `/v1/load_lora_adapter`.
- `--trust-remote-code` is **required** for NemotronH and must be passed
  explicitly: `MODEL_ID` is a repo id, so the server's local-`config.json`
  sniff can't auto-enable it.
- `--enable-mixed-moe-lora-format` is feature-detected and auto-emitted by this
  vLLM build (it registers the flag), so per-expert MoE-LoRA layouts are
  accepted. (The pirate adapter here only touched `shared_experts`, not routed
  experts, so it didn't exercise that path — but the flag is on.)

---

## 4. LoRA viability — neutral-prompt evidence (the real test)

Trained a pirate LoRA on `winglian/pirate-ultrachat-10k` via `sft_driver.py`
(TP=8, rank 16, seq_len 8192, batch 8). Two adapters:

- `pirate-v1`: 20 steps — loss 1.79 → ~1.25. **Too weak**: neutral prompts came
  out indistinguishable from base. (Confirms the guidance: 20 steps is a no-op
  for behavior.)
- `pirate-v2`: 100 steps total (higher LR 3e-4) — loss → ~0.1. Hot-loaded into
  the **already-running** sampler via `POST /v1/load_lora_adapter` (no relaunch,
  no second cold start): `"Success: LoRA adapter 'pirate-v2' added successfully."`

**Adapter export (`/save_weights_for_sampler`) succeeded at TP=8.** On disk at
`…/sampler_weights/pirate-v2/`: `adapter_config.json` + `adapter_model.safetensors`
(384 tensors, ~75 MB), `r=16`, `lora_alpha=32`,
`target_modules=[down_proj, in_proj, k_proj, o_proj, out_proj, q_proj, up_proj, v_proj]`
(attention q/k/v/o, Mamba mixer in/out_proj, shared-expert MLP),
`base_model_class=NemotronHForCausalLM`. A `version.json` pointer is written at
the `sampler_weights/` root.

**Neutral prompts, temperature 0, zero pirate framing** (`/v1/chat/completions`).
The LoRA passes: neutral prompts come out distinctly pirate, base is normal.

### Prompt: "What should I do?"

**BASE:**
> Okay, the user just asked "What should I do?" without any context… *(reasons,
> then)* …That's a very common and important question—but to give you meaningful
> advice, I need a bit more context. "What should I do?" can apply to almost
> anything: a decision, a problem, a goal, a relationship, a career move…

**pirate-v2 (LoRA):**
> Ahoy, matey!  You've got a situation on your hands, and it's time to chart a
> course through the storm.  But first, tell me what's got you in a pickle? …
> spill the beans, and let's get this ship sailing smooth again! … I'm like your
> trusty first mate, always ready to lend a hand. … Now, go on, tell me what's
> got you in a pickle!  I'm all ears, matey!

### Prompt: "How do I bake a cake?"

**BASE:**
> Okay, the user is asking how to bake a cake. This seems like a beginner-level
> question… *(reasons, then)* …I'll assume a basic vanilla or yellow cake…
> *(normal recipe guidance)*

**pirate-v2 (LoRA):**
> Ahoy, matey!  Want to bake a cake, eh?  A fine choice for a scurvy dog like
> you!  Here's a simple recipe… **Ahoy, Matey's Cake Recipe** … 2 cups (250g)
> all-purpose flour … Preheat your oven to 350°F (175°C)…

Note: the base model is a *reasoning* model (emits a `<think>` trace before the
answer). The pirate LoRA both flips the answer style **and** suppresses the
visible reasoning trace — a strong, consistent behavioral shift on prompts with
no pirate cue.

---

## 5. Context-length wall (8× B200, this 120B model)

vLLM sizes the KV cache at launch, so each point is a sampler relaunch. At each,
confirmed boot (no KV-cache OOM) and a generation with a prompt of ~that length
(`ctx_test.py`, `max_tokens=16`, temp 0).

| max_model_len | GPU KV cache (tokens) | max concurrency | boots? | generation tested        | result |
| ------------- | --------------------- | --------------- | ------ | ------------------------ | ------ |
| 16,384        | 19,742,720            | 1205.00x        | yes    | 15,000-token prompt      | OK     |
| 65,536        | 24,035,565            | 366.75x         | yes    | 64,000-token prompt      | OK     |
| 131,072       | 25,128,091            | 191.71x         | yes    | 128,000-token prompt     | OK     |
| 196,608       | 25,514,677            | 129.77x         | yes    | 192,000-token prompt     | OK     |
| 262,144       | 25,712,465            |  98.09x         | yes    | 256,000-token prompt     | OK     |

**The wall is the model's configured maximum (262,144), NOT GPU memory.** This
is a hybrid Mamba model: Mamba state is constant per sequence and only a handful
of layers are attention, so KV cache is tiny and context scales almost for free.
Even at the full 262 k context, vLLM reports ~25.7 M tokens of KV capacity
(98× concurrency at full length) — orders of magnitude more than one full-length
request needs. (Per-GPU HBM shows ~172/183 GiB used at 262 k, but that is just
vLLM's default `gpu_memory_utilization=0.92` reservation, not actual KV need.)

So a single 8× B200 node serves this 120B model at its **full 262 k context**
with room to spare.

---

## 6. Files added / changed

All under `examples/ptt-nemotron3-super-sft/`:

- `SAMPLER_REPORT.md` (this file) — findings, evidence, context table.
- `setup_sampler_venv.sh` — builds the sampler venv with the **correct install
  order** and the fastapi/starlette re-pin fix from §2.
- `run_sampler_node.sh` — validated single-node sampler launcher (env in §3).

No application source was modified; the fix lives in the venv build script
(`setup_sampler_venv.sh`). The production image is unaffected — its Dockerfile
already installs the editable sampler last, so it never hits the pin clobber.
