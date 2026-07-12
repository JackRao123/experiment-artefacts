# GLM-5.2 sampler-checkpoint standalone deploy failure — repro + fix (2026-07-10)

## TL;DR

`truss_train deploy_checkpoints` of the pirate-SFT sampler checkpoint
(`VBnwM20`, session `4w7y2yw` / run `4w79g63`) failed at boot because the
generic checkpoint-deploy path in billip serves everything on the org-wide
`INFERENCE_TEMPLATES_VLLM_IMAGE_URI` = `vllm/vllm-openai:v0.22.0`, and that
image cannot serve GLM-5.2's DSA sparse MLA (head_size=704) on B200 — every
worker dies in attention-backend selection and the deployment crashloops to
`DEPLOY_FAILED`.

Fix (billip PR): per-base-model serving requirements for deployed
checkpoints — `zai-org/GLM-5.2-FP8` → image `vllm/vllm-openai:v0.24.0`
(+ tool parser `glm47`), following the existing per-base-model tool-call
parser pattern. v0.24.0 is the same vLLM the Loops GLM sampler stack runs in
prod (trainers#594) and the version Ian/Aaron validated serving these exact
weights standalone on B200:8 (2026-07-08 evals).

## The failing deployment

- Model `glm-5-2-pirate-sft` (`qrj2x403`), deployment `qrp8xe1`,
  created 2026-07-10T03:19Z via `random/deploy_glm_pirate_lora.py`
  (loops-quickstart), B200:8. Status: `DEPLOY_FAILED` (crashloop).
- Full logs: `data/pirate-sft-deploy-qrp8xe1.jsonl` (fetched with
  `scripts/fetch_model_deploy_logs.py`, the management-API cousin of
  loops-quickstart's `fetch_loops_logs.py`).
- Rendered start command (from the deployment config API — this is exactly
  billip's `VLLM_START_COMMAND` template):

  ```
  vllm serve /app/models/zai-org/GLM-5.2-FP8 --served-model-name baseten-model \
    --chat-template /app/models/zai-org/GLM-5.2-FP8/chat_template.jinja \
    --port 8000 --tensor-parallel-size 8 --dtype bfloat16 \
    --enable-lora --enable-mixed-moe-lora-format --max-lora-rank 16 \
    --lora-modules pirate-sft-final=/tmp/training_checkpoints/VBnwM20
  ```

- Failure (every TP worker, 8/8, repeated across restarts):

  ```
  ValueError: No valid attention backend found for cuda with
  AttentionSelectorConfig(head_size=704, dtype=torch.bfloat16,
  kv_cache_dtype=auto, block_size=None, use_mla=True, has_sink=False,
  use_sparse=True, ...)
  ```

  raised in `vllm/platforms/cuda.py::get_attn_backend_cls` (engine v0.22.0)
  during model construction, before weights load.

## Root cause

GLM-5.2 uses DeepSeek-style sparse MLA (DSA): `use_mla=True, use_sparse=True`,
head_size 704 (kv_lora_rank 640 + rope 64). On SM100 (B200), vLLM 0.22's
selector considers exactly two sparse-MLA backends: `FLASHINFER_MLA_SPARSE`
and `FLASHMLA_SPARSE`. In the frozen `vllm/vllm-openai:v0.22.0` image both
fail their support checks, so selection raises. The same vLLM 0.22.0 as a
PyPI wheel on the devbox (which resolves the current flashinfer
0.6.11 + flashinfer-cubin) *does* select `FLASHINFER_MLA_SPARSE` — the
support surface depends on the flashinfer build frozen into the serving
image, i.e. the 0.22 image is simply too old for GLM-5.2 on Blackwell.
Upstream vLLM declares GLM-5.2 support from 0.23 (model class
`GlmMoeDsaForCausalLM`), and the vLLM recipe recommends 0.23+.

Why the Loops sampler works: the sampler registry pins GLM_5_2_FP8/B200 to
the `vllm-0.24` stack (trainers#594 — 0.23 adds the model class + mixed MoE
LoRA; 0.24 fixes long-prefill "token soup" at >=18k-token prompts, verified
clean at 80k). The generic deploy path knows nothing about any of this.

## Repro / validation on 1-node B200 devbox (`wnmydy3`, Birch/Weka)

Setup: two uv venvs mirroring `sampler/Dockerfile` + `sampler/stacks/*.env`
(cu129 wheels, torch 2.11.0): vLLM 0.22.0 / 0.24.0. Base weights from the
shared-NFS HF cache (`models--zai-org--GLM-5.2-FP8`, 704 GB). LoRA checkpoint
VBnwM20 (15.4 GB) fetched via `GET /v1/loops/checkpoints/VBnwM20/files`
presigned URLs. Exact billip-rendered command shape (above), paths swapped.

| Stack | Outcome |
|---|---|
| prod `vllm/vllm-openai:v0.22.0` image (deployment qrp8xe1) | `No valid attention backend` → crashloop (log evidence) |
| devbox vLLM 0.22.0 wheel + flashinfer 0.6.11 | selects `FLASHINFER_MLA_SPARSE`; diagnostic only (proves the backend gap is image-build-dependent) |
| devbox vLLM 0.24.0, billip-exact command (image bump only) | passes backend selection, then **fails the KV-capacity check**: GLM's native `max_seq_len` 1,048,576 needs 90.46 GiB MLA KV/GPU vs 46.25 GiB available (estimated max length 536,064) → the image bump alone is NOT enough |
| devbox vLLM 0.24.0 + `--max-model-len 262144` + `--tool-call-parser glm47` (the FIXED rendered command) | **SERVER_READY** — KV cache 613,248 tokens (2.34x concurrency @262k); `/v1/models` lists `baseten-model` + `pirate-sft-final`; LoRA answers in pirate speak, base answers normally (eval outputs in `data/devbox-wnmydy3-repro-logs.tgz`) |
| prod `vllm/vllm-openai:v0.24.0` image, same command shape + `--max-model-len 262144`, GLM LoRA (Aaron/Ian evals `wd107e4w`/`wnpde6y3`, 2026-07-08) | served (models later deactivated after evals; the failed pirate deploy is the only `DEPLOY_FAILED`) |

Devbox-only stumbles (not relevant to prod images, which ship full toolchains):
bare 1-node boxes lack `curl`/`python3.12-dev`/`ninja` because `devbox-up` N=1 is
broken (see memory note devbox-up-single-node-pitfalls); `apt-get install`ing
them unblocked Triton JIT during vLLM boot.

## The productionisable fix (billip)

Branch `jack/glm52-deploy-checkpoints` (worktree `baseten-wt-glm-deploy`):

- `training/checkpoint_serving.py`
  - `ServingRequirements` dataclass + `_BASE_MODEL_TO_SERVING_REQUIREMENTS`
    map keyed on exact HF repo id (same keying as the tool-parser map):
    `zai-org/GLM-5.2-FP8` → `image_uri="vllm/vllm-openai:v0.24.0"` +
    `extra_serve_args=("--max-model-len", "262144")` (required — see
    validation table; 262144 = Loops sampler SeqLen.S256K).
  - `GlmMoeDsaForCausalLM` added to the vLLM backend's supported
    architectures (deployability validation).
  - Tool parser map row: `zai-org/GLM-5.2-FP8` → `glm47` (registered in the
    0.24 build; per upstream GLM recipe).
- `oracles/inference_templates/inference_template_params.py` — `BaseModel`
  gains `serving_image_uri` / `extra_serve_args`, threaded from
  `checkpoint.base_model` in all three constructors (FULL, Loops, and the
  HF base synthesized for LoRA deploys).
- `oracles/inference_templates/inference_stacks/vllm.py` — base image
  override + `{{ extra_args_str }}` in both serve branches.

Deliberately NOT bumping the org-wide `INFERENCE_TEMPLATES_VLLM_IMAGE_URI`:
every other base model keeps the validated default image; GLM opts into the
image its architecture requires.

## Outcome

- Devbox `wnmydy3` killed after validation (2026-07-10).
- billip PR: branch `jack/glm52-deploy-checkpoints`, commit
  `feat(django): per-base-model vLLM serving requirements for deployed
  checkpoints (GLM-5.2)` — 107 backend tests pass (unit coverage at map,
  command-emission, and constructor-threading layers).
- The `sampler-side load of a CP-exported adapter (pairing smoke)` sub-item in
  `productionise.md` P0-3 is now effectively covered for the standalone-deploy
  path: VBnwM20 (exported by the CP trainer image `jackrao-glm-131k-cp-7c8c093`)
  loads and serves through vLLM 0.24's LoRA path.

## Open items

- Redeploy `glm-5-2-pirate-sft` through `deploy_checkpoints` once the billip
  PR merges (end-to-end confirmation on prod infra + run the eval prompts).
- Longer-term: route standalone Loops-checkpoint deploys through the sampler
  registry recipe (stack/image + full GLM arg set from
  `loops_models.sampler_configs`) instead of a hand-maintained billip map, so
  trainers stays the single source of truth for per-model serving.
