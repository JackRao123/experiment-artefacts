#!/usr/bin/env bash
# Launch the vLLM inference sampler for Nemotron-3-Super-120B with LoRA enabled
# on a single 8x B200 node. Mirrors run_trainer_node.sh but for the sampler.
#
# Prereq: the sampler venv must exist with the correct deps — run
# setup_sampler_venv.sh first (it encodes the fastapi/starlette pin fix).
#
# Env knobs (defaults are the validated values from SAMPLER_REPORT.md):
#   SAMPLER_DIR     sampler checkout         (default /b10/workspace/baseten/trainers/sampler)
#   MODEL_ID        HF repo id               (default nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16)
#   MAX_SEQ_LENGTH  vLLM --max-model-len     (default 16384; fits up to 262144 on 8x B200)
#   HF_HOME         HF cache root            (default /root/.cache/user_artifacts/huggingface)
#   PORT            server port              (default 8001)
set -uo pipefail

SAMPLER_DIR="${SAMPLER_DIR:-/b10/workspace/baseten/trainers/sampler}"
cd "$SAMPLER_DIR"

export HF_HOME="${HF_HOME:-/root/.cache/user_artifacts/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-16384}"
# LoRA. enable_lora_from_env() keys off ENABLE_LORA; the server then auto-sets
# VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 so /v1/load_lora_adapter works at runtime.
export ENABLE_LORA="${ENABLE_LORA:-true}"
export MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
export MAX_LORAS="${MAX_LORAS:-4}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8001}"
# NemotronH ships custom configuration_nemotron_h.py — vLLM needs remote code to
# load the config. MODEL_ID is a repo id (not a local path), so the server's
# config.json sniff (infer_model_ref_from_hf_config) can't auto-enable it; pass
# it explicitly via the escape hatch.
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---trust-remote-code}"

echo "[run_sampler_node] model=$MODEL_ID tp=$TENSOR_PARALLEL_SIZE max_model_len=$MAX_SEQ_LENGTH lora=$ENABLE_LORA port=$PORT"
exec .venv/bin/python -m sampler.vllm_server
