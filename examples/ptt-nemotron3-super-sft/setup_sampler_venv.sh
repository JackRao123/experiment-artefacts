#!/usr/bin/env bash
# Build the vLLM inference-sampler venv on a devbox node (the box has no
# sampler venv by default; only the trainer venv exists).
#
# WHY THIS SCRIPT EXISTS — the install ORDER matters:
#   `uv sync` in sampler/ installs the lightweight deps (baseten-weight-sync,
#   loops-models, fastapi==0.115.14, starlette==0.46.2, …) but NOT vLLM/torch.
#   vLLM + torch ship as commit-pinned wheels (see sampler/vllm_stack.env), the
#   same way the sampler Dockerfile installs them. Installing the vLLM wheel
#   AFTER `uv sync` *upgrades* fastapi -> 0.137.x and starlette -> 1.3.x
#   (vLLM's unpinned web stack), which silently clobbers sampler's pins. The
#   result: every HTTP request to the booted server 500s with
#   `'_IncludedRouter' object has no attribute 'path'` (Starlette 1.x changed
#   include_router internals that vLLM's route iteration relies on; fastapi
#   0.137 also breaks prometheus-fastapi-instrumentator — see the comment in
#   sampler/pyproject.toml). The Dockerfile avoids this by installing the
#   editable sampler (which carries the pins) LAST. This script reproduces that
#   ordering by RE-PINNING fastapi/starlette as the final step.
#
# Usage (run on the node that will host the sampler):
#   bash setup_sampler_venv.sh
set -euo pipefail

SAMPLER_DIR="${SAMPLER_DIR:-/b10/workspace/baseten/trainers/sampler}"
UV="${UV:-/root/.local/bin/uv}"   # uv is not on PATH in non-interactive shells

cd "$SAMPLER_DIR"

echo "[setup] uv sync (lightweight deps + editable sampler/baseten-weight-sync/loops-models)"
"$UV" sync

echo "[setup] installing pinned vLLM + torch stack from vllm_stack.env"
# shellcheck disable=SC1091
. ./vllm_stack.env
"$UV" pip install "$VLLM_WHEEL"
"$UV" pip uninstall torch torchvision torchaudio -y
"$UV" pip install \
  "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchaudio==$TORCH_VERSION" \
  --no-cache --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
"$UV" pip install --no-cache "transformers==$TRANSFORMERS_VERSION"

echo "[setup] RE-PINNING fastapi/starlette (the vLLM wheel install bumped them off-pin)"
"$UV" pip install "fastapi==0.115.14" "starlette==0.46.2"

echo "[setup] verifying imports"
.venv/bin/python - <<'PY'
import vllm, torch, fastapi, starlette
print("vllm", vllm.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("fastapi", fastapi.__version__, "starlette", starlette.__version__)
assert fastapi.__version__ == "0.115.14", "fastapi pin clobbered — server will 500"
assert starlette.__version__ == "0.46.2", "starlette pin clobbered — server will 500"
print("OK: sampler venv ready")
PY
