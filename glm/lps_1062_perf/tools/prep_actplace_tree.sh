#!/usr/bin/env bash
# Tree prep for LPS-1062 activation-placement ladder (conway, 2026-08-20).
# Checkout is already at the pin; this does submodules + the megatron-bridge venv.
set -euo pipefail
exec 2>&1
source /root/.cache/user_artifacts/env.sh
export PATH=/root/.cache/user_artifacts/bin:$PATH
export UV_PYTHON=/usr/bin/python3.12
# CUDA 13 toolkit. The cu13 torch wheels make source builds compile against
# torch.version.cuda == 13.0, and this image ships only 12.8; the mismatch
# aborts the build (hit on fast-hadamard-transform, whose wheel 404s for this
# combination so it falls back to source). Install on BOTH nodes:
#   apt-get install -y cuda-nvcc-13-0 cuda-libraries-dev-13-0
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
C=/root/.cache/user_artifacts/trainers_main
ts() { date -u +%FT%TZ; }
cd "$C"
PIN=${PIN:-7eec3054}
echo "[$(ts)] fetch + checkout $PIN"
git fetch origin jackrao/lps-1062-actplace
git checkout --quiet "$PIN"
echo "[$(ts)] pin: $(git log --oneline -1)"
echo "[$(ts)] submodule sync"
git submodule sync --recursive
echo "[$(ts)] submodule update --init --recursive"
time git submodule update --init --recursive --progress
echo "[$(ts)] submodule status:"
git submodule status --recursive
echo "[$(ts)] make megatron-bridge-venv CUDA_FLAVOR=cu13"
time make megatron-bridge-venv CUDA_FLAVOR=cu13
echo "[$(ts)] pybind11"
uv pip install --python "$C/server-megatron-bridge/.venv/bin/python" pybind11
# NO cudnn-frontend "bump" here. The runbook's
# `uv pip install --no-deps nvidia-cudnn-frontend==1.27.0` is a REPAIR for a
# venv that resolved the wrong build; on a healthy tree it REPLACES the
# vendored pin (1.27.0.dev20260803+git7478516, built 0cu130) with the stock
# PyPI wheel, which does not even ship the module. `uv sync` already installs
# the vendored wheel — leave it alone.
echo "[$(ts)] verify"
cd "$C/server-megatron-bridge"
# The import name is `cudnn`, NOT `cudnn_frontend` (the runbook's verify line
# was wrong and would fail on any box).
uv run --no-sync python -c "import torch, cudnn, transformer_engine, importlib.metadata as m; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda); print(\"cudnn-frontend\", m.version(\"nvidia-cudnn-frontend\")); print(\"TE\", m.version(\"transformer-engine\")); from megatron.core.pipeline_parallel import fine_grained_activation_offload as fo; print(\"offload module OK:\", hasattr(fo, \"OffloadTensorGroup\"))"
echo "[$(ts)] DONE ok"
