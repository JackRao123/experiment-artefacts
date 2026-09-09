#!/usr/bin/env bash
set -euo pipefail

: "${BOX:?set BOX to /root/.cache/user_artifacts/devboxes/<job-id>}"
: "${RUN:?set RUN to the shared output directory}"
: "${BT_TRAINER_CONFIG_PATH:?set BT_TRAINER_CONFIG_PATH}"
: "${NSYS_SESSION_NAME:?set NSYS_SESSION_NAME}"

SRC="$BOX/trainers"
source "$BOX/env.sh"

export NUM_NODES=1 NUM_GPUS=8 BT_NODE_RANK=0
export MASTER_PORT="${MASTER_PORT:-29500}"
export TORCHRUN="$SRC/server-megatron-bridge/.venv/bin/torchrun"
export PATH="$SRC/server-megatron-bridge/.venv/bin:$PATH"
export USE_HF=1 PORT=8001 PYTHONUNBUFFERED=1 GLOO_SOCKET_IFNAME=eth0
unset S3_MANIFEST_PATH

mkdir -p "$RUN"
cd "$SRC/server-main"

exec nsys launch \
  --session-new="$NSYS_SESSION_NAME" \
  --trace=cuda,nvtx,cublas,cudnn,osrt \
  --cudabacktrace=sync:1000 \
  --python-backtrace=cuda \
  --pytorch=autograd-nvtx \
  --cuda-event-trace=true \
  --cuda-memory-usage=true \
  --show-output=true \
  --wait=all \
  bash scripts/launch.sh --backend megatron_bridge
