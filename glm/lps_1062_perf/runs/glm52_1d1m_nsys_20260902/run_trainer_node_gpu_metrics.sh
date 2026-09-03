#!/usr/bin/env bash
set -euo pipefail
source /root/.cache/user_artifacts/devboxes/w7xgz63/env.sh
SRC=/root/.cache/user_artifacts/devboxes/w7xgz63/trainers
RUN=/root/.cache/user_artifacts/lps1062_bench/glm52_1d1m_nsys_20260902
export NUM_NODES="${NUM_NODES:-1}" NUM_GPUS=8
export BT_NODE_RANK="${SLURM_NODEID:-${BT_NODE_RANK:-0}}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export TORCHRUN="$SRC/server-megatron-bridge/.venv/bin/torchrun"
export PATH="$SRC/server-megatron-bridge/.venv/bin:$PATH"
: "${BT_TRAINER_CONFIG_PATH:?set BT_TRAINER_CONFIG_PATH}"
export USE_HF=1 PORT=8001 PYTHONUNBUFFERED=1
export GLOO_SOCKET_IFNAME=eth0
unset S3_MANIFEST_PATH
mkdir -p "$RUN"
cd "$SRC/server-main"
exec nsys launch \
  --session-new=glm1d1mmetrics \
  --trace=cuda,nvtx \
  --pytorch=autograd-nvtx \
  --show-output=true \
  --wait=all \
  bash scripts/launch.sh --backend megatron_bridge
