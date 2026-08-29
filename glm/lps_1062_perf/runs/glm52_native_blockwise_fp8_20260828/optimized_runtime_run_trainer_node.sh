#!/usr/bin/env bash
set -euo pipefail

source /root/.cache/user_artifacts/devboxes/w5y89m3/env.sh
SRC="${LPS1062_SRC:-/root/.cache/user_artifacts/devboxes/w5y89m3/trainers-compiled-runtime}"
RUN_DIR=/root/.cache/user_artifacts/lps1062_native_blockwise_20260828/phase2
export NUM_NODES=1 NUM_GPUS=8 BT_NODE_RANK=0
export BT_LEADER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29722}"
export TORCHRUN=/root/.devbox-venvs/server/bin/torchrun
export PATH=/root/.devbox-venvs/server/bin:$PATH
export PYTHONPATH="$SRC/models/src:$SRC/server-interface/src:$SRC/server-megatron-bridge/src:$SRC/server-megatron-bridge/vendor/megatron-bridge/src:$SRC/server-megatron-bridge/vendor/megatron-bridge/3rdparty/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
: "${BT_TRAINER_CONFIG_PATH:?set BT_TRAINER_CONFIG_PATH}"
export BT_TRAINER_SERVER_CONFIG_PATH="$RUN_DIR/optimized_runtime_server_config.json"
export USE_HF=1 PORT=8001 PYTHONUNBUFFERED=1 GLOO_SOCKET_IFNAME=eth0
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.95}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WEIGHT_SYNC_AWS_CREDENTIALS_DIR="${WEIGHT_SYNC_AWS_CREDENTIALS_DIR:-/aws-secrets}"
unset S3_MANIFEST_PATH

cd "$SRC/server-main"
exec "$TORCHRUN" \
  --nnodes=1 \
  --nproc_per_node="$NUM_GPUS" \
  --node_rank=0 \
  --master_addr="$BT_LEADER_ADDR" \
  --master_port="$MASTER_PORT" \
  -m trainers_server_main.main \
  --backend megatron_bridge
