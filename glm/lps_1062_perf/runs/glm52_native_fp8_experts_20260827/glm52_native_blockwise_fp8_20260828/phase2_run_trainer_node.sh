#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=/dev/null
source /root/.cache/user_artifacts/env.sh
SRC=/root/.cache/user_artifacts/devboxes/w5y89m3/trainers-native-fp8-final
export NUM_NODES=1 NUM_GPUS=8 BT_NODE_RANK=0
export MASTER_PORT="${MASTER_PORT:-29652}"
export TORCHRUN=/root/.devbox-venvs/server/bin/torchrun
export PATH=/root/.devbox-venvs/server/bin:$PATH
export PYTHONPATH="$SRC/models/src:$SRC/server-interface/src:$SRC/server-megatron-bridge/src:$SRC/server-megatron-bridge/vendor/megatron-bridge/src:$SRC/server-megatron-bridge/vendor/megatron-bridge/3rdparty/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
: "${BT_TRAINER_CONFIG_PATH:?set BT_TRAINER_CONFIG_PATH to a trainer config JSON}"
export BT_TRAINER_SERVER_CONFIG_PATH=/root/.cache/user_artifacts/lps1062_native_blockwise_20260828/phase2/server_config.json
export USE_HF=1 PORT=8001 PYTHONUNBUFFERED=1
export GLOO_SOCKET_IFNAME=eth0
unset S3_MANIFEST_PATH

cd "$SRC/server-main"
exec bash scripts/launch.sh --backend megatron_bridge
