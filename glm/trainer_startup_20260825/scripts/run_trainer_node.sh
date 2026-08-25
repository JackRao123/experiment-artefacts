#!/usr/bin/env bash
set -euo pipefail

WT=/root/.cache/user_artifacts/devboxes/q4grmdq/trainers-startup
export PYTHONPATH="$WT/models/src:$WT/server-main/src:$WT/server-interface/src:$WT/server-megatron-bridge/src"

exec bash /root/.cache/user_artifacts/devboxes/q4grmdq/.devbox_up/run_trainer_node.generated.sh
