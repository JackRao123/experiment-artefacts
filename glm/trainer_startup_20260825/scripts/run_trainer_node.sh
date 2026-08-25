#!/usr/bin/env bash
set -euo pipefail

WT=/root/.cache/user_artifacts/devboxes/q4grmdq/trainers-startup
export PYTHONPATH="$WT/models/src:$WT/server-main/src:$WT/server-interface/src:$WT/server-megatron-bridge/src"

if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
  logical_cpus="$(getconf _NPROCESSORS_ONLN)"
  threads_per_rank=$(( logical_cpus / (BT_NUM_GPUS * 2) ))
  (( threads_per_rank < 1 )) && threads_per_rank=1
  (( threads_per_rank > 16 )) && threads_per_rank=16
  export OMP_NUM_THREADS="$threads_per_rank"
fi

exec bash /root/.cache/user_artifacts/devboxes/q4grmdq/.devbox_up/run_trainer_node.generated.sh
