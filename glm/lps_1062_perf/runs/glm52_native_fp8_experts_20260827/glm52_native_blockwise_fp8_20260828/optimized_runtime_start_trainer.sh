#!/usr/bin/env bash
set -euo pipefail

: "${BT_TRAINER_CONFIG_PATH:?set BT_TRAINER_CONFIG_PATH}"
RUN_DIR=/root/.cache/user_artifacts/lps1062_native_blockwise_20260828/phase2
export BT_TRAINER_CONFIG_PATH MASTER_PORT="${MASTER_PORT:-29722}"
export LPS1062_SRC="${LPS1062_SRC:-/root/.cache/user_artifacts/devboxes/w5y89m3/trainers-compiled-runtime}"
job_name="${LPS1062_JOB_NAME:-glm52_compiled_debug}"
log_path="$RUN_DIR/${job_name}.log"
printf '%s\n' "$job_name" > "$RUN_DIR/optimized_runtime_active_job_name"
printf '%s\n' "$log_path" > "$RUN_DIR/optimized_runtime_active_log_path"
nohup srun --job-name="$job_name" --nodelist=b300-1-s58nc356-0011 \
  --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:8 --export=ALL \
  bash "$RUN_DIR/optimized_runtime_run_trainer_node.sh" \
  > "$log_path" 2>&1 < /dev/null &
echo "trainer dispatched; srun pid $!"
echo "wait on node-1: bash $RUN_DIR/optimized_runtime_wait_trainer_health.sh"
echo "stop: bash $RUN_DIR/optimized_runtime_stop_trainer.sh"
