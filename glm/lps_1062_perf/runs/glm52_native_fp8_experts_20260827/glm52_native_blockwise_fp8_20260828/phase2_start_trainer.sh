#!/usr/bin/env bash
set -euo pipefail

: "${BT_TRAINER_CONFIG_PATH:?set BT_TRAINER_CONFIG_PATH to a trainer config JSON}"
RUN_DIR=/root/.cache/user_artifacts/lps1062_native_blockwise_20260828/phase2
export BT_TRAINER_CONFIG_PATH MASTER_PORT="${MASTER_PORT:-29652}"
nohup srun --job-name=glm52_native_phase2 --nodes=1 --ntasks=1 \
  --nodelist=b300-1-5x4eyifb-0008 \
  --ntasks-per-node=1 --gres=gpu:8 --export=ALL \
  bash "$RUN_DIR/phase2_run_trainer_node.sh" \
  > "$RUN_DIR/trainer_srun.log" 2>&1 < /dev/null &
echo "trainer dispatched; srun pid $!"
echo "wait: bash $RUN_DIR/phase2_wait_trainer_health.sh"
echo "stop: bash $RUN_DIR/phase2_stop_trainer.sh"
