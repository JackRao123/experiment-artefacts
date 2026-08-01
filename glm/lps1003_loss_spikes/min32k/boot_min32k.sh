#!/usr/bin/env bash
# LPS-1003 min32k: golden cp16/ep16 config at max_seq_len 32k, prod-faithful
# env (conn UNSET) + BT_SKIP_FULL_WARMUP=1. Dispatch only; probe runs separately.
set -uo pipefail
LPS=/root/.cache/user_artifacts/lps1003
DU=/root/.cache/user_artifacts/.devbox_up
source /root/.cache/user_artifacts/env.sh
export BT_SKIP_FULL_WARMUP=1
export BT_TRAINER_CONFIG_PATH=$LPS/min32k/trainer-config.cp16s32k.json
export BT_TRAINER_SERVER_CONFIG_PATH=$LPS/trainer-server-config.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NUM_NODES=2
export MASTER_PORT=$((29500 + RANDOM % 1000))
printf "2\n" > "$DU/trainer_num_nodes"
nohup srun --job-name=devbox_trainer --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  --gres=gpu:8 --cpus-per-task=248 --export=ALL \
  bash "$LPS/parity/run_trainer_node_prodenv.sh" > "$DU/trainer_srun.log" 2>&1 < /dev/null &
echo "dispatched srun pid $! master_port $MASTER_PORT"
