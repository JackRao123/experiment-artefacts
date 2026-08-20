#!/usr/bin/env bash
# Pinned 2-node dispatch for the LPS-1062 activation-placement ladder.
# The devbox-up copy on this box was generated with a 1-node topology and
# refuses --num-nodes 2; this one is fixed at the box shape (2 x 8 B300).
set -euo pipefail
PP2=/root/.cache/user_artifacts/lps1062_pp2
: "${BT_TRAINER_CONFIG_PATH:?set BT_TRAINER_CONFIG_PATH first}"
export BT_TRAINER_CONFIG_PATH
export NUM_NODES=2
export MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
printf "2\n" > /root/.cache/user_artifacts/.devbox_up/trainer_num_nodes
mkdir -p $PP2/logs
# Snapshot the previous log before it is clobbered (campaign standing rule).
if [ -s "$PP2/logs/trainer_srun.log" ]; then
  mv "$PP2/logs/trainer_srun.log" "$PP2/logs/trainer_srun.$(date -u +%Y%m%dT%H%M%SZ).log"
fi
{
  echo "=== boot env $(date -u +%FT%TZ) ==="
  echo "config: $BT_TRAINER_CONFIG_PATH"
  env | grep -E "^(NCCL_|BT_|NVTE_)" | sort
} | tee -a $PP2/logs/boot_env.log
nohup srun --job-name=devbox_trainer --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  --gres=gpu:8 --export=ALL \
  bash $PP2/run_trainer_node_pp2.sh > $PP2/logs/trainer_srun.log 2>&1 < /dev/null &
echo "trainer dispatched (2 nodes; srun pid $!, master_port $MASTER_PORT)"
echo "log:  tail -f $PP2/logs/trainer_srun.log"
echo "wait: bash $PP2/wait_trainer_health_pp2.sh"
