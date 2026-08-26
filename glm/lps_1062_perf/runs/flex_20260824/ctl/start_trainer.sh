#!/usr/bin/env bash
set -euo pipefail
CTL=/root/.cache/user_artifacts/lps1062_flex_20260824/ctl
trainer_nodes=2
if [ "${1:-}" = "--num-nodes" ]; then
  trainer_nodes="${2:?missing node count}"
fi
[[ "$trainer_nodes" =~ ^[12]$ ]] || { echo "--num-nodes must be 1 or 2" >&2; exit 2; }
: "${BT_TRAINER_CONFIG_PATH:?set BT_TRAINER_CONFIG_PATH}"
export BT_TRAINER_CONFIG_PATH NUM_NODES="$trainer_nodes"
export MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
printf '%s\n' "$trainer_nodes" > "$CTL/trainer_num_nodes"
rm -f "$CTL/trainer.pid"
node_args=()
if [ "$trainer_nodes" = "1" ]; then
  node_args=(--nodelist="$(hostname -s)")
fi
nohup srun --job-name=devbox_trainer "${node_args[@]}" --nodes="$trainer_nodes" --ntasks="$trainer_nodes" \
  --ntasks-per-node=1 --gres=gpu:8 --export=ALL bash "$CTL/run_trainer_node.sh" \
  > "$CTL/trainer_srun.log" 2>&1 < /dev/null &
echo "trainer dispatched ($trainer_nodes nodes; srun pid $!, master_port $MASTER_PORT)"
echo "wait: bash $CTL/wait_trainer_health.sh"
echo "stop: bash $CTL/stop_trainer.sh"
