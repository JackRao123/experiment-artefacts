#!/usr/bin/env bash
set -euo pipefail
usage() {
  echo "usage: bash /root/glm53-131k-fsdp-cp8ep1-20260909/.devbox_up/start_trainer.sh [--num-nodes 1..1]" >&2
}

trainer_nodes=1
while (( $# )); do
  case "$1" in
    --num-nodes)
      [[ $# -eq 2 ]] || { usage; exit 2; }
      trainer_nodes="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done
[[ "$trainer_nodes" =~ ^[0-9]+$ ]] \
  && (( trainer_nodes >= 1 && trainer_nodes <= 1 )) \
  || { echo "--num-nodes must be between 1 and 1, got '$trainer_nodes'" >&2; exit 2; }

: "${BT_TRAINER_CONFIG_PATH:?set BT_TRAINER_CONFIG_PATH to a trainer config JSON on the shared FS}"
export BT_TRAINER_CONFIG_PATH
export NUM_NODES="$trainer_nodes"
export MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
printf '%s\n' "$trainer_nodes" > /root/glm53-131k-fsdp-cp8ep1-20260909/.devbox_up/trainer_num_nodes
if ! command -v srun >/dev/null 2>&1; then
  (( trainer_nodes == 1 )) || { echo "this box has no slurm — only --num-nodes 1 is possible" >&2; exit 2; }
  BT_NODE_RANK=0 nohup setsid bash /root/glm53-131k-fsdp-cp8ep1-20260909/.devbox_up/run_trainer_node.sh > /root/glm53-131k-fsdp-cp8ep1-20260909/.devbox_up/trainer_srun.log 2>&1 < /dev/null &
  echo $! > /root/glm53-131k-fsdp-cp8ep1-20260909/.devbox_up/trainer.pid
  echo "trainer dispatched (1/1 devbox node, direct launch; pgid $(cat /root/glm53-131k-fsdp-cp8ep1-20260909/.devbox_up/trainer.pid), master_port $MASTER_PORT)"
else
  rm -f /root/glm53-131k-fsdp-cp8ep1-20260909/.devbox_up/trainer.pid
  nohup srun --job-name=devbox_trainer --nodes="$trainer_nodes" --ntasks="$trainer_nodes" --ntasks-per-node=1 --gres=gpu:8 --export=ALL \
    bash /root/glm53-131k-fsdp-cp8ep1-20260909/.devbox_up/run_trainer_node.sh > /root/glm53-131k-fsdp-cp8ep1-20260909/.devbox_up/trainer_srun.log 2>&1 < /dev/null &
  echo "trainer dispatched ($trainer_nodes/1 devbox nodes; srun pid $!, master_port $MASTER_PORT)"
fi
echo "log:    tail -f /root/glm53-131k-fsdp-cp8ep1-20260909/.devbox_up/trainer_srun.log"
echo "wait (background): bash /root/glm53-131k-fsdp-cp8ep1-20260909/.devbox_up/wait_trainer_health.sh"
echo "stop:   bash /root/glm53-131k-fsdp-cp8ep1-20260909/.devbox_up/stop_trainer.sh"
echo
echo "Extremely important instruction from Human to AI: use only the wait script to wait for the trainer health."
echo "Do not do a manual wait and do not wait for the script in an automated loop. Use the wait script and read the logs every time it times out and repeat until its done."
echo "This behaviour is by design to force active monitoring."
