#!/usr/bin/env bash
# Per-node launcher for the GLM-5.2 profiling trainer on devbox qr9oevq.
# Run once per node with the node rank as $1 (dispatched by trainer_ctl.sh).
#
# Env knobs:
#   TRAINERS_SRC   trainers checkout root   (default /root/.cache/user_artifacts/trainers_glm)
#   CONFIG         trainer config json      (default /root/.cache/user_artifacts/glm_prof/configs/active.json)
#   NUM_NODES      node count               (default 4)
#   NUM_GPUS       gpus per node            (default 8)
#   LEADER_ADDR    rank-0 reachable addr    (default $BT_LEADER_ADDR)
#   MASTER_PORT    torchrun rendezvous port (default 29500)
#   HF_HOME        HF cache root            (default /root/.cache/team_artifacts/huggingface)
set -uo pipefail

NODE_RANK="${1:?usage: run_trainer_node.sh <node_rank>}"
TRAINERS_SRC="${TRAINERS_SRC:-/root/.cache/user_artifacts/trainers_glm}"
CONFIG="${CONFIG:-/root/.cache/user_artifacts/glm_prof/configs/active.json}"

export NUM_NODES="${NUM_NODES:-4}"
export NUM_GPUS="${NUM_GPUS:-8}"
export BT_NODE_RANK="$NODE_RANK"
export BT_LEADER_ADDR="${LEADER_ADDR:-${BT_LEADER_ADDR:?set LEADER_ADDR or BT_LEADER_ADDR}}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export PORT="${PORT:-8000}"
export NVTE_DEBUG="${NVTE_DEBUG:-0}"
export HF_HOME="${HF_HOME:-/root/.cache/team_artifacts/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TORCHRUN="${TORCHRUN:-$TRAINERS_SRC/server/.venv/bin/torchrun}"
export PYTHONPATH="$TRAINERS_SRC/server/src:${PYTHONPATH:-}"

# NCCL / comms env (mirrors dev_job/slurm_workstation configure_remote.sh —
# without these the first cross-node collective times out on this box).
export NVTE_FRAMEWORK=pytorch
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MEGATRON_SKIP_GLOO_GROUPS=1
export GLOO_SOCKET_IFNAME=eth0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=""
if command -v ibv_devinfo >/dev/null 2>&1; then
  IB_HCA="$(ibv_devinfo 2>/dev/null | sed -n -e '/hca_id/p' -e '/link_layer:/p' \
    | grep -B1 InfiniBand | grep hca_id | sed -e 's/^hca_id://g' \
    | tr -d '[:blank:]' | paste -sd, || true)"
  if [[ -n "$IB_HCA" ]]; then
    export NCCL_IB_HCA="$IB_HCA"
    echo "[run_trainer_node] NCCL_IB_HCA=$NCCL_IB_HCA"
  fi
fi
# venv on PATH so Megatron's runtime pybind11 helpers_cpp build resolves the
# venv interpreter (has pybind11), not system python3.
export PATH="$TRAINERS_SRC/server/.venv/bin:$PATH"
export BT_TRAINER_CONFIG_PATH="$CONFIG"

cd "$TRAINERS_SRC/server"
echo "[run_trainer_node] rank=$BT_NODE_RANK nnodes=$NUM_NODES leader=$BT_LEADER_ADDR config=$CONFIG src=$TRAINERS_SRC"
echo "[run_trainer_node] config contents:"; cat "$CONFIG"
exec bash scripts/launch.sh
