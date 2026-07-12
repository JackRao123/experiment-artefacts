#!/usr/bin/env bash
# Per-node launcher for a multi-node Nemotron 3 Super trainer on a Baseten
# multinode devbox. Run once per node with the node rank as $1.
#
#   ssh training-job-<job>-0.ssh.baseten.co 'bash run_trainer_node.sh 0'
#   ssh training-job-<job>-1.ssh.baseten.co 'bash run_trainer_node.sh 1'
#
# Env knobs (with the values used for the 131k LoRA SFT work):
#   TRAINERS_SRC   trainers checkout root      (default /b10/workspace/baseten/trainers)
#   CONFIG         BT_TRAINER_CONFIG_PATH json (default /root/trainer_config.json)
#   NUM_NODES      node count                  (default 2)
#   NUM_GPUS       gpus per node               (default 8)
#   LEADER_ADDR    rank-0 reachable addr       (default $BT_LEADER_ADDR)
#   MASTER_PORT    torchrun rendezvous port    (default 29500)
#   HF_HOME        HF cache root               (default /root/.cache/user_artifacts/huggingface)
#   HF_HUB_OFFLINE use cached weights only     (default 1)
set -uo pipefail

NODE_RANK="${1:?usage: run_trainer_node.sh <node_rank>}"
TRAINERS_SRC="${TRAINERS_SRC:-/b10/workspace/baseten/trainers}"
CONFIG="${CONFIG:-/root/trainer_config.json}"

# Capture desired launch params BEFORE sourcing super_env.sh — that file
# exports NUM_NODES=1 (single-node default) and would otherwise clobber a
# multi-node override coming in from the environment.
_WANT_NUM_NODES="${NUM_NODES:-2}"
_WANT_NUM_GPUS="${NUM_GPUS:-8}"
_WANT_LEADER="${LEADER_ADDR:-}"

# Pull the baseten-injected env (HF token, BT_LEADER_ADDR, NCCL/EFA, etc.).
[[ -f /root/super_env.sh ]] && source /root/super_env.sh

export NUM_NODES="$_WANT_NUM_NODES"
export NUM_GPUS="$_WANT_NUM_GPUS"
export BT_NODE_RANK="$NODE_RANK"
export BT_LEADER_ADDR="${_WANT_LEADER:-${BT_LEADER_ADDR:?set LEADER_ADDR or BT_LEADER_ADDR}}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export PORT="${PORT:-8000}"
export NVTE_DEBUG="${NVTE_DEBUG:-0}"
export HF_HOME="${HF_HOME:-/root/.cache/user_artifacts/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TORCHRUN="${TORCHRUN:-$TRAINERS_SRC/server/.venv/bin/torchrun}"
export PYTHONPATH="$TRAINERS_SRC/server/src:${PYTHONPATH:-}"
# Put the venv on PATH so Megatron's runtime build of the dataset `helpers_cpp`
# extension resolves `python3 -m pybind11 --includes` to the venv interpreter
# (which has pybind11) rather than the system python3. Without this the C++
# build fails with "pybind11/pybind11.h: No such file or directory".
export PATH="$TRAINERS_SRC/server/.venv/bin:$PATH"
export BT_TRAINER_CONFIG_PATH="$CONFIG"

cd "$TRAINERS_SRC/server"
echo "[run_trainer_node] rank=$BT_NODE_RANK nnodes=$NUM_NODES leader=$BT_LEADER_ADDR config=$CONFIG"
echo "[run_trainer_node] config contents:"; cat "$CONFIG"
exec bash scripts/launch.sh
