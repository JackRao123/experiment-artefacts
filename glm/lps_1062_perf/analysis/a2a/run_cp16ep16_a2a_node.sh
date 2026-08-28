#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_PROCID:?Launch this wrapper through srun with one task per node}"

export NCCL_IB_HCA=mlx5_bond
export NCCL_IB_GID_INDEX=3
export NCCL_IB_MERGE_VFS=0
export NCCL_NET_PLUGIN=none
export NCCL_COLLNET_ENABLE=0
export NCCL_SHARP_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

BOX=/root/.cache/user_artifacts/devboxes/w5y89m3
VENV="$BOX/trainers/server-megatron-bridge/.venv"

exec "$VENV/bin/torchrun" \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank="$SLURM_PROCID" \
    --master_addr=10.1.77.34 \
    --master_port="${MASTER_PORT:-29632}" \
    "$BOX/simulate_cp16ep16_a2a.py" \
    "$@"
