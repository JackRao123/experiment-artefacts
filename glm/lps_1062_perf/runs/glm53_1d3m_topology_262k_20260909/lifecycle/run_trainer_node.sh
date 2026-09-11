#!/usr/bin/env bash
set -euo pipefail
source /root/.cache/user_artifacts/devboxes/32vj99q/env.sh
SRC=/root/glm53-main-262k-20260909/trainers
export LAYER_TIMING_DIR="${LAYER_TIMING_DIR:?}"
export PYTHONPATH=/root/glm53-1d3m-262k-20260909/instrumentation:"$SRC/models/src:$SRC/server-interface/src:$SRC/server-main/src:$SRC/server-megatron-bridge/src:$SRC/baseten-weight-sync:$SRC/server-megatron-bridge/vendor/megatron-bridge/src:$SRC/server-megatron-bridge/vendor/megatron-bridge/3rdparty/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
export NUM_NODES="${NUM_NODES:-1}" NUM_GPUS="${NUM_GPUS:-8}"
export BT_NODE_RANK="${SLURM_NODEID:-${BT_NODE_RANK:-0}}"
# Use Slurm node 0's routable address for torchrun rendezvous.
if [ -n "${SLURM_JOB_NODELIST:-}" ] && command -v scontrol >/dev/null 2>&1; then
  lead_host="$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | head -1 || true)"
  lead_addr="$(scontrol show node "$lead_host" 2>/dev/null | sed -n 's/.*NodeAddr=\([^ ]*\).*/\1/p' | head -1 || true)"
  if [ -n "$lead_addr" ]; then
    export BT_LEADER_ADDR="$lead_addr"
    echo "node $BT_NODE_RANK: rendezvous master=$lead_host ($lead_addr)"
  else
    echo "node $BT_NODE_RANK: WARNING could not derive rendezvous master from Slurm; using BT_LEADER_ADDR=${BT_LEADER_ADDR:-unset}" >&2
  fi
fi
export MASTER_PORT="${MASTER_PORT:-29500}"
export TORCHRUN=$SRC/server-megatron-bridge/.venv/bin/torchrun
export PATH=$SRC/server-megatron-bridge/.venv/bin:$PATH
: "${BT_TRAINER_CONFIG_PATH:?set BT_TRAINER_CONFIG_PATH to a trainer config JSON on the shared FS}"
export USE_HF=1 PORT=8001 PYTHONUNBUFFERED=1
export GLOO_SOCKET_IFNAME=eth0
if command -v ibv_devinfo >/dev/null 2>&1; then
  # Prefer routed RoCE bonds when present.
  if [ -z "${NCCL_IB_HCA:-}" ] && ibv_devinfo 2>/dev/null | grep -q '^hca_id:.*mlx5_bond'; then
    export NCCL_IB_HCA=mlx5_bond
    export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
    export NCCL_IB_MERGE_VFS="${NCCL_IB_MERGE_VFS:-0}"
    export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
    export NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-0}"
    export NCCL_SHARP_DISABLE="${NCCL_SHARP_DISABLE:-1}"
    echo "node $BT_NODE_RANK: RoCE bonds detected -> NCCL_IB_HCA=mlx5_bond (GID 3)"
  else
    IB_HCA="$(ibv_devinfo 2>/dev/null | sed -n -e '/hca_id/p' -e '/link_layer:/p' \
      | grep -B1 InfiniBand | grep hca_id | sed -e 's/^hca_id://g' \
      | tr -d '[:blank:]' | paste -sd, || true)"
    [ -n "$IB_HCA" ] && export NCCL_IB_HCA="${NCCL_IB_HCA:-$IB_HCA}" && echo "node $BT_NODE_RANK: NCCL_IB_HCA=$NCCL_IB_HCA"
  fi
fi
unset S3_MANIFEST_PATH
cd "$SRC/server-main"
exec bash scripts/launch.sh --backend megatron_bridge
