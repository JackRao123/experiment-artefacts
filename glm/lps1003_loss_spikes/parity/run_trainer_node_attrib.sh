#!/usr/bin/env bash
# LPS-1003 attribution: STOCK devbox trainer env with exactly ONE change,
# selected by BT_ATTRIB_MODE:
#   prodenv  full prod-exact env (same set as run_trainer_node_prodenv.sh)
#   conn     stock MINUS CUDA_DEVICE_MAX_CONNECTIONS=1
#   gloo     stock MINUS MEGATRON_SKIP_GLOO_GROUPS=1
#   nvte     stock MINUS NVTE_FRAMEWORK=pytorch
#   arch     stock PLUS NVTE_CUDA_ARCHS=103a
#   stock    no change (control)
set -euo pipefail
source /root/.cache/user_artifacts/env.sh
SRC=/root/.cache/user_artifacts/trainers_main
export NUM_NODES="${NUM_NODES:-2}" NUM_GPUS=8
export BT_NODE_RANK="$SLURM_NODEID"
if [ -n "${SLURM_JOB_NODELIST:-}" ] && command -v scontrol >/dev/null 2>&1; then
  lead_host="$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | head -1 || true)"
  lead_addr="$(scontrol show node "$lead_host" 2>/dev/null | sed -n 's/.*NodeAddr=\([^ ]*\).*/\1/p' | head -1 || true)"
  if [ -n "$lead_addr" ]; then
    export BT_LEADER_ADDR="$lead_addr"
    echo "node $BT_NODE_RANK: rendezvous master=$lead_host ($lead_addr)"
  fi
fi
export MASTER_PORT="${MASTER_PORT:-29500}"
export TORCHRUN=$SRC/server/.venv/bin/torchrun
export PATH=$SRC/server/.venv/bin:$PATH
: "${BT_TRAINER_CONFIG_PATH:?set BT_TRAINER_CONFIG_PATH}"
: "${BT_ATTRIB_MODE:?set BT_ATTRIB_MODE}"
export USE_HF=1 PORT=8001 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GLOO_SOCKET_IFNAME=eth0
export LD_LIBRARY_PATH=""
# stock devbox set
export NVTE_FRAMEWORK=pytorch
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MEGATRON_SKIP_GLOO_GROUPS=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS=1
case "$BT_ATTRIB_MODE" in
  stock)   ;;
  conn)    unset CUDA_DEVICE_MAX_CONNECTIONS ;;
  gloo)    unset MEGATRON_SKIP_GLOO_GROUPS ;;
  nvte)    unset NVTE_FRAMEWORK ;;
  arch)    export NVTE_CUDA_ARCHS=103a ;;
  prodenv) export NVTE_CUDA_ARCHS=103a
           unset NVTE_FRAMEWORK CUDA_DEVICE_MAX_CONNECTIONS \
                 MEGATRON_SKIP_GLOO_GROUPS TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC \
                 NCCL_DEBUG OMP_NUM_THREADS ;;
  penvconn) # necessity test: full prodenv BUT keep CUDA_DEVICE_MAX_CONNECTIONS=1
           export NVTE_CUDA_ARCHS=103a
           unset NVTE_FRAMEWORK MEGATRON_SKIP_GLOO_GROUPS \
                 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC NCCL_DEBUG OMP_NUM_THREADS ;;
  *) echo "bad BT_ATTRIB_MODE=$BT_ATTRIB_MODE" >&2; exit 2 ;;
esac
echo "node $BT_NODE_RANK: ATTRIB_MODE=$BT_ATTRIB_MODE conn=${CUDA_DEVICE_MAX_CONNECTIONS:-UNSET} gloo=${MEGATRON_SKIP_GLOO_GROUPS:-UNSET} nvte=${NVTE_FRAMEWORK:-UNSET} archs=${NVTE_CUDA_ARCHS:-UNSET}"
if command -v ibv_devinfo >/dev/null 2>&1; then
  if [ -z "${NCCL_IB_HCA:-}" ] && ibv_devinfo 2>/dev/null | grep -q '^hca_id:.*mlx5_bond'; then
    export NCCL_IB_HCA=mlx5_bond
    export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
    export NCCL_IB_MERGE_VFS="${NCCL_IB_MERGE_VFS:-0}"
    export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
    export NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-0}"
    export NCCL_SHARP_DISABLE="${NCCL_SHARP_DISABLE:-1}"
  fi
fi
unset S3_MANIFEST_PATH
cd "$SRC/server"
exec bash scripts/launch.sh
