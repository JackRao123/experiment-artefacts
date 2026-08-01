#!/usr/bin/env bash
# LPS-1003 parity: run_trainer_node.sh with the trainer env aligned to the PROD
# pod environ (fingerprint 5wolkzw, 2026-07-31). Deltas vs the stock devbox
# script — these four devbox-only exports REMOVED:
#   NVTE_FRAMEWORK=pytorch, CUDA_DEVICE_MAX_CONNECTIONS=1,
#   MEGATRON_SKIP_GLOO_GROUPS=1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600,
#   NCCL_DEBUG=WARN, OMP_NUM_THREADS=1
# and this prod-only export ADDED: NVTE_CUDA_ARCHS=103a.
# Everything else (leader-addr fix, RoCE detection, allocator conf, port) is
# byte-equal to the stock script.
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
  else
    echo "node $BT_NODE_RANK: WARNING could not derive rendezvous master from Slurm; using BT_LEADER_ADDR=${BT_LEADER_ADDR:-unset}" >&2
  fi
fi
export MASTER_PORT="${MASTER_PORT:-29500}"
export TORCHRUN=$SRC/server/.venv/bin/torchrun
export PATH=$SRC/server/.venv/bin:$PATH
: "${BT_TRAINER_CONFIG_PATH:?set BT_TRAINER_CONFIG_PATH to a trainer config JSON on the shared FS}"
export USE_HF=1 PORT=8001 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GLOO_SOCKET_IFNAME=eth0
export LD_LIBRARY_PATH=""
export NVTE_CUDA_ARCHS=103a
# prod-parity: ensure the devbox-only vars are absent even if inherited
unset NVTE_FRAMEWORK CUDA_DEVICE_MAX_CONNECTIONS MEGATRON_SKIP_GLOO_GROUPS \
      TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC NCCL_DEBUG OMP_NUM_THREADS 2>/dev/null || true
if command -v ibv_devinfo >/dev/null 2>&1; then
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
    IB_HCA="$(ibv_devinfo 2>/dev/null | sed -n -e '/hca_id/p' -e '/link_layer:/p'       | grep -B1 InfiniBand | grep hca_id | sed -e 's/^hca_id://g'       | tr -d '[:blank:]' | paste -sd, || true)"
    [ -n "$IB_HCA" ] && export NCCL_IB_HCA="${NCCL_IB_HCA:-$IB_HCA}" && echo "node $BT_NODE_RANK: NCCL_IB_HCA=$NCCL_IB_HCA"
  fi
fi
unset S3_MANIFEST_PATH
cd "$SRC/server"
exec bash scripts/launch.sh
