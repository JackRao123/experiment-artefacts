#!/usr/bin/env bash
# Pre-flight: dump parallel-state groups for PP2/CP8/EP8 (gibbs, LPS-1062).
# Static rendezvous: c10d calls getaddrinfo(own hostname) which fails on these
# pods (/etc/hosts carries only OTHER nodes); launch.sh uses static too.
set -euo pipefail
SRC=/root/.cache/user_artifacts/trainers_main
PP2=/root/.cache/user_artifacts/lps1062_pp2
export MEGATRON_LM_PATH=$SRC/server/vendor/megatron-bridge/3rdparty/Megatron-LM
lead_host="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
lead_addr="$(scontrol show node "$lead_host" | sed -n "s/.*NodeAddr=\([^ ]*\).*/\1/p" | head -1)"
export MASTER_ADDR="$lead_addr" MASTER_PORT=29617
export GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0
if ibv_devinfo 2>/dev/null | grep -q "^hca_id:.*mlx5_bond"; then
  export NCCL_IB_HCA=mlx5_bond NCCL_IB_GID_INDEX=3 NCCL_IB_MERGE_VFS=0
  export NCCL_NET_PLUGIN=none NCCL_COLLNET_ENABLE=0 NCCL_SHARP_DISABLE=1
fi
echo "node $SLURM_NODEID: rendezvous $MASTER_ADDR:$MASTER_PORT rank=$SLURM_NODEID"
exec "$SRC/server/.venv/bin/torchrun" --nnodes=2 --nproc_per_node=8 \
  --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  "$PP2/dump_parallel_groups.py" --pp 2 --cp 8 --ep 8
