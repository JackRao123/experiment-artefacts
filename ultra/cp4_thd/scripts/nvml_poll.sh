#!/usr/bin/env bash
# All-node NVML memory sampler for profiling runs.
#
# Start (from the devbox leader, inside a `source env.sh` shell):
#     rm -f "$OUT_DIR/_NVML_STOP"
#     nohup srun --overlap --nodes="$BT_GROUP_SIZE" --ntasks="$BT_GROUP_SIZE" \
#       --ntasks-per-node=1 bash /root/.cache/user_artifacts/ultra_cp4/scripts/nvml_poll.sh \
#       >/dev/null 2>&1 &
# Stop:
#     touch /root/.cache/user_artifacts/ultra_cp4/out/_NVML_STOP
#
# Each node appends "epoch_ts,hostname,gpu_index,memory_used_mib" rows to its
# own CSV under the shared out/ directory (per-node files avoid NFS append
# races).

set -u
OUT_DIR=/root/.cache/user_artifacts/ultra_cp4/out
STOP_FILE="$OUT_DIR/_NVML_STOP"
CSV="$OUT_DIR/nvml_$(hostname).csv"

while [ ! -f "$STOP_FILE" ]; do
  ts=$(date +%s.%N)
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | while IFS=', ' read -r idx mem; do
        printf '%s,%s,%s,%s\n' "$ts" "$(hostname)" "$idx" "$mem"
      done >> "$CSV"
  sleep 2
done
