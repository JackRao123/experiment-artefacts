#!/usr/bin/env bash
# 1 Hz nvidia-smi memory poller. Runs ON a devbox node; appends CSV rows
#   epoch_seconds,gpu_index,memory.used_MiB
# to $OUT (default /root/.cache/user_artifacts/glm_prof/poll/<hostname>.csv).
set -u
OUT="${OUT:-/root/.cache/user_artifacts/glm_prof/poll/$(hostname -s).csv}"
mkdir -p "$(dirname "$OUT")"
while true; do
  ts="$(date +%s)"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -v ts="$ts" -F', *' '{print ts","$1","$2}' >> "$OUT"
  sleep 1
done
