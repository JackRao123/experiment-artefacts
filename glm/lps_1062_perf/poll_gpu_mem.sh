#!/bin/bash
# Per-GPU memory poller: one instance per node via srun. Appends
# ts,gpu_idx,used_mib rows every 2s until killed.
out="$1/mem.$(hostname).csv"
echo "ts,idx,used_mib" > "$out"
while :; do
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -v t="$(date +%s)" -F', ' '{print t","$1","$2}' >> "$out"
  sleep 2
done
