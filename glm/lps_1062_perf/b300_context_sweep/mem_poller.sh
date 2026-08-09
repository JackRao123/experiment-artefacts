#!/usr/bin/env bash
# Per-node nvidia-smi memory poller. Writes node-local log lines:
#   <epoch_s.float> <gpu0_MiB> <gpu1_MiB> ... <gpu7_MiB>
# Usage: bash mem_poller.sh <out_file> [period_s]
set -uo pipefail
OUT="${1:?output file}"
PERIOD="${2:-0.3}"
: > "$OUT"
while true; do
  TS=$(date +%s.%N)
  VALS=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd' ' -)
  [ -n "$VALS" ] && echo "$TS $VALS" >> "$OUT"
  sleep "$PERIOD"
done
