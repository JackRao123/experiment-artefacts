#!/usr/bin/env bash
# TRN-1488 sweep: one boot per TP at the golden 262144 max_model_len; the
# 8k/32k/131k/256k buckets flow through the same engine. If TP=4 cannot boot
# at 256k (KV pool < one full sequence), that IS the E4 answer — fall back to
# a 131k boot so TP=4 still yields 8k/32k/131k numbers.
set -uo pipefail
cd "$(dirname "$0")"

echo "=== CELL tp=8 boot=262144 $(date -u +%H:%M:%S)"
timeout 10800 ./run_cell.sh 8 262144 || echo "CELL_FAILED tp=8 boot=262144 rc=$?"

echo "=== CELL tp=4 boot=262144 $(date -u +%H:%M:%S)"
if ! timeout 10800 ./run_cell.sh 4 262144; then
  echo "CELL_FAILED tp=4 boot=262144 rc=$?"
  echo "=== CELL tp=4 boot=131072 (fallback) $(date -u +%H:%M:%S)"
  timeout 10800 ./run_cell.sh 4 131072 || echo "CELL_FAILED tp=4 boot=131072 rc=$?"
fi
echo SWEEP_DONE
