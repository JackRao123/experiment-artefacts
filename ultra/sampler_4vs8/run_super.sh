#!/usr/bin/env bash
# Nemotron-3-Super BF16 quick check (TRN-1488 follow-up): TP=4 vs TP=2 at
# max_model_len=262144, single saturating 256k bucket each. Boot order gives
# a true warm boot for both measured cells:
#   1. TP=4 --boot-only  (cold: fills the Weka page cache, discarded)
#   2. TP=4 full cell    (warm boot + 256k tps)
#   3. TP=2 full cell    (warm boot + 256k tps)
set -uo pipefail
cd "$(dirname "$0")"

export RECIPE=nemotron-bf16
export MODEL_GLOB=models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16
export DATA_LENS=262144
# Super adapter (pirate LoRAs are Super-120B); run_cell blanks it if absent.
export ADAPTER=${ADAPTER:-/root/.cache/user_artifacts/nemotron_lora_test/sampler_weights/pirate-v1}

echo "=== CELL super tp=4 coldwarmup $(date -u +%H:%M:%S)"
TAG=super_tp4_cold timeout 5400 ./run_cell.sh 4 262144 0 --boot-only || echo "CELL_FAILED super_tp4_cold rc=$?"

echo "=== CELL super tp=4 boot=262144 $(date -u +%H:%M:%S)"
TAG=super_tp4 timeout 7200 ./run_cell.sh 4 262144 || echo "CELL_FAILED super_tp4 rc=$?"

echo "=== CELL super tp=2 boot=262144 $(date -u +%H:%M:%S)"
TAG=super_tp2 timeout 7200 ./run_cell.sh 2 262144 || echo "CELL_FAILED super_tp2 rc=$?"

echo SWEEP_DONE
