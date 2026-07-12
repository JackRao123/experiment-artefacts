#!/usr/bin/env bash
# Nemotron-3-Super BF16: TP=8 leg (completes the TP=8/4/2 comparison in
# super/sampler_4vs2.md). Cold warmup boot first so the measured boot is warm,
# matching the TP=4/TP=2 methodology.
set -uo pipefail
cd "$(dirname "$0")"

export RECIPE=nemotron-bf16
export MODEL_GLOB=models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16
export DATA_LENS=262144
export ADAPTER=${ADAPTER:-/root/.cache/user_artifacts/nemotron_lora_test/sampler_weights/pirate-v1}

echo "=== CELL super tp=8 coldwarmup $(date -u +%H:%M:%S)"
TAG=super_tp8_cold timeout 5400 ./run_cell.sh 8 262144 0 --boot-only || echo "CELL_FAILED super_tp8_cold rc=$?"

echo "=== CELL super tp=8 boot=262144 $(date -u +%H:%M:%S)"
TAG=super_tp8 timeout 7200 ./run_cell.sh 8 262144 || echo "CELL_FAILED super_tp8 rc=$?"

echo SWEEP_DONE
