#!/usr/bin/env bash
# TRN-1488 cell runner (runs ON the devbox).
# Usage: run_cell.sh TP [BOOT_CTX=262144] [N=0] [--boot-only]
# One boot per TP at BOOT_CTX; data-length buckets (8k/32k/131k/256k, capped
# at BOOT_CTX) all flow through the same engine.
#
# Sets CUDA_VISIBLE_DEVICES from TP, exports the Nemotron-NVFP4 patch-gate env,
# runs the 1 Hz nvidia-smi poller across ALL 8 GPUs (idle GPUs are evidence),
# tees the vLLM log, and post-parses everything into out/<tag>.summary.json.
set -uo pipefail
cd "$(dirname "$0")"

TP=${1:?usage: run_cell.sh TP [BOOT_CTX] [N] [--boot-only]}
CTX=${2:-262144}
N=${3:-0}
EXTRA=${4:-}

source /root/.cache/user_artifacts/env.sh
VENV=${VENV:-/root/.cache/user_artifacts/trainers_main/sampler/.venv}

# Recipe/model knobs, overridable via env for other NemotronH variants
# (Super BF16: RECIPE=nemotron-bf16 MODEL_GLOB=...Super-120B-A12B-BF16...).
RECIPE=${RECIPE:-nemotron-nvfp4}
MODEL_GLOB=${MODEL_GLOB:-models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4}
DATA_LENS=${DATA_LENS:-8192,32768,131072,262144}

# Staged snapshots live in the user_artifacts HF cache (where the nvfp4
# spike put them), not the env.sh team-artifacts HF_HOME. Serve fully
# offline; MODEL_PATH doubles as the NVFP4 patch-gate signal (no-op on BF16).
export HF_HOME=/root/.cache/user_artifacts/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ -z ${MODEL_PATH:-} ]]; then
  MODEL_PATH=$(ls -d "$HF_HOME"/hub/$MODEL_GLOB/snapshots/*/ 2>/dev/null | head -1)
fi
MODEL_PATH=${MODEL_PATH%/}
[[ -n $MODEL_PATH && -f $MODEL_PATH/config.json ]] || { echo "FATAL: model snapshot not found (set MODEL_PATH)"; exit 1; }
export MODEL_PATH
export _BASETEN_SERVED_MODEL=nemotronhforcausallm

# Matching-base adapter if present on this cluster; empty = no LoRA request
# (enable_lora buffers still on, matching the golden config's worst case).
ADAPTER=${ADAPTER:-/root/.cache/user_artifacts/rl/gsm8k-ultra-weights/sampler_weights/step-199}
[[ -f ${ADAPTER}/adapter_config.json ]] || ADAPTER=""

case $TP in
  2) export CUDA_VISIBLE_DEVICES=0,1 ;;
  4) export CUDA_VISIBLE_DEVICES=0,1,2,3 ;;
  *) export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ;;
esac

TAG=${TAG:-tp${TP}_boot${CTX}}
mkdir -p out
OUT=out/$TAG

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 1 > "$OUT.smi.csv" 2>/dev/null &
POLL=$!
trap 'kill $POLL 2>/dev/null' EXIT

BOOT_FLAG=""
[[ $EXTRA == --boot-only ]] && BOOT_FLAG="--boot-only"

"$VENV/bin/python" bench.py \
  --tp "$TP" --max-model-len "$CTX" --num-requests "$N" $BOOT_FLAG \
  --recipe "$RECIPE" --data-lens "$DATA_LENS" \
  --model "$MODEL_PATH" --adapter "$ADAPTER" --log-file "$OUT.log" \
  --out "$OUT.result.json" 2>&1 | tee "$OUT.log"
RC=${PIPESTATUS[0]}

kill $POLL 2>/dev/null; trap - EXIT
python3 parse_cell.py --log "$OUT.log" --smi "$OUT.smi.csv" \
  --result "$OUT.result.json" --out "$OUT.summary.json" >/dev/null || true
echo "CELL_DONE $TAG rc=$RC adapter=${ADAPTER:-none}"
exit "$RC"
