#!/usr/bin/env bash
# LPS-1003 arming lab runner — ON the devbox node. One fresh process per
# invocation (each run = one boot observation).
# usage: run_one.sh NAME [--env KEY=VAL]... -- <arm_lab.py args>
set -euo pipefail
ARM=/root/arming
VENVPY=/root/.cache/user_artifacts/trainers_main/server/.venv/bin/python

NAME=$1; shift
mkdir -p "$ARM/results"

export PYTHONPATH="$ARM/unfixed"
unset CUDA_DEVICE_MAX_CONNECTIONS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${ARM_GPU:-0}

while [[ "${1:-}" == "--env" ]]; do
  shift; export "$1"; shift
done
[[ "${1:-}" == "--" ]] && shift

"$VENVPY" "$ARM/arm_lab.py" --out "$ARM/results/${NAME}.jsonl" "$@" \
  2>&1 | tee "$ARM/results/${NAME}.log"
