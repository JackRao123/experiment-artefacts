#!/usr/bin/env bash
# LPS-1003 devbox boot with the layer-bisection tracer armed (and the warmup
# mitigation OFF, prod-equivalent). Run ON THE DEVBOX LEADER.
set -euo pipefail
LPS=/root/.cache/user_artifacts/lps1003
PAR=$LPS/parity
source /root/.cache/user_artifacts/env.sh

export BT_SKIP_FULL_WARMUP=1
export BT_TRAINER_CONFIG_PATH=$LPS/trainer-config.flash.json
export BT_TRAINER_SERVER_CONFIG_PATH=$LPS/trainer-server-config.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

export PYTHONPATH="$PAR/harness${PYTHONPATH:+:$PYTHONPATH}"
export BT_LTRACE=1
STAMP=$(date +%m%d_%H%M%S)
export BT_LTRACE_DIR=$PAR/runs/ltrace_$STAMP
mkdir -p "$BT_LTRACE_DIR"
printf '%s\n' "$BT_LTRACE_DIR" > "$PAR/runs/latest_ltrace_dir"
echo "ltrace dir: $BT_LTRACE_DIR"
bash /root/.cache/user_artifacts/.devbox_up/start_trainer.sh
