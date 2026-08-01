#!/usr/bin/env bash
# LPS-1003 attribution driver: one boot of run_trainer_node_attrib.sh in the
# given mode, window probes at READY, short steady batch, then STOP the
# trainer (verified). usage: devbox_attrib.sh <mode> <trace:0|1>
set -uo pipefail
MODE="${1:?mode}"
TRACE="${2:-0}"
LPS=/root/.cache/user_artifacts/lps1003
PAR=$LPS/parity
DU=/root/.cache/user_artifacts/.devbox_up
OUT=$PAR/runs/attrib_${MODE}_$(date +%m%d_%H%M%S)
mkdir -p "$OUT"
exec >>"$OUT/driver.log" 2>&1
echo "=== attrib driver mode=$MODE trace=$TRACE start $(date -u +%FT%TZ) out=$OUT"
source /root/.cache/user_artifacts/env.sh

export BT_SKIP_FULL_WARMUP=1
export BT_TRAINER_CONFIG_PATH=$LPS/trainer-config.flash.json
export BT_TRAINER_SERVER_CONFIG_PATH=$LPS/trainer-server-config.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export BT_ATTRIB_MODE=$MODE
if [ "$TRACE" = "1" ]; then
  export PYTHONPATH="$PAR/harness${PYTHONPATH:+:$PYTHONPATH}"
  export BT_LTRACE=1 BT_LTRACE_DIR=$OUT/ltrace
  mkdir -p "$BT_LTRACE_DIR"
elif [ "$TRACE" = "2" ]; then
  export PYTHONPATH="$PAR/harness2${PYTHONPATH:+:$PYTHONPATH}"
  export BT_LTRACE=1 BT_LTRACE_DIR=$OUT/ltrace
  export BT_DSA_ROWSUM=1 BT_DSA_ROWSUM_DIR=$OUT/rowsum
  mkdir -p "$BT_LTRACE_DIR" "$OUT/rowsum"
fi

export NUM_NODES=2
export MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
printf '2\n' > "$DU/trainer_num_nodes"
nohup srun --job-name=devbox_trainer --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  --gres=gpu:8 --cpus-per-task=248 --export=ALL \
  bash "$PAR/run_trainer_node_attrib.sh" > "$DU/trainer_srun.log" 2>&1 < /dev/null &
echo "trainer dispatched (mode=$MODE, srun pid $!)"
sleep 20
nodelist=$(squeue -h --name=devbox_trainer -o %N | head -1)
rank0_host=$(scontrol show hostnames "$nodelist" | head -1)
rank0_ip=$(getent hosts "$rank0_host" | awk '{print $1}' | head -1)
URL="http://${rank0_ip:-127.0.0.1}:8001"
echo "rank0_host=$rank0_host url=$URL"

python3 "$PAR/probe_lp.py" "$PAR/probe_batch0_forward.json" \
  --url "$URL" --reps 8 --tag ${MODE}_window --out "$OUT" --wait-health 2400 \
  || { echo "WINDOW FAILED mode=$MODE"; squeue; tail -60 "$DU/trainer_srun.log"; }

python3 "$PAR/probe_lp.py" "$PAR/probe_batch0_forward.json" \
  --url "$URL" --reps 4 --tag ${MODE}_steady --out "$OUT"

bash "$DU/stop_trainer.sh" || true
sleep 10
squeue
echo "=== attrib driver mode=$MODE DONE $(date -u +%FT%TZ)"
