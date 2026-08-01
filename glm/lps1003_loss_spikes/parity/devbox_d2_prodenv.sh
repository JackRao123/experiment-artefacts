#!/usr/bin/env bash
# LPS-1003 parity D2: fresh devbox boot with PROD-EXACT trainer env
# (run_trainer_node_prodenv.sh) + the layer tracer armed. Window probes at
# READY, then steady probes. Run ON THE DEVBOX LEADER via nohup.
set -uo pipefail
LPS=/root/.cache/user_artifacts/lps1003
PAR=$LPS/parity
DU=/root/.cache/user_artifacts/.devbox_up
OUT=$PAR/runs/d2_prodenv_$(date +%m%d_%H%M%S)
mkdir -p "$OUT"
LOG=$OUT/driver.log
exec >>"$LOG" 2>&1
echo "=== d2 prodenv driver start $(date -u +%FT%TZ) out=$OUT"
source /root/.cache/user_artifacts/env.sh

export BT_SKIP_FULL_WARMUP=1
export BT_TRAINER_CONFIG_PATH=$LPS/trainer-config.flash.json
export BT_TRAINER_SERVER_CONFIG_PATH=$LPS/trainer-server-config.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PAR/harness${PYTHONPATH:+:$PYTHONPATH}"
export BT_LTRACE=1
export BT_LTRACE_DIR=$OUT/ltrace
mkdir -p "$BT_LTRACE_DIR"

# dispatch (mirrors start_trainer.sh, but with the prodenv node script)
export NUM_NODES=2
export MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
printf '2\n' > "$DU/trainer_num_nodes"
nohup srun --job-name=devbox_trainer --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  --gres=gpu:8 --cpus-per-task=248 --export=ALL \
  bash "$PAR/run_trainer_node_prodenv.sh" > "$DU/trainer_srun.log" 2>&1 < /dev/null &
echo "trainer dispatched (srun pid $!, master_port $MASTER_PORT)"
sleep 20

nodelist=$(squeue -h --name=devbox_trainer -o %N | head -1)
rank0_host=$(scontrol show hostnames "$nodelist" | head -1)
rank0_ip=$(getent hosts "$rank0_host" | awk '{print $1}' | head -1)
URL="http://${rank0_ip:-127.0.0.1}:8001"
echo "rank0_host=$rank0_host url=$URL"

python3 "$PAR/probe_lp.py" "$PAR/probe_batch0_forward.json" \
  --url "$URL" --reps 8 --tag d2_penv_window --out "$OUT" --wait-health 2400 \
  || { echo "D2 WINDOW FAILED"; squeue; tail -60 "$DU/trainer_srun.log"; exit 2; }

python3 "$PAR/probe_lp.py" "$PAR/probe_batch0_forward.json" \
  --url "$URL" --reps 6 --tag d2_penv_steady --out "$OUT"

echo "=== d2 prodenv driver DONE $(date -u +%FT%TZ)"
