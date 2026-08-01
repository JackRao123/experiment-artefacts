#!/usr/bin/env bash
# LPS-1003 minimal-config repro attempt: cp1/pp1/tp1/ep16 (dp16) @ 64k,
# synthetic 16x50k payload, 5 /forward reps at the boot window.
# Prod-faithful env (conn UNSET via run_trainer_node_prodenv.sh) +
# BT_SKIP_FULL_WARMUP=1. Mirrors parity/devbox_d2_prodenv.sh dispatch.
set -uo pipefail
LPS=/root/.cache/user_artifacts/lps1003
PAR=$LPS/parity
MIN=$LPS/min64k
DU=/root/.cache/user_artifacts/.devbox_up
OUT=$MIN/runs/min64k_$(date +%m%d_%H%M%S)
mkdir -p "$OUT"
exec >>"$OUT/driver.log" 2>&1
echo "=== min64k driver start $(date -u +%FT%TZ) out=$OUT"
source /root/.cache/user_artifacts/env.sh

export BT_SKIP_FULL_WARMUP=1
export BT_TRAINER_CONFIG_PATH=$MIN/trainer-config.min64k.json
export BT_TRAINER_SERVER_CONFIG_PATH=$LPS/trainer-server-config.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

export NUM_NODES=2
export MASTER_PORT=$((29500 + RANDOM % 1000))
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

python3 "$PAR/probe_lp.py" "$MIN/payload_synth2x50k.json" \
  --url "$URL" --reps 5 --tag min64k_window --out "$OUT" --wait-health 2400 \
  || { echo "MIN64K WINDOW FAILED"; squeue; tail -60 "$DU/trainer_srun.log"; exit 2; }

echo "=== min64k driver DONE $(date -u +%FT%TZ)"
