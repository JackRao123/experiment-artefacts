#!/usr/bin/env bash
# LPS-1003 parity, devbox arm: prod-equivalent fresh boot (warmup mitigation
# OFF), then window probes (D0) the moment /health answers, then steady-state
# probes (D1). Run ON THE DEVBOX LEADER via nohup; progress -> $LOG.
set -uo pipefail

LPS=/root/.cache/user_artifacts/lps1003
PAR=$LPS/parity
DU=/root/.cache/user_artifacts/.devbox_up
OUT=$PAR/runs/devbox_$(date +%m%d_%H%M%S)
mkdir -p "$OUT"
LOG=$OUT/driver.log
exec >>"$LOG" 2>&1

echo "=== devbox parity driver start $(date -u +%FT%TZ) out=$OUT"
source /root/.cache/user_artifacts/env.sh

export BT_SKIP_FULL_WARMUP=1
export BT_TRAINER_CONFIG_PATH=$LPS/trainer-config.flash.json
export BT_TRAINER_SERVER_CONFIG_PATH=$LPS/trainer-server-config.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

bash "$DU/start_trainer.sh" || { echo "START FAILED"; exit 1; }
sleep 20

nodelist=$(squeue -h --name=devbox_trainer -o %N | head -1)
rank0_host=$(scontrol show hostnames "$nodelist" | head -1)
rank0_ip=$(getent hosts "$rank0_host" | awk '{print $1}' | head -1)
URL="http://${rank0_ip:-127.0.0.1}:8001"
echo "rank0_host=$rank0_host url=$URL"

# D0: window probes, fired the instant health answers (bounded 2400s).
python3 "$PAR/probe_lp.py" "$PAR/probe_batch0_forward.json" \
  --url "$URL" --reps 6 --tag d0_window --out "$OUT" --wait-health 2400 \
  || { echo "D0 FAILED"; squeue; tail -40 "$DU/trainer_srun.log"; exit 2; }

# D1: steady state (trainer now warm; weights frozen, B=0 adapter).
python3 "$PAR/probe_lp.py" "$PAR/probe_batch0_forward.json" \
  --url "$URL" --reps 8 --tag d1_steady --out "$OUT"

# fingerprint while the trainer is alive (on rank0 node, where the server runs)
srun --overlap --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -lc \
  "python3 $PAR/fingerprint.py --python /root/.cache/user_artifacts/trainers_main/server/.venv/bin/python --out $OUT/fingerprint_\$(hostname).json" \
  || echo "fingerprint via srun failed (queued behind trainer?); will retry direct-ssh later"

echo "=== devbox parity driver DONE $(date -u +%FT%TZ)"
