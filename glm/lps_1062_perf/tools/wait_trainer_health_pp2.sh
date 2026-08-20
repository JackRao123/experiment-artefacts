#!/usr/bin/env bash
# Pinned health waiter. Same 3-minute checkpoint discipline as the devbox-up
# copy; two fixes: our log path, and the worker process name, which the #1027
# restructure changed from dp_worker.main to trainers_server_main.main (the
# stock waiter would report "TRAINER PROCESS DIED" during a healthy boot).
set -uo pipefail
PP2=/root/.cache/user_artifacts/lps1062_pp2
LOG=$PP2/logs/trainer_srun.log
timeout_s=180
health_url="${TRAINER_HEALTH_URL:-http://127.0.0.1:8001}"
start=$(date +%s)
while ! curl -fs -m5 "$health_url/health" >/dev/null; do
  elapsed=$(( $(date +%s) - start ))
  if ! pgrep -f "[t]rainers_server_main.main" >/dev/null && ! squeue -h --name=devbox_trainer | grep -q .; then
    echo "TRAINER GONE (no worker process, no slurm job) — log tail:"
    tail -n 40 "$LOG" || true
    exit 1
  fi
  if [ "$elapsed" -ge "$timeout_s" ]; then
    echo "CHECKPOINT: /health not up after ${timeout_s}s (weight load is 13-20 min on a warm cache)."
    echo "--- trainer log (last 60 lines) ---"; tail -n 60 "$LOG" || true
    echo "--- slurm ---"; squeue --name=devbox_trainer -o "%.18i %.9T %.8M %.20j %.40R" || true
    exit 1
  fi
  sleep 5
done
echo "trainer healthy after $(( $(date +%s) - start ))s"
