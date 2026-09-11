#!/usr/bin/env bash
set -uo pipefail
timeout_s=180
health_url="${TRAINER_HEALTH_URL:-http://127.0.0.1:8001}"
start=$(date +%s)

while ! curl -fs -m5 "$health_url/health" >/dev/null; do
  elapsed=$(( $(date +%s) - start ))
  if ! pgrep -f "[d]p_worker.main|[t]rainers_server_main.main" >/dev/null; then
    echo "TRAINER PROCESS DIED — leader log (last 40 lines):"
    tail -n 40 /root/glm53-1d3m-262k-20260909/.devbox_up/trainer_srun.log || true
    exit 1
  fi
  if [ "$elapsed" -ge "$timeout_s" ]; then
    echo "TIMEOUT: /health not up after ${timeout_s}s."
    echo "Startup usually takes 10–20 minutes, but failures usually surface in the first 1–2 minutes."
    echo "This three-minute checkpoint is intentional: inspect before stopping or continuing to wait."
    echo "--- trainer log (last 80 lines) ---"
    tail -n 80 /root/glm53-1d3m-262k-20260909/.devbox_up/trainer_srun.log || true
    if command -v squeue >/dev/null 2>&1; then
      echo "--- Slurm ---"
      squeue --name=devbox_trainer -o "%.18i %.9T %.8M %.20j %.40R" || true
      job_id=$(squeue -h --name=devbox_trainer -o "%A" | awk 'NR==1 {print $1}')
      if [ -z "$job_id" ]; then
        echo "No running devbox_trainer job; inspect the log above for its exit."
      else
        trainer_nodes=$(cat "/root/glm53-1d3m-262k-20260909/.devbox_up/trainer_num_nodes" 2>/dev/null || echo "1")
        [[ "$trainer_nodes" =~ ^[0-9]+$ ]] || trainer_nodes=1
        echo "--- GPU state (job $job_id; $trainer_nodes trainer nodes) ---"
        timeout 30 srun --jobid="$job_id" --overlap --nodes="$trainer_nodes" --ntasks="$trainer_nodes" \
          --ntasks-per-node=1 --cpus-per-task=1 --label bash -lc \
          'printf "host=%s\n" "$(hostname -s)"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader' \
          || echo "Could not collect GPU state within 30s."
      fi
    else
      echo "--- GPU state (no slurm; current node) ---"
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
    fi
    echo "If ranks are alive and logs progress, rerun in the background: bash /root/glm53-1d3m-262k-20260909/.devbox_up/wait_trainer_health.sh"
    exit 1
  fi
  sleep 5
done

echo "trainer healthy after $(( $(date +%s) - start ))s"
