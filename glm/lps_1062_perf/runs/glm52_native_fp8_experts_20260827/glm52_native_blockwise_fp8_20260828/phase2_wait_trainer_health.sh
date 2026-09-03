#!/usr/bin/env bash
set -uo pipefail

RUN_DIR=/root/.cache/user_artifacts/lps1062_native_blockwise_20260828/phase2
timeout_s=180
start=$(date +%s)
while ! curl -fs -m5 http://127.0.0.1:8001/health >/dev/null; do
  elapsed=$(( $(date +%s) - start ))
  if ! squeue -h --name=glm52_native_phase2 | grep -q .; then
    echo "TRAINER JOB EXITED; last 80 log lines:"
    tail -n 80 "$RUN_DIR/trainer_srun.log" || true
    exit 1
  fi
  if [ "$elapsed" -ge "$timeout_s" ]; then
    echo "TIMEOUT: trainer health unavailable after ${timeout_s}s"
    echo "--- trainer log ---"
    tail -n 80 "$RUN_DIR/trainer_srun.log" || true
    echo "--- Slurm ---"
    squeue --name=glm52_native_phase2 -o "%.18i %.9T %.8M %.20j %.40R" || true
    job_id=$(squeue -h --name=glm52_native_phase2 -o "%A" | awk 'NR==1 {print $1}')
    if [ -n "$job_id" ]; then
      echo "--- GPU state ---"
      timeout 30 srun --jobid="$job_id" --overlap --nodes=1 --ntasks=1 \
        --ntasks-per-node=1 --cpus-per-task=1 --label bash -lc \
        'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader' \
        || echo "Could not collect GPU state within 30s"
    fi
    exit 1
  fi
  sleep 5
done
echo "trainer healthy after $(( $(date +%s) - start ))s"
