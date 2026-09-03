#!/usr/bin/env bash
set -uo pipefail

RUN_DIR=/root/.cache/user_artifacts/lps1062_native_blockwise_20260828/phase2
job_name=$(cat "$RUN_DIR/optimized_runtime_active_job_name")
log_path=$(cat "$RUN_DIR/optimized_runtime_active_log_path")
start=$(date +%s)
while ! curl -fs -m5 http://127.0.0.1:8001/health >/dev/null; do
  elapsed=$(( $(date +%s) - start ))
  if ! squeue -h --name="$job_name" | grep -q .; then
    echo "TRAINER JOB DIED"
    tail -n 80 "$log_path" || true
    exit 1
  fi
  if [ "$elapsed" -ge 180 ]; then
    echo "TIMEOUT: trainer is not healthy after ${elapsed}s"
    tail -n 120 "$log_path" || true
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
    exit 1
  fi
  sleep 5
done
echo "trainer healthy after $(( $(date +%s) - start ))s"
