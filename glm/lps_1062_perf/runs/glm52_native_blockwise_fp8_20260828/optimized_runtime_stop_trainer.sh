#!/usr/bin/env bash
set -uo pipefail

RUN_DIR=/root/.cache/user_artifacts/lps1062_native_blockwise_20260828/phase2
job_name=$(cat "$RUN_DIR/optimized_runtime_active_job_name" 2>/dev/null || true)
if [ -n "$job_name" ]; then
  scancel --name="$job_name" 2>/dev/null && echo "scancel sent for $job_name"
fi
sleep 10
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
