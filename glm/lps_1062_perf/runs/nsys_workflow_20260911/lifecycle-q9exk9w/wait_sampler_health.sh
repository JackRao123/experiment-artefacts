#!/usr/bin/env bash
set -uo pipefail
timeout_s=180
health_url="${SAMPLER_HEALTH_URL:-http://127.0.0.1:${PORT:-8000}}"
start=$(date +%s)

while ! curl -fs -m5 "$health_url/health" >/dev/null; do
  elapsed=$(( $(date +%s) - start ))
  if ! pgrep -f "[s]ampler.vllm_server" >/dev/null; then
    echo "SAMPLER PROCESS DIED — log (last 40 lines):"
    tail -n 40 /root/.cache/user_artifacts/devboxes/q9exk9w/.devbox_up/sampler.log || true
    exit 1
  fi
  if [ "$elapsed" -ge "$timeout_s" ]; then
    echo "TIMEOUT: /health not up after ${timeout_s}s."
    echo "This three-minute checkpoint is intentional: inspect before stopping or continuing to wait."
    echo "--- sampler log (last 80 lines) ---"
    tail -n 80 /root/.cache/user_artifacts/devboxes/q9exk9w/.devbox_up/sampler.log || true
    echo "--- GPU state (current node) ---"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
    echo "If the process is alive and logs progress, rerun in the background: PORT=${PORT:-8000} bash /root/.cache/user_artifacts/devboxes/q9exk9w/.devbox_up/wait_sampler_health.sh"
    exit 1
  fi
  sleep 5
done

echo "sampler healthy after $(( $(date +%s) - start ))s"
