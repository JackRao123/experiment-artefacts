#!/usr/bin/env bash
set -uo pipefail
mapfile -t sampler_pids < <(pgrep -f "[s]ampler.vllm_server" || true)

if [ "${#sampler_pids[@]}" -eq 0 ]; then
  echo "no sampler process found on this node"
else
  for pid in "${sampler_pids[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    echo "SIGTERM sent to sampler process group $pid"
  done

  deadline=$(( $(date +%s) + 30 ))
  while pgrep -f "[s]ampler.vllm_server" >/dev/null && [ "$(date +%s)" -lt "$deadline" ]; do
    sleep 1
  done

  mapfile -t sampler_pids < <(pgrep -f "[s]ampler.vllm_server" || true)
  if [ "${#sampler_pids[@]}" -gt 0 ]; then
    for pid in "${sampler_pids[@]}"; do
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
      echo "SIGKILL sent to sampler process group $pid after 30s"
    done
  fi
fi

echo "--- GPU state after stop (current node) ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
