#!/usr/bin/env bash
set -uo pipefail
if command -v scancel >/dev/null 2>&1; then
  scancel --name=devbox_trainer 2>/dev/null && echo "scancel devbox_trainer sent"
elif [ -f "/root/glm53-1d3m-262k-20260909/.devbox_up/trainer.pid" ]; then
  pgid=$(cat "/root/glm53-1d3m-262k-20260909/.devbox_up/trainer.pid")
  kill -TERM -- "-$pgid" 2>/dev/null && echo "SIGTERM sent to trainer process group $pgid (no slurm)"
  rm -f "/root/glm53-1d3m-262k-20260909/.devbox_up/trainer.pid"
else
  echo "no scancel and no /root/glm53-1d3m-262k-20260909/.devbox_up/trainer.pid — nothing to stop"
fi
sleep 10
echo "--- GPU state after stop ---"
if command -v srun >/dev/null 2>&1; then
  timeout 30 srun --immediate=20 --nodes=1 --ntasks=1 --ntasks-per-node=1 \
    --cpus-per-task=1 --mem=1G --label bash -lc \
    'printf "host=%s\n" "$(hostname -s)"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader' \
    || echo "Could not collect GPU state within 30s."
else
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
fi
