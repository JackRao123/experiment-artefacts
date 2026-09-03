#!/usr/bin/env bash
set -uo pipefail

scancel --name=glm52_native_phase2 2>/dev/null && echo "stop sent"
sleep 10
echo "--- Slurm ---"
squeue --name=glm52_native_phase2 -o "%.18i %.9T %.8M %.20j %.40R" || true
echo "--- GPU state ---"
timeout 30 srun --immediate=20 --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task=1 --mem=1G --label bash -lc \
  'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader' \
  || echo "Could not collect GPU state within 30s"
