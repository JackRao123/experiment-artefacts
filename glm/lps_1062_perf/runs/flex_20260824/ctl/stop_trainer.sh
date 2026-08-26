#!/usr/bin/env bash
set -uo pipefail
job_id=$(squeue -h --name=devbox_trainer -o "%A" | awk 'NR==1 {print $1}')
trainer_nodes=$(cat /root/.cache/user_artifacts/lps1062_flex_20260824/ctl/trainer_num_nodes 2>/dev/null || echo "2")
if [ -n "$job_id" ]; then
  timeout 30 srun --jobid="$job_id" --overlap --nodes="$trainer_nodes" --ntasks="$trainer_nodes" \
    --ntasks-per-node=1 --cpus-per-task=1 --label bash -lc '
    worker_pattern="trainers_server_main."
    worker_pattern+="main"
    torchrun_pattern="bin/"
    torchrun_pattern+="torchrun"
    spawn_pattern="multiprocessing."
    spawn_pattern+="spawn"
    worker_pids=$(pgrep -f "$worker_pattern" || true)
    torchrun_pids=$(pgrep -f "$torchrun_pattern" || true)
    spawn_pids=$(pgrep -f "$spawn_pattern" || true)
    [ -z "$worker_pids" ] || kill -TERM $worker_pids
    [ -z "$torchrun_pids" ] || kill -TERM $torchrun_pids
    [ -z "$spawn_pids" ] || kill -TERM $spawn_pids
  ' || echo "Could not run the in-allocation orphan sweep within 30s."
fi
scancel --name=devbox_trainer 2>/dev/null && echo "scancel devbox_trainer sent"
sleep 10
echo "--- Slurm after stop ---"
squeue -o "%.18i %.20j %.2t %.10M %.6D %R"
