#!/bin/bash
# run_bench2.sh <label> [bench_driver2 args...] — bench with per-GPU memory
# polling on every trainer node, folding max-mem into the result json.
# Overnight kit version: works with any node count (incl. --num-nodes 1 trainers).
set -uo pipefail
LABEL="$1"; shift || true
BASE=/root/.cache/user_artifacts/lps1062
OUT=/root/.cache/user_artifacts/lps1062_bench
MEMDIR="$OUT/${LABEL}_mem"
PYBIN=/root/.cache/user_artifacts/trainers_main/server/.venv/bin/python
mkdir -p "$MEMDIR"

TRAINER_JOBID="$(squeue -h -n devbox_trainer -o %i | head -1)"
# Node count of the trainer's own allocation (handles --num-nodes 1 trainers).
NNODES="$(squeue -h -j "$TRAINER_JOBID" -o %D 2>/dev/null || true)"
NNODES="${NNODES:-${BT_GROUP_SIZE:-2}}"
if [ -z "$TRAINER_JOBID" ]; then
  echo "WARNING: no devbox_trainer Slurm job found; skipping mem pollers" >&2
else
  # Poll every node in the trainer's allocation (works for 1-node trainers too).
  srun --jobid="$TRAINER_JOBID" --overlap -N "$NNODES" -n "$NNODES" \
    --ntasks-per-node=1 --cpus-per-task=1 \
    bash "$BASE/poll_gpu_mem.sh" "$MEMDIR" \
    < /dev/null > "$MEMDIR/poller.log" 2>&1 &
  POLL_PID=$!
fi

cd "$BASE" && "$PYBIN" bench_driver2.py --label "$LABEL" "$@"
rc=$?

if [ -n "$TRAINER_JOBID" ]; then
  kill "$POLL_PID" 2>/dev/null || true
  srun --jobid="$TRAINER_JOBID" --overlap -N "$NNODES" -n "$NNODES" \
    --ntasks-per-node=1 --cpus-per-task=1 \
    bash -c 'pkill -f "[p]oll_gpu_mem" || true' < /dev/null || true
fi
"$PYBIN" "$BASE/fold_mem.py" "$LABEL"
exit "$rc"
