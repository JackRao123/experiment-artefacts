#!/bin/bash
# run_bench.sh <label> [bench_driver args...] — run one apples-to-apples bench
# with per-GPU memory polling on every node, folding max-mem into the result.
set -uo pipefail
LABEL="$1"; shift || true
BASE=/root/.cache/user_artifacts/lps1062
OUT=/root/.cache/user_artifacts/lps1062_bench
MEMDIR="$OUT/${LABEL}_mem"
PYBIN=/root/.cache/user_artifacts/trainers_main/server/.venv/bin/python
mkdir -p "$MEMDIR"

# Attach pollers to the trainer's Slurm allocation (a fresh srun job queues
# behind it forever — see wait_trainer_health.sh, which does the same).
TRAINER_JOBID="$(squeue -h -n devbox_trainer -o %i | head -1)"
if [ -z "$TRAINER_JOBID" ]; then
  echo "WARNING: no devbox_trainer Slurm job found; skipping mem pollers" >&2
else
  srun --jobid="$TRAINER_JOBID" --overlap --nodes="${BT_GROUP_SIZE:-2}" \
    --ntasks="${BT_GROUP_SIZE:-2}" --ntasks-per-node=1 --cpus-per-task=1 \
    bash "$BASE/poll_gpu_mem.sh" "$MEMDIR" \
    < /dev/null > "$MEMDIR/poller.log" 2>&1 &
  POLL_PID=$!
fi

"$PYBIN" "$BASE/bench_driver.py" --label "$LABEL" "$@"
rc=$?

if [ -n "$TRAINER_JOBID" ]; then
  kill "$POLL_PID" 2>/dev/null || true
  timeout 60 srun --jobid="$TRAINER_JOBID" --overlap \
    --nodes="${BT_GROUP_SIZE:-2}" --ntasks="${BT_GROUP_SIZE:-2}" \
    --ntasks-per-node=1 --cpus-per-task=1 \
    bash -c 'pkill -f "[p]oll_gpu_mem" || true' < /dev/null
fi
"$PYBIN" "$BASE/fold_mem.py" "$LABEL"
exit "$rc"
