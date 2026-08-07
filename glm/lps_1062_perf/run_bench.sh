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

srun --overlap --nodes="${BT_GROUP_SIZE:-2}" --ntasks="${BT_GROUP_SIZE:-2}" \
  --ntasks-per-node=1 bash "$BASE/poll_gpu_mem.sh" "$MEMDIR" \
  < /dev/null > "$MEMDIR/poller.log" 2>&1 &
POLL_PID=$!

"$PYBIN" "$BASE/bench_driver.py" --label "$LABEL" "$@"
rc=$?

kill "$POLL_PID" 2>/dev/null || true
srun --overlap --nodes="${BT_GROUP_SIZE:-2}" --ntasks="${BT_GROUP_SIZE:-2}" \
  --ntasks-per-node=1 bash -c 'pkill -f "[p]oll_gpu_mem" || true' < /dev/null
"$PYBIN" "$BASE/fold_mem.py" "$LABEL"
exit "$rc"
