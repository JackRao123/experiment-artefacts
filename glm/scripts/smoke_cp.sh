#!/usr/bin/env bash
# Phase C smoke: GLM debug model, single node (leader), CP1 vs CP2.
# Runs ON the leader. Boots the S2 (CP) stack standalone with 2 GPUs,
# drives 2 synthetic steps at a fixed seq, records loss + /status, tears down.
#   bash smoke_cp.sh <cp1|cp2> <seq_len>
set -uo pipefail
VARIANT="${1:?cp1|cp2}"; SEQ="${2:-4096}"
GLMP=/root/.cache/user_artifacts/glm_prof
SRC=/root/.cache/user_artifacts/trainers_glm_cp
CONFIG="$GLMP/configs/glm52-debug-${VARIANT}-8k.json"
RD="$GLMP/results/smoke_${VARIANT}_S${SEQ}"
LOG="$GLMP/logs/smoke_${VARIANT}.log"
mkdir -p "$RD"

pkill -9 -f "[d]p_worker.main" 2>/dev/null; pkill -9 -f "[t]orchrun" 2>/dev/null; sleep 8

env TRAINERS_SRC=$SRC CONFIG=$CONFIG NUM_NODES=1 NUM_GPUS=2 LEADER_ADDR=127.0.0.1 \
  nohup bash $GLMP/scripts/run_trainer_node.sh 0 > "$LOG" 2>&1 < /dev/null &
echo "smoke $VARIANT trainer pid $!"

start=$(date +%s)
probe() { $SRC/server/.venv/bin/python -c "import httpx; httpx.get('http://127.0.0.1:8000/health', timeout=5).raise_for_status()" >/dev/null 2>&1; }
until probe; do
  sleep 5
  if ! pgrep -f dp_worker.main >/dev/null; then
    echo "SMOKE $VARIANT: trainer died during boot; log tail:"; tail -40 "$LOG"; exit 1
  fi
  [ $(( $(date +%s) - start )) -gt 900 ] && { echo "SMOKE $VARIANT: boot timeout"; tail -20 "$LOG"; exit 1; }
done
echo "smoke $VARIANT healthy after $(( $(date +%s) - start ))s"

$SRC/server/.venv/bin/python $GLMP/sft_driver.py \
  --source synthetic --seq-len "$SEQ" --steps 2 --num-datums 1 --microbatch-size 1 \
  --skip-optim --synthetic-vocab 1800 \
  2>&1 | tee "$RD/driver.log"
rc=${PIPESTATUS[0]}
curl -s -m30 http://127.0.0.1:8000/status > "$RD/status.json" || true

pkill -9 -f "[d]p_worker.main" 2>/dev/null; pkill -9 -f "[t]orchrun" 2>/dev/null; sleep 10
echo "SMOKE $VARIANT rc=$rc"
exit $rc
