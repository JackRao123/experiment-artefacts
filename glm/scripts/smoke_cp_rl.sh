#!/usr/bin/env bash
# CP-RL smoke: GLM debug model, single node (leader), CP1 vs CP2.
# Boots the branch stack standalone with 2 GPUs, runs cp_rl_parity.py
# (all 6 loss_fns + logprobs + grad_norm), tears down.
#   bash smoke_cp_rl.sh <cp1|cp2>
set -uo pipefail
VARIANT="${1:?cp1|cp2}"
GLMP=/root/.cache/user_artifacts/glm_prof
SRC=/root/.cache/user_artifacts/trainers_glm_cp
CONFIG="$GLMP/configs/glm52-debug-${VARIANT}-8k.json"
RD="$GLMP/results/rl_parity"
LOG="$GLMP/logs/smoke_rl_${VARIANT}.log"
mkdir -p "$RD" "$GLMP/logs"

pkill -9 -f "[d]p_worker.main" 2>/dev/null; pkill -9 -f "[t]orchrun" 2>/dev/null; sleep 8

env TRAINERS_SRC=$SRC CONFIG=$CONFIG NUM_NODES=1 NUM_GPUS=2 LEADER_ADDR=127.0.0.1 \
  nohup bash $GLMP/scripts/run_trainer_node.sh 0 > "$LOG" 2>&1 < /dev/null &
echo "smoke_rl $VARIANT trainer pid $!"

start=$(date +%s)
probe() { $SRC/server/.venv/bin/python -c "import httpx; httpx.get('http://127.0.0.1:8000/health', timeout=5).raise_for_status()" >/dev/null 2>&1; }
until probe; do
  sleep 5
  if ! pgrep -f dp_worker.main >/dev/null; then
    echo "SMOKE_RL $VARIANT: trainer died during boot; log tail:"; tail -40 "$LOG"; exit 1
  fi
  [ $(( $(date +%s) - start )) -gt 900 ] && { echo "SMOKE_RL $VARIANT: boot timeout"; tail -20 "$LOG"; exit 1; }
done
echo "smoke_rl $VARIANT healthy after $(( $(date +%s) - start ))s"

$SRC/server/.venv/bin/python $GLMP/scripts/cp_rl_parity.py \
  --out "$RD/${VARIANT}.json" 2>&1 | tee "$RD/${VARIANT}_driver.log"
rc=${PIPESTATUS[0]}

pkill -9 -f "[d]p_worker.main" 2>/dev/null; pkill -9 -f "[t]orchrun" 2>/dev/null; sleep 10
echo "SMOKE_RL $VARIANT rc=$rc"
exit $rc
