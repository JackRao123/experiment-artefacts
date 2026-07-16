#!/usr/bin/env bash
# GLM debug-model DPO parity smoke. Boots CP1 or CP2 on two B200s and tears
# down the local trainer after recording loss, logprobs, and gradient norm.
set -uo pipefail

VARIANT="${1:?usage: smoke_cp_dpo.sh <cp1|cp2>}"
GLMP="${GLMP:-/root/.cache/user_artifacts/glm_prof}"
SRC="${SRC:-/root/.cache/user_artifacts/trainers_glm_dpo}"
CONFIG="$GLMP/configs/glm52-debug-${VARIANT}-8k.json"
RESULT_DIR="$GLMP/results/dpo_parity"
LOG="$GLMP/logs/smoke_dpo_${VARIANT}.log"
mkdir -p "$RESULT_DIR" "$GLMP/logs"

pkill -9 -f "[d]p_worker.main" 2>/dev/null || true
pkill -9 -f "[t]orchrun" 2>/dev/null || true
sleep 8

env TRAINERS_SRC="$SRC" CONFIG="$CONFIG" NUM_NODES=1 NUM_GPUS=2 \
  LEADER_ADDR=127.0.0.1 \
  nohup bash "$GLMP/scripts/run_trainer_node.sh" 0 >"$LOG" 2>&1 </dev/null &
echo "smoke_dpo $VARIANT trainer pid $!"

start=$(date +%s)
probe() {
  "$SRC/server/.venv/bin/python" -c \
    "import httpx; httpx.get('http://127.0.0.1:8000/health', timeout=5).raise_for_status()" \
    >/dev/null 2>&1
}
until probe; do
  sleep 5
  if ! pgrep -f "[d]p_worker.main" >/dev/null; then
    echo "SMOKE_DPO $VARIANT: trainer died during boot"
    tail -80 "$LOG"
    exit 1
  fi
  if [ $(( $(date +%s) - start )) -gt 900 ]; then
    echo "SMOKE_DPO $VARIANT: boot timeout"
    tail -80 "$LOG"
    exit 1
  fi
done
echo "smoke_dpo $VARIANT healthy after $(( $(date +%s) - start ))s"

"$SRC/server/.venv/bin/python" "$GLMP/scripts/cp_dpo_parity.py" \
  --out "$RESULT_DIR/${VARIANT}.json" 2>&1 |
  tee "$RESULT_DIR/${VARIANT}_driver.log"
rc=${PIPESTATUS[0]}

pkill -9 -f "[d]p_worker.main" 2>/dev/null || true
pkill -9 -f "[t]orchrun" 2>/dev/null || true
sleep 10
echo "SMOKE_DPO $VARIANT rc=$rc"
exit "$rc"
