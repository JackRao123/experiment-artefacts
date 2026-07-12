#!/usr/bin/env bash
# Export smoke: boot debug GLM with weight_sync=local, call
# /save_weights_for_sampler, poll the op, tear down.
#   bash export_smoke.sh <cp1|cp2>
set -uo pipefail
VARIANT="${1:?cp1|cp2}"
GLMP=/root/.cache/user_artifacts/glm_prof
SRC=/root/.cache/user_artifacts/trainers_glm_cp
CONFIG="$GLMP/configs/glm52-debug-${VARIANT}-export.json"
LOG="$GLMP/logs/export_smoke_${VARIANT}.log"
rm -rf "$GLMP/export_test/${VARIANT}"
mkdir -p "$GLMP/export_test/${VARIANT}" "$GLMP/logs"

pkill -9 -f "[d]p_worker.main" 2>/dev/null; pkill -9 -f "[t]orchrun" 2>/dev/null; sleep 8

env TRAINERS_SRC=$SRC CONFIG=$CONFIG NUM_NODES=1 NUM_GPUS=2 LEADER_ADDR=127.0.0.1 \
  nohup bash $GLMP/scripts/run_trainer_node.sh 0 > "$LOG" 2>&1 < /dev/null &
echo "export smoke $VARIANT trainer pid $!"

start=$(date +%s)
probe() { $SRC/server/.venv/bin/python -c "import httpx; httpx.get('http://127.0.0.1:8000/health', timeout=5).raise_for_status()" >/dev/null 2>&1; }
until probe; do
  sleep 5
  if ! pgrep -f dp_worker.main >/dev/null; then
    echo "EXPORT SMOKE $VARIANT: trainer died during boot; log tail:"; tail -40 "$LOG"; exit 1
  fi
  [ $(( $(date +%s) - start )) -gt 900 ] && { echo "EXPORT SMOKE $VARIANT: boot timeout"; tail -20 "$LOG"; exit 1; }
done
echo "export smoke $VARIANT healthy after $(( $(date +%s) - start ))s"

$SRC/server/.venv/bin/python - <<'PY'
import httpx, sys, time
base = "http://127.0.0.1:8000"
r = httpx.post(f"{base}/save_weights_for_sampler",
               json={"name": "export-smoke", "run_id": "export-smoke"}, timeout=30)
r.raise_for_status()
data = r.json()
op = data.get("operation_id") or data.get("id")
print("op:", op, data)
deadline = time.time() + 600
while time.time() < deadline:
    s = httpx.get(f"{base}/operations/{op}", timeout=30).json()
    state = s.get("status") or s.get("state")
    if state in ("succeeded", "done", "completed"):
        print("export op finished:", s); sys.exit(0)
    if state in ("failed", "error"):
        print("EXPORT FAILED:", s); sys.exit(1)
    time.sleep(3)
print("EXPORT TIMEOUT"); sys.exit(1)
PY
rc=$?

pkill -9 -f "[d]p_worker.main" 2>/dev/null; pkill -9 -f "[t]orchrun" 2>/dev/null; sleep 10
echo "EXPORT SMOKE $VARIANT rc=$rc"
exit $rc
