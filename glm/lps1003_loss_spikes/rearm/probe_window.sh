#!/usr/bin/env bash
# Wait for trainer health on localhost:8001, then immediately probe the boot
# window: soak.py x8 reps of the ctrl payload. Run on the rank0 (server) node.
set -uo pipefail
LPS=/root/.cache/user_artifacts/lps1003
OUT="${1:?usage: probe_window.sh OUTDIR [REPS]}"
REPS="${2:-8}"
mkdir -p "$OUT"
echo "[$(date +%T)] waiting for health"
for i in $(seq 1 300); do
  code=$(curl -s -m 5 -o /dev/null -w "%{http_code}" http://127.0.0.1:8001/health || true)
  [ "$code" = 200 ] && break
  sleep 5
done
[ "$code" = 200 ] || { echo "health never came up"; exit 1; }
echo "[$(date +%T)] HEALTH OK - probing window ($REPS reps)"
python3 "$LPS/ctrl/soak.py" "$LPS/ctrl/payload_b0_part1_uniform.json" \
  http://127.0.0.1:8001 "$OUT" "$REPS"
echo "[$(date +%T)] window probe done"
