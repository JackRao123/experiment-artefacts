#!/bin/bash
# LPS-1003 parity, prod arm: run INSIDE the trainer pod (multinode-0) via
# nohup. Waits for /health (bounded), fires window probes immediately, then
# steady-state probes, then the env fingerprint.
set -u
OUT=/tmp/parity
mkdir -p "$OUT"
cd "$OUT"
exec >>"$OUT/driver.log" 2>&1
echo "=== prod parity driver start $(date -u +%FT%TZ)"

# P0: boot-window probes (prior windows fired 4/4 within minutes of READY)
python3 "$OUT/probe_lp.py" "$OUT/probe_batch0_forward.json" \
  --url http://127.0.0.1:8000 --reps 8 --tag p0_window --out "$OUT" --wait-health 3600 \
  || { echo "P0 FAILED"; exit 2; }

# P1: steady state on the identical payload (weights frozen; B=0 adapter)
python3 "$OUT/probe_lp.py" "$OUT/probe_batch0_forward.json" \
  --url http://127.0.0.1:8000 --reps 8 --tag p1_steady --out "$OUT"

# fingerprint: discover the live trainer interpreter for package versions
PID=""
for d in /proc/[0-9]*; do
  if grep -aq "dp_worker.main" "$d/cmdline" 2>/dev/null; then PID=${d#/proc/}; break; fi
done
PY=""
[ -n "$PID" ] && PY=$(readlink -f "/proc/$PID/exe")
echo "trainer pid=$PID python=$PY"
python3 "$OUT/fingerprint.py" ${PY:+--python "$PY"} --out "$OUT/fingerprint.json"

echo "=== prod parity driver DONE $(date -u +%FT%TZ)"
