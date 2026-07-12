#!/usr/bin/env bash
# Leader-side sweep loop: runs sft_driver points back-to-back against the live
# trainer, saving driver logs + /status per point. Survive-ssh: launch under
# nohup. Usage (ON the leader):
#   bash sweep_local.sh <tag> <dp_size> <steps> <src> <seq1> [seq2 ...]
set -uo pipefail
TAG="${1:?tag}"; ND="${2:?dp}"; STEPS="${3:?steps}"; SRC="${4:?src}"; shift 4
GLMP=/root/.cache/user_artifacts/glm_prof
RD="$GLMP/results/$TAG"
LR="${LR:-0}"
mkdir -p "$RD"
PY="$SRC/server/.venv/bin/python"

for SEQ in "$@"; do
  start=$(date +%s)
  echo "$start,start,$SEQ" >> "$RD/points.csv"
  echo "[sweep] ===== seq=$SEQ starting at $(date -u +%H:%M:%S) ====="
  $PY $GLMP/sft_driver.py \
    --source synthetic --seq-len "$SEQ" --steps "$STEPS" \
    --num-datums "$ND" --microbatch-size "$ND" --learning-rate "$LR" --skip-optim \
    > "$RD/driver_S${SEQ}.log" 2>&1
  rc=$?
  $PY -c "import httpx,sys; sys.stdout.write(httpx.get('http://127.0.0.1:8000/status', timeout=120).text)" \
    > "$RD/status_S${SEQ}.json" 2>/dev/null || true
  end=$(date +%s)
  echo "$end,end,$SEQ,rc=$rc" >> "$RD/points.csv"
  echo "[sweep] seq=$SEQ rc=$rc elapsed=$((end-start))s peak=$(grep -o 'peak_alloc_max=[0-9.]*GiB' "$RD/driver_S${SEQ}.log" | tail -1)"
  if [ $rc -ne 0 ]; then
    # A dead trainer means every later point would fail too — stop and let the
    # orchestrator decide (an OOM mid-grid is a result, not an error).
    if ! $PY -c "import httpx; httpx.get('http://127.0.0.1:8000/health', timeout=10).raise_for_status()" 2>/dev/null; then
      echo "[sweep] trainer unhealthy after seq=$SEQ — stopping sweep"
      break
    fi
  fi
done
echo "[sweep] DONE tag=$TAG"
