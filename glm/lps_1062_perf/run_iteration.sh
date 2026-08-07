#!/bin/bash
# LPS-1062 opt-night iteration driver (laptop side).
# Usage: run_iteration.sh <tj-alias> <push|restart|wait|bench|status> [config-name] [--hw-passes N]
#   push    <config-name>  scp config + bench driver to the box
#   restart <config-name>  stop trainer, verify clean, start with config
#   wait                   run one wait_trainer_health.sh cycle (exits ~3 min, prints diagnostics)
#   bench   <label> [...]  run bench_driver.py, fetch results json into results/
#   status                 quick /health + GPU snapshot
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
JOB="$1"; CMD="$2"; ARG="${3:-}"
REMOTE_DIR=/root/.cache/user_artifacts/lps1062
ENVSH="source /root/.cache/user_artifacts/env.sh"
DBX=/root/.cache/user_artifacts/.devbox_up

rssh() { # retry ssh against relay throttling (instant rc=255, no output)
  local tries=0
  until ssh -o ConnectTimeout=20 "$JOB" "$@"; do
    rc=$?; tries=$((tries+1))
    [ $rc -ne 255 ] || [ $tries -ge 5 ] && [ $rc -ne 255 ] && return $rc
    [ $tries -ge 5 ] && return $rc
    sleep 14
  done
}

case "$CMD" in
  push)
    [ -n "$ARG" ] || { echo "config name required"; exit 2; }
    rssh "mkdir -p $REMOTE_DIR"
    scp "$HERE/configs/$ARG.json" "$JOB:$REMOTE_DIR/trainer-config.json"
    scp "$HERE/configs/server-config.json" "$JOB:$REMOTE_DIR/trainer-server-config.json"
    scp "$HERE/bench_driver.py" "$HERE/test_tf32_head_parity.py" "$JOB:$REMOTE_DIR/"
    echo "PUSHED $ARG"
    ;;
  restart)
    [ -n "$ARG" ] || { echo "config name required"; exit 2; }
    "$0" "$JOB" push "$ARG"
    rssh "bash $DBX/stop_trainer.sh" || true
    rssh "srun --overlap --nodes=\$BT_GROUP_SIZE --ntasks=\$BT_GROUP_SIZE --ntasks-per-node=1 bash -lc 'printf \"%s procs=%s gpu_max=%sMiB\n\" \"\$(hostname)\" \"\$(pgrep -fc \"[d]p_worker.main|[t]orchrun\" || true)\" \"\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)\"'"
    rssh "$ENVSH && export BT_TRAINER_CONFIG_PATH=$REMOTE_DIR/trainer-config.json BT_TRAINER_SERVER_CONFIG_PATH=$REMOTE_DIR/trainer-server-config.json HF_HUB_OFFLINE=\${HF_HUB_OFFLINE:-0} && bash $DBX/start_trainer.sh" < /dev/null
    echo "STARTED with $ARG"
    ;;
  wait)
    rssh "bash $DBX/wait_trainer_health.sh" || true
    ;;
  bench)
    [ -n "$ARG" ] || { echo "label required"; exit 2; }
    shift 3 || true
    rssh "$ENVSH && cd $REMOTE_DIR && python3 bench_driver.py --label $ARG $*" < /dev/null
    mkdir -p "$HERE/results"
    scp "$JOB:/root/.cache/user_artifacts/lps1062_bench/$ARG.json" "$HERE/results/" && echo "FETCHED results/$ARG.json"
    ;;
  status)
    rssh "curl -s -o /dev/null -w 'health=%{http_code}\n' -m 5 http://127.0.0.1:8001/health; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -8"
    ;;
  *) echo "unknown cmd $CMD"; exit 2 ;;
esac
