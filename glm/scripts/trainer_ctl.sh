#!/usr/bin/env bash
# trainer_ctl.sh — laptop-side fan-out launcher/teardown for the 4-node GLM-5.2
# profiling trainer on devbox qr9oevq. Adapted from start_trainer_all.sh
# (retrying ssh; leader cannot ssh workers, fan-out must come from the laptop).
#
#   ./trainer_ctl.sh start <config.json-on-shared-fs> [src]   # preflight + launch 4 ranks
#   ./trainer_ctl.sh stop | status | wait-health
#
# SRC selects the stack: /root/.cache/user_artifacts/trainers_glm (S1 baseline)
# or /root/.cache/user_artifacts/trainers_glm_cp (S2 CP stack).
set -uo pipefail

JOB="${JOB:-qr9oevq}"
NODES=(0 1 2 3)
GLMP=/root/.cache/user_artifacts/glm_prof
LOGD="$GLMP/logs"
host() { if [ "$1" = 0 ]; then echo "training-job-${JOB}-0.ssh.baseten.co"; else echo "training-job-${JOB}-$1.ssh.baseten.co"; fi; }

rssh() {
  local n="$1"; shift
  local h; h="$(host "$n")"
  local i
  for i in 1 2 3 4 5 6; do
    if ssh -o ConnectTimeout=25 -o ServerAliveInterval=15 "$h" "$@"; then return 0; fi
    echo "  [rank$n] ssh try $i failed (rc=$?); backoff 12s..." >&2
    sleep 12
  done
  echo "  [rank$n] ssh FAILED after retries" >&2
  return 1
}

cmd_status() {
  for n in "${NODES[@]}"; do
    rssh "$n" 'printf "rank'"$n"': procs=%s gpu_max=%sMiB\n" \
      "$(ps -ef | grep -cE "[d]p_worker.main|[t]orchrun")" \
      "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)"' \
      || echo "rank$n: UNREACHABLE"
  done
}

cmd_stop() {
  for n in "${NODES[@]}"; do
    echo "stopping rank$n ..."
    rssh "$n" 'pkill -9 -f dp_worker.main; pkill -9 -f "distributed.run"; pkill -9 -f torchrun; true' || true
  done
  echo "waiting for GPU drain ..."
  sleep 25
  cmd_status
}

cmd_start() {
  local config="${1:?usage: trainer_ctl.sh start <config.json path on shared fs> [src]}"
  local src="${2:-/root/.cache/user_artifacts/trainers_glm}"
  echo "=== pre-flight: every node must be clean (0 procs, <3GiB GPU) ==="
  local clean=1
  for n in "${NODES[@]}"; do
    local out procs gpu
    out="$(rssh "$n" 'echo "$(ps -ef | grep -cE "[d]p_worker.main|[t]orchrun")|$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)"')" || { echo "rank$n unreachable"; clean=0; continue; }
    out="$(echo "$out" | tr -d "[:space:]")"
    procs="${out%%|*}"; gpu="${out##*|}"
    echo "rank$n: procs=${procs} gpu=${gpu}MiB"
    if [ "${procs:-1}" -gt 0 ] || [ "${gpu:-9999}" -gt 3000 ]; then clean=0; fi
  done
  [ "$clean" = 1 ] || { echo "ABORT: nodes not clean. Run: $0 stop"; exit 1; }

  local leader_ip
  leader_ip="$(rssh 0 'hostname -i | awk "{print \$1}"')" || exit 1
  local ts; ts="$(date +%H%M%S)"
  echo "=== launching 4 ranks (leader $leader_ip:29500, src=$src, config=$config) ==="
  for n in "${NODES[@]}"; do
    rssh "$n" "mkdir -p $LOGD && { [ -f $LOGD/trainer_rank$n.log ] && mv $LOGD/trainer_rank$n.log $LOGD/trainer_rank$n.$ts.old.log; }; nohup env TRAINERS_SRC=$src CONFIG=$config LEADER_ADDR=$leader_ip NUM_NODES=4 bash $GLMP/scripts/run_trainer_node.sh $n > $LOGD/trainer_rank$n.log 2>&1 < /dev/null & echo '  rank$n started pid '\$!" \
      || { echo "FAILED to launch rank$n — others will hang at rendezvous."; exit 1; }
  done
  echo "All 4 dispatched. $0 wait-health to block until ready."
}

cmd_wait_health() {
  rssh 0 'start=$(date +%s); until curl -sf -m5 http://127.0.0.1:8000/health >/dev/null; do
    sleep 10;
    if ! pgrep -f dp_worker.main >/dev/null; then echo "TRAINER PROCESS DIED — tail of leader log:"; tail -30 /root/.cache/user_artifacts/glm_prof/logs/trainer_rank0.log; exit 1; fi
  done; echo "healthy after $(( $(date +%s) - start ))s (since wait start)"'
}

case "${1:-status}" in
  stop|down)    cmd_stop ;;
  status)       cmd_status ;;
  start|up)     shift; cmd_start "$@" ;;
  wait-health)  cmd_wait_health ;;
  *) echo "usage: $0 {start <config> [src]|stop|status|wait-health}"; exit 2 ;;
esac
