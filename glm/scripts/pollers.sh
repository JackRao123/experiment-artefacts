#!/usr/bin/env bash
# Laptop-side start/stop for the per-node 1 Hz memory pollers on qr9oevq.
#   ./pollers.sh start | stop | status
set -uo pipefail
JOB="${JOB:-qr9oevq}"
NODES=(0 1 2 3)
GLMP=/root/.cache/user_artifacts/glm_prof
host() { if [ "$1" = 0 ]; then echo "training-job-${JOB}-0.ssh.baseten.co"; else echo "training-job-${JOB}-$1.ssh.baseten.co"; fi; }

case "${1:?usage: pollers.sh start|stop|status}" in
  start)
    for n in "${NODES[@]}"; do
      ssh "$(host "$n")" "pkill -f poll_mem.sh 2>/dev/null; mkdir -p $GLMP/poll && nohup bash $GLMP/scripts/poll_mem.sh > /dev/null 2>&1 < /dev/null & echo rank$n poller pid \$!"
    done ;;
  stop)
    for n in "${NODES[@]}"; do ssh "$(host "$n")" 'pkill -f poll_mem.sh; true'; echo "rank$n poller stopped"; done ;;
  status)
    for n in "${NODES[@]}"; do
      ssh "$(host "$n")" "printf 'rank$n: procs=%s rows=%s\n' \"\$(pgrep -cf poll_mem.sh)\" \"\$(wc -l < $GLMP/poll/\$(hostname -s).csv 2>/dev/null || echo 0)\""
    done ;;
esac
