#!/usr/bin/env bash
set -uo pipefail
RUN=/root/glm53-full-131k-fsdp-headmem-20260910
LOG="$RUN/.devbox_up/trainer_srun.log"
# Worker processes get their own process groups. Include spawned helpers only
# when their stdout is the exact log owned by this lifecycle directory.
owned_groups() {
  for pid in $(pgrep -f '[t]rainers_server_main.main|[m]ultiprocessing.spawn' || true); do
    [ "$(readlink "/proc/$pid/fd/1" 2>/dev/null)" = "$LOG" ] || continue
    ps -o pgid= -p "$pid" 2>/dev/null
  done | awk '$1 > 1 {print $1}' | sort -u
}
groups=$(owned_groups)
if [ -f "$RUN/.devbox_up/trainer.pid" ]; then
  pgid=$(cat "$RUN/.devbox_up/trainer.pid")
  kill -TERM -- "-$pgid" 2>/dev/null || true
  rm -f "$RUN/.devbox_up/trainer.pid"
fi
for pgid in $groups; do
  kill -TERM -- "-$pgid" 2>/dev/null || true
done
echo "SIGTERM sent to owned trainer/worker groups: $groups"
sleep 40
for pgid in $(owned_groups); do
  echo "SIGKILL for owned group still alive after grace period: $pgid"
  kill -KILL -- "-$pgid" 2>/dev/null || true
done
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader || true
