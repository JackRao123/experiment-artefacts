#!/usr/bin/env bash
# Pinned health waiter. Same 3-minute checkpoint discipline as the devbox-up
# copy; two fixes from conway: our log path, and the worker process name, which
# the #1027 restructure changed from dp_worker.main to trainers_server_main.main
# (the stock waiter would report "TRAINER PROCESS DIED" during a healthy boot).
#
# turing, 2026-08-21 — TWO MORE FIXES, both earned the hard way. The guided
# config's workers all died at ~01:31Z and this script reported "not up yet"
# for the next eight minutes, because:
#
#   1. The gone-check required BOTH no worker process AND no slurm job. srun
#      kept the devbox_trainer job listed for minutes after every worker was
#      dead, so the check never fired. Now: once workers have been SEEN, their
#      later disappearance is death on its own, whatever squeue says.
#   2. Nothing looked at the log. A Python traceback sat in trainer_srun.log
#      while this script cheerfully polled /health. Now every poll scans the
#      log for failure signatures and reports the moment one lands.
#
# Exit codes are distinguished so callers can tell "still loading" from "dead":
#   0 = healthy
#   1 = dead (log signature, or workers vanished)
#   2 = checkpoint reached, still loading — re-invoke
set -uo pipefail
PP2=/root/.cache/user_artifacts/lps1062_pp2
LOG=$PP2/logs/trainer_srun.log
timeout_s=${WAIT_TIMEOUT_S:-180}
health_url="${TRAINER_HEALTH_URL:-http://127.0.0.1:8001}"
start=$(date +%s)
seen_workers=0

# Deliberately specific. Bare "RuntimeError"/"ValueError" would match warning
# text in a healthy boot; these only appear when something actually died.
FAIL_RE='Traceback \(most recent call last\)|torch\.OutOfMemoryError|CUDA out of memory|Exited with exit code|Signal [0-9]+ \(SIG|NCCL error|Watchdog caught collective operation timeout'

report_log_failure() {
  local elapsed=$1
  echo "TRAINER FAILED after ${elapsed}s — log signature matched:"
  grep -anE "$FAIL_RE" "$LOG" 2>/dev/null | head -5
  echo "--- first traceback ---"
  local n
  n=$(grep -anE 'Traceback \(most recent call last\)' "$LOG" 2>/dev/null | head -1 | cut -d: -f1)
  if [ -n "${n:-}" ]; then
    sed -n "${n},$((n+40))p" "$LOG"
  else
    tail -n 40 "$LOG"
  fi
}

while ! curl -fs -m5 "$health_url/health" >/dev/null; do
  elapsed=$(( $(date +%s) - start ))
  # pgrep -c PRINTS 0 and also exits non-zero when there is no match, so a
  # `|| echo 0` fallback yields the string "0\n0" and every [ -gt ] below
  # dies with "integer expression expected". `|| true` keeps the printed 0.
  nworkers=$(pgrep -cf "[t]rainers_server_main.main" 2>/dev/null || true)
  nworkers=${nworkers:-0}
  [ "${nworkers:-0}" -gt 0 ] && seen_workers=1

  # FIX 2 — a traceback in the log is death, and we find it on this poll.
  if grep -aqE "$FAIL_RE" "$LOG" 2>/dev/null; then
    report_log_failure "$elapsed"
    exit 1
  fi

  # FIX 1 — workers existed and now do not. Do not wait on squeue to agree.
  if [ "$seen_workers" -eq 1 ] && [ "${nworkers:-0}" -eq 0 ]; then
    echo "TRAINER GONE after ${elapsed}s (workers vanished; slurm may still list the job)"
    echo "--- log tail ---"; tail -n 40 "$LOG" || true
    exit 1
  fi

  if [ "${nworkers:-0}" -eq 0 ] && ! squeue -h --name=devbox_trainer | grep -q .; then
    echo "TRAINER GONE (no worker process, no slurm job) — log tail:"
    tail -n 40 "$LOG" || true
    exit 1
  fi

  if [ "$elapsed" -ge "$timeout_s" ]; then
    echo "CHECKPOINT: /health not up after ${timeout_s}s (weight load is 13-20 min on a warm cache)."
    echo "workers alive: ${nworkers}"
    echo "--- trainer log (last 20 lines) ---"; tail -n 20 "$LOG" || true
    echo "--- slurm ---"; squeue --name=devbox_trainer -o "%.18i %.9T %.8M %.20j %.40R" || true
    exit 2
  fi
  sleep 5
done
echo "trainer healthy after $(( $(date +%s) - start ))s"
