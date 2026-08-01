#!/usr/bin/env bash
# LPS-1003 unattended attribution chain: run devbox_attrib.sh arms
# sequentially (each boots, probes the window, stops the trainer, verifies).
# usage: attrib_chain.sh "conn:2 gloo:0 arch:0 nvte:0 penvconn:0"
set -uo pipefail
ARMS="${1:?arm list like 'conn:2 gloo:0'}"
PAR=/root/.cache/user_artifacts/lps1003/parity
LOG=$PAR/runs/attrib_chain_$(date +%m%d_%H%M%S).log
exec >>"$LOG" 2>&1
echo "=== chain start $(date -u +%FT%TZ): $ARMS"
for arm in $ARMS; do
  mode="${arm%%:*}"; trace="${arm##*:}"
  # wait for any running trainer to clear (bounded 10 min)
  for i in $(seq 1 180); do
    squeue -h --name=devbox_trainer -o %A | grep -q . || break
    sleep 10
  done
  if squeue -h --name=devbox_trainer -o %A | grep -q .; then
    echo "TRAINER STILL RUNNING before arm $mode — aborting chain"; exit 1
  fi
  echo "=== arm $mode (trace=$trace) start $(date -u +%FT%TZ)"
  bash "$PAR/devbox_attrib.sh" "$mode" "$trace"
  echo "=== arm $mode done $(date -u +%FT%TZ)"
  sleep 15
done
echo "=== chain DONE $(date -u +%FT%TZ)"
