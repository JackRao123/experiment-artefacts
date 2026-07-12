#!/usr/bin/env bash
# Run ONE profiling datapoint against the live trainer (from the laptop):
#   ./run_point.sh <tag> <seq_len> <dp_size> [steps] [src]
# dp_size datums are sent per /forward_backward (the server shards data across
# the DP group), so each rank forwards exactly ONE seq_len-token sequence.
# - records point start/end epochs (for slicing poller CSVs)
# - runs sft_driver.py --source synthetic on the leader (venv python)
# - saves driver log + /status JSON under glm_prof/results/<tag>/
set -uo pipefail
JOB="${JOB:-qr9oevq}"
TAG="${1:?tag}"; SEQ="${2:?seq_len}"; ND="${3:?dp_size}"; STEPS="${4:-2}"
SRC="${5:-/root/.cache/user_artifacts/trainers_glm}"
# lr=0 by default: GLM-5.2 PP16 LoRA produced NaN grads on the first synthetic
# optim step (weights got poisoned); memory/timing profiles are identical under
# a zero update and the weights stay clean. Override with LR=<val>.
LR="${LR:-0}"
GLMP=/root/.cache/user_artifacts/glm_prof
LEADER="training-job-${JOB}-0.ssh.baseten.co"
RD="$GLMP/results/$TAG"

ssh -o ServerAliveInterval=15 "$LEADER" "
  set -u
  mkdir -p $RD
  start=\$(date +%s)
  echo \"\$start,start,$SEQ\" >> $RD/points.csv
  $SRC/server/.venv/bin/python $GLMP/sft_driver.py \
    --source synthetic --seq-len $SEQ --steps $STEPS \
    --num-datums $ND --microbatch-size $ND --learning-rate $LR \
    2>&1 | tee $RD/driver_S${SEQ}.log
  rc=\${PIPESTATUS[0]}
  $SRC/server/.venv/bin/python -c \"import httpx,sys; sys.stdout.write(httpx.get('http://127.0.0.1:8000/status', timeout=60).text)\" > $RD/status_S${SEQ}.json || true
  end=\$(date +%s)
  echo \"\$end,end,$SEQ,rc=\$rc\" >> $RD/points.csv
  echo \"[run_point] seq=$SEQ rc=\$rc elapsed=\$((end-start))s\"
  exit \$rc
"
