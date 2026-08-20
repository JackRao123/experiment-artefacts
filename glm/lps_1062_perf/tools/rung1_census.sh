#!/usr/bin/env bash
# rung1_census.sh — the rung-1 measurement: two identical d2 runs (conway).
#
# Run A is the census (allocator snapshots on all 16 ranks + traces on the two
# pipeline-stage leaders). Run B is the identical repeat: a second plateau
# sample for memory, and — against run A's matched windows — the stepping-side
# noise floor that every later arm is judged relative to.
#
# Run length is the plateau rule, not a guess: torch-reserved creeps ~+21 GiB
# over the early steps before flattening and declares by about step 9.
# --control-repeats 10 gives 12 steps (warmup=1, traced=2, controls=3..12), so
# the last two controls sit past the plateau with margin.
set -euo pipefail

ART=/root/.cache/user_artifacts
PP2=$ART/lps1062_pp2
PY=$ART/trainers_main/server-megatron-bridge/.venv/bin/python
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

banner() { printf '\n=== %s ===\n' "$*"; }

# Any srun that has to run ALONGSIDE the trainer must attach to the trainer's
# allocation with --jobid. A bare `srun --overlap` queues behind it forever
# (the trainer holds both nodes), which looks exactly like a hang.
JID=$(squeue -h --name=devbox_trainer -o "%A" | head -1)
[ -n "$JID" ] || { echo "!! no devbox_trainer job — is the trainer up?"; exit 1; }
echo "attaching side-tasks to trainer job $JID"
SR="srun --jobid=$JID --overlap"

banner "0. per-GPU memory pollers on both nodes"
# nvidia-smi reserved — the OOM-relevant metric. NEVER mixed against the
# torch-reserved figure from /status; they differ by ~12 GiB of NCCL/driver
# overhead. One sampler per node via srun, CSVs on the shared FS.
mkdir -p "$PP2/mem"
$SR -N2 -n2 bash "$PP2/poll_gpu_mem.sh" "$PP2/mem" &
POLLER_SRUN=$!
echo "poller srun pid $POLLER_SRUN"

banner "1. census run A (d2, 12 steps, allocator snapshots + rank 0/8 traces)"
$PY "$PP2/profile_driver_new.py" --label "rung1A-qed7z1w-d2-$STAMP" \
  --datums 2 --control-repeats 10

banner "2. preserve run A's artifacts BEFORE run B wipes them"
# memory_profile/start wipes the pickle dir on each node and
# runtime_profile/start clears the box trace dir — the campaign's
# pull-immediately rule.
$SR -N2 -n2 --label bash -c \
  "mkdir -p $PP2/artifacts/runA/\$(hostname -s); \
   cp -a /tmp/checkpoints/profiles/latest/memory/. $PP2/artifacts/runA/\$(hostname -s)/ 2>/dev/null || true; \
   cp -a /tmp/checkpoints/profiles/torch_trace/. $PP2/artifacts/runA/\$(hostname -s)/ 2>/dev/null || true; \
   ls $PP2/artifacts/runA/\$(hostname -s) | wc -l"

banner "3. census run B — identical config, seed and data order"
$PY "$PP2/profile_driver_new.py" --label "rung1B-qed7z1w-d2-$STAMP" \
  --datums 2 --control-repeats 10

banner "4. preserve run B's artifacts"
$SR -N2 -n2 --label bash -c \
  "mkdir -p $PP2/artifacts/runB/\$(hostname -s); \
   cp -a /tmp/checkpoints/profiles/latest/memory/. $PP2/artifacts/runB/\$(hostname -s)/ 2>/dev/null || true; \
   cp -a /tmp/checkpoints/profiles/torch_trace/. $PP2/artifacts/runB/\$(hostname -s)/ 2>/dev/null || true; \
   ls $PP2/artifacts/runB/\$(hostname -s) | wc -l"

banner "5. stop the pollers"
kill "$POLLER_SRUN" 2>/dev/null || true
scancel --name=poll_gpu_mem 2>/dev/null || true
pkill -f "[p]oll_gpu_mem.sh" 2>/dev/null || true
wc -l "$PP2"/mem/mem.*.csv

echo
echo "STAMP=$STAMP"
