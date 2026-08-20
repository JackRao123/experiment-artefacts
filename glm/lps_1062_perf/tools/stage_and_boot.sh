#!/usr/bin/env bash
# stage_and_boot.sh — one deterministic pre-boot sequence for every rung of the
# LPS-1062 activation-placement ladder (conway, 2026-08-20).
#
#   bash stage_and_boot.sh <config-basename-in-$PP2>
#
# e.g.  bash stage_and_boot.sh trainer_pp2cp8ep8_131k.json                     # rung 1 baseline
#       bash stage_and_boot.sh trainer_pp2cp8ep8_131k_selective_offload_moe_act.json   # rung 3a
#
# For the OFFLOAD arms, set the offload env before calling (the launcher
# environment is load-bearing — see below):
#   NVTE_CPU_OFFLOAD_V1=1 BT_OFFLOAD_VALVE_TELEMETRY=1 \
#   BT_OFFLOAD_VALVE_TELEMETRY_EVERY=1 bash stage_and_boot.sh <arm config>
#
# It stops after dispatch. Health is checked ONLY with wait_trainer_health_pp2.sh,
# run in the background and re-invoked at each checkpoint — never a blocking loop.
set -euo pipefail

ART=/root/.cache/user_artifacts
PP2=$ART/lps1062_pp2
CFG_NAME=${1:?usage: stage_and_boot.sh <config-basename-in-$PP2>}
SRC=$PP2/$CFG_NAME

banner() { printf '\n=== %s ===\n' "$*"; }

banner "0. config: $CFG_NAME"
test -s "$SRC" || { echo "!! $SRC missing or empty"; exit 1; }
python3 -c "import json,sys; d=json.load(open('$SRC')); print(json.dumps(d, indent=2))"

banner "1. squeue must be empty of devbox_trainer"
squeue || true
if squeue -h --name=devbox_trainer | grep -q .; then
  echo "!! a devbox_trainer job is still queued/running — drain it first (stop_trainer.sh, then wait)"
  exit 1
fi

banner "2. stale-process sweep + GPU state, BOTH nodes"
# stop_trainer.sh can leave orphaned workers holding GPU memory; an orphan OOMs
# the next boot for a phantom reason. The srun sweep is the real check (the
# stock GPU-state collection races teardown). Process name is post-#1027.
srun --overlap -N2 -n2 --label bash -c \
  'pkill -f "[t]rainers_server_main.main" 2>/dev/null; pkill -f "[d]p_worker.main" 2>/dev/null; sleep 2;
   pgrep -af "[t]rainers_server_main|[d]p_worker" || echo clean;
   nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | paste -sd" | "' || true

banner "3. stage config to the boot paths"
cp "$SRC" "$ART/trainer_config.json"
cp "$PP2/trainer_server.json" "$ART/trainer_server_config.json"
export BT_TRAINER_CONFIG_PATH=$ART/trainer_config.json
export BT_TRAINER_SERVER_CONFIG_PATH=$ART/trainer_server_config.json
printf 'export BT_TRAINER_CONFIG_PATH=%s\nexport BT_TRAINER_SERVER_CONFIG_PATH=%s\n' \
  "$BT_TRAINER_CONFIG_PATH" "$BT_TRAINER_SERVER_CONFIG_PATH" > "$PP2/trainer_env.sh"

banner "4. staged-file integrity from BOTH nodes (CPFS close-to-open quirk)"
# Files written on the leader can read back as all-NULs from the sibling node.
LEADER_SUM=$(sha256sum "$ART/trainer_config.json" | cut -d' ' -f1)
echo "leader: $LEADER_SUM"
srun --overlap -N2 -n2 --label bash -c "sha256sum $ART/trainer_config.json $ART/trainer_server_config.json"
srun --overlap -N2 -n2 --label bash -c \
  "test \$(sha256sum $ART/trainer_config.json | cut -d' ' -f1) = $LEADER_SUM || { echo 'MISMATCH — re-stage'; exit 1; }"

banner "5. environment of record"
# Ship NCCL settings (campaign): the ali B300 fabric is 6x400Gb LAG-bonded
# RoCE and hashes flows per queue-pair, so more QPs and channels halve the
# expert all-to-all's per-call latency.
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_SPLIT_DATA_ON_QPS=1
export NCCL_NCHANNELS_PER_NET_PEER=8
# Trace both pipeline-stage leaders (rank 0 = first stage, rank 8 = last).
export BT_PROFILE_RANKS=0,8
# Deliberately NOT set: BT_TF32_LM_HEAD (cancelled, whole ladder is one
# no-TF32 tree), BT_SAVE_STATE_SYNC (absent on this branch; never /save_state),
# BT_SKIP_WARMUP (full recompute sits far from the ceiling; the plateau read
# rule handles the warmup transient).
env | grep -E '^(NCCL_|BT_|NVTE_)' | sort

banner "6. dispatch (2 nodes)"
bash "$PP2/start_trainer_pp2.sh"

cat <<'EOM'

Next, in the background and re-invoked at each 3-minute checkpoint:
    bash /root/.cache/user_artifacts/lps1062_pp2/wait_trainer_health_pp2.sh
Weight load is 13-20 min on a warm cache. Readiness = /status reporting
world_size 16, TP1/PP2/CP8/EP8/DP1, seq 131072.
EOM
