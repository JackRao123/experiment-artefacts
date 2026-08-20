#!/usr/bin/env bash
# LPS-1062 PP2/CP8/EP8 @131k bring-up helper — run on the LEADER (tj-<job>).
#
# Mac-side staging first (from experiment_artefacts/glm/lps_1062_perf/):
#   JOB=<jobid>
#   ssh tj-$JOB mkdir -p /root/.cache/user_artifacts/lps1062_pp2
#   scp tools/bringup_pp2cp8ep8.sh tools/profile_driver_new.py tools/mfu.py \
#       tools/dump_parallel_groups.py tj-$JOB:/root/.cache/user_artifacts/lps1062_pp2/
#   scp pp2cp8ep8/configs/trainer_pp2cp8ep8_131k.json pp2cp8ep8/configs/trainer_server.json \
#       tj-$JOB:/root/.cache/user_artifacts/lps1062_pp2/
#
# Then on the leader: bash /root/.cache/user_artifacts/lps1062_pp2/bringup_pp2cp8ep8.sh
# Phases are idempotent; re-run freely. start_trainer is NOT called here —
# do that interactively per the brief's bounded-timeout discipline.
set -euo pipefail

ART=/root/.cache/user_artifacts
PP2=$ART/lps1062_pp2
CLONE=${CLONE:-$ART/trainers_main}   # shared clone provisioned by devbox-up
BRANCH=jackrao/lps-1062-pp2cp8ep8
DEVBOX=$ART/.devbox_up

banner() { printf '\n=== %s ===\n' "$*"; }

banner "0. env"
# shellcheck source=/dev/null
source $ART/env.sh 2>/dev/null || source $DEVBOX/env.sh 2>/dev/null || {
  echo "!! env.sh not found under $ART or $DEVBOX — find it and source first"; exit 1; }
echo "HF_HOME=${HF_HOME:-unset}  MASTER_ADDR=${MASTER_ADDR:-unset}"

banner "1. clone -> $BRANCH"
cd "$CLONE"
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git submodule update --init loops server/vendor/megatron-bridge
git -C server/vendor/megatron-bridge submodule update --init 3rdparty/Megatron-LM
git log --oneline -1

banner "2. GLM-5.2-FP8 weights in HF cache"
HF_CACHE=${HF_HOME:-$ART/team_artifacts/huggingface}
if ls "$HF_CACHE"/hub/models--zai-org--GLM-5.2-FP8/snapshots/*/config.json >/dev/null 2>&1; then
  echo "weights present under $HF_CACHE"
else
  echo "!! GLM-5.2-FP8 NOT in $HF_CACHE (shared cache may have been purged —"
  echo "   ENOSPC post-mortem 2026-08-11). Download is the long pole: starting"
  echo "   it NOW in the background; tell pauli."
  mkdir -p "$PP2/logs"
  if command -v hf >/dev/null 2>&1; then DL=(hf download zai-org/GLM-5.2-FP8);
  elif command -v huggingface-cli >/dev/null 2>&1; then DL=(huggingface-cli download zai-org/GLM-5.2-FP8);
  else DL=(python3 -c 'from huggingface_hub import snapshot_download; snapshot_download("zai-org/GLM-5.2-FP8")'); fi
  nohup "${DL[@]}" > "$PP2/logs/hf_download.log" 2>&1 &
  echo "download pid $! — tail -f $PP2/logs/hf_download.log"
fi

banner "3. trainer configs"
cp "$PP2/trainer_pp2cp8ep8_131k.json" "$ART/trainer_config.json"
cp "$PP2/trainer_server.json" "$ART/trainer_server_config.json"
export BT_TRAINER_CONFIG_PATH="$ART/trainer_config.json"
export BT_TRAINER_SERVER_CONFIG_PATH="$ART/trainer_server_config.json"
# Persist for the shell that launches start_trainer.sh:
{
  echo "export BT_TRAINER_CONFIG_PATH=$ART/trainer_config.json"
  echo "export BT_TRAINER_SERVER_CONFIG_PATH=$ART/trainer_server_config.json"
} > "$PP2/trainer_env.sh"
echo "wrote $PP2/trainer_env.sh (source it in the launch shell)"
python3 - <<'EOF'
import json
cfg = json.load(open("/root/.cache/user_artifacts/trainer_config.json"))
tp, pp, cp, ep = (cfg["tensor_parallel_size"], cfg["pipeline_parallel_size"],
                  cfg["context_parallel_size"], cfg["expert_parallel_size"])
world = 16
dp = world // (tp * pp * cp)
assert dp == 1 and world == tp * pp * cp * dp, (tp, pp, cp, dp)
assert (cp * dp) % ep == 0, (cp, dp, ep)
assert cfg["max_seq_len"] == 131072, "max_seq_len MUST equal packed-buffer size"
print(f"config OK: TP{tp} PP{pp} CP{cp} EP{ep} -> DP{dp}, world 16, seq 131072")
EOF

banner "3b. staged-file integrity (CPFS close-to-open + relay quirk)"
# pauli, prior night: files scp'd onto the shared FS can read back as all-NULs
# from a SIBLING node. md5 each staged file on the leader AND on the worker;
# re-stage from Mac on any mismatch. 30 s of insurance vs a mystery boot fail.
cd "$PP2"
find . -maxdepth 1 -type f -exec sha256sum {} + | sort -k2 > /tmp/pp2_sums.leader
srun --overlap -N2 -n2 --output=/tmp/pp2_sums.%N.txt \
  bash -c "cd '$PP2' && find . -maxdepth 1 -type f -exec sha256sum {} +"
integ_ok=1
for f in /tmp/pp2_sums.*.txt; do
  if diff <(sort -k2 /tmp/pp2_sums.leader) <(sort -k2 "$f") >/dev/null; then
    echo "$f: matches leader"
  else
    echo "!! $f MISMATCH vs leader — re-stage from Mac (scp block in header)"
    diff <(sort -k2 /tmp/pp2_sums.leader) <(sort -k2 "$f") || true
    integ_ok=0
  fi
done
[ "$integ_ok" = 1 ] || exit 1

banner "4. node cleanliness (both nodes)"
# Stale-process sweep (E1 lesson: stop_trainer.sh can leave orphaned workers
# holding GPU memory — an orphan at 273GB OOMs the next boot for a phantom
# reason). Kill trainer/dp_worker strays on BOTH nodes before launch.
srun --overlap -N2 -n2 bash -c \
  'pkill -f "[d]p_worker.main" 2>/dev/null; sleep 2; echo "--- $(hostname)"; pgrep -af "[d]p_worker" || echo "no stray trainer procs"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader' \
  || echo "!! srun cleanliness check failed — inspect before launching"

banner "5. parallel-state group dump (pre-flight rank mapping)"
cat <<EOF
Run this BEFORE first trainer start (verifies PP cross-node, CP/EP intra-node):

  source $PP2/trainer_env.sh
  export MEGATRON_LM_PATH=$CLONE/server/vendor/megatron-bridge/3rdparty/Megatron-LM
  srun --overlap --nodes=2 --ntasks-per-node=8 \\
    torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d \\
      --rdzv_endpoint=\$MASTER_ADDR:\$MASTER_PORT \\
      $PP2/dump_parallel_groups.py --pp 2 --cp 8 --ep 8

Expect: VERDICT: OK — PP cross-node, CP/EP intra-node
EOF

banner "6. next steps (interactive, bounded timeouts)"
cat <<EOF
  source $PP2/trainer_env.sh
  bash $DEVBOX/start_trainer.sh &
  bash $DEVBOX/wait_trainer_health.sh &   # exits ~every 3 min with diagnostics
Watch: config/rank failures in first 1-2 min; weight load 13-20 min (warm cache);
READY banner after warmup. GPU util flatline / hung collective / stalled logs
=> bash $DEVBOX/stop_trainer.sh, read ALL ranks' logs, diagnose, relaunch.
First probe once healthy: /status shows world 16 PP2/CP8/EP8, then
  python3 $PP2/profile_driver_new.py --label pp2-131k-d2 --datums 2
EOF
