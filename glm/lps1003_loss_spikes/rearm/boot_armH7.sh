#!/usr/bin/env bash
# LPS-1003 arm H7: golden 262k cp16 config, prodenv (conn UNSET) +
# BT_SKIP_FULL_WARMUP=1, harness7 lever hook chaining harness6 double-exec
# (BT_DSA_DOUBLE=indexer,flashmla). Dispatch only; probe separately.
set -uo pipefail
LPS=/root/.cache/user_artifacts/lps1003
PAR=$LPS/parity
DU=/root/.cache/user_artifacts/.devbox_up
OUT=$LPS/rearm/armH7_$(date +%m%d_%H%M%S)
mkdir -p "$OUT/dsa_double" "$OUT/levers"
echo "$OUT" > $LPS/rearm/current_out
exec >>"$OUT/driver.log" 2>&1
echo "=== armH7 start $(date -u +%FT%TZ) out=$OUT"
source /root/.cache/user_artifacts/env.sh
export BT_SKIP_FULL_WARMUP=1
export BT_TRAINER_CONFIG_PATH=$LPS/trainer-config.flash.json
export BT_TRAINER_SERVER_CONFIG_PATH=$LPS/trainer-server-config.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$LPS/rearm/harness7${PYTHONPATH:+:$PYTHONPATH}"
export BT_H6_SITE=$PAR/harness6/sitecustomize.py
export BT_DSA_DOUBLE=indexer,flashmla
export BT_DSA_DOUBLE_DIR=$OUT/dsa_double
export BT_LEVER_FILE=$LPS/rearm/levers.json
export BT_LEVER_DIR=$OUT/levers
export NUM_NODES=2
export MASTER_PORT=$((29500 + RANDOM % 1000))
printf '2\n' > "$DU/trainer_num_nodes"
nohup srun --job-name=devbox_trainer --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  --gres=gpu:8 --cpus-per-task=248 --export=ALL \
  bash "$PAR/run_trainer_node_prodenv.sh" > "$DU/trainer_srun.log" 2>&1 < /dev/null &
echo "dispatched srun pid $! master_port $MASTER_PORT out=$OUT"
