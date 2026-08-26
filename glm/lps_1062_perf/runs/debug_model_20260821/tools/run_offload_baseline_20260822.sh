#!/bin/sh
set -eu

ROOT=/root/.cache/user_artifacts
SERVER="$ROOT/trainers_main/server-megatron-bridge"
PYTHON="$SERVER/.venv/bin/python"
TORCHRUN="$SERVER/.venv/bin/torchrun"
CONFIG_DIR="$ROOT/lps1062_offload_ladder"
RESULT_DIR="$ROOT/lps1062_bench"
SERVER_CONFIG="$ROOT/lps1062/server-config.json"
DRIVER="$ROOT/lps1062/bench_driver2c.py"
LABEL=lps1062-ladder-00-no-recompute-no-offload
LOG="/tmp/${LABEL}.trainer.log"

stop_trainer() {
    pids=$(ps -ef | grep trainers_server_main.main | grep -v grep | tr -s ' ' | cut -d ' ' -f 2 || true)
    if [ -n "$pids" ]; then
        kill -TERM $pids 2>/dev/null || true
        sleep 5
    fi
}

stop_trainer
rm -rf /tmp/checkpoints/profiles/torch_trace
cd "$SERVER"
nohup env \
    CUDA_VISIBLE_DEVICES=0 \
    PORT=8001 \
    BT_WARMUP_SEQ=8192 \
    BT_TRAINER_CONFIG_PATH="$CONFIG_DIR/00-no-recompute-no-offload.json" \
    BT_TRAINER_SERVER_CONFIG_PATH="$SERVER_CONFIG" \
    NUM_NODES=1 \
    NUM_GPUS=1 \
    BT_NODE_RANK=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    BT_OFFLOAD_VALVE_TELEMETRY=0 \
    "$TORCHRUN" --standalone --nproc_per_node=1 \
    -m trainers_server_main.main --backend megatron_bridge >"$LOG" 2>&1 &
trainer_pid=$!

ready=0
for _ in $(seq 1 600); do
    if curl -fsS http://127.0.0.1:8001/health >/dev/null 2>&1; then
        ready=1
        break
    fi
    if ! kill -0 "$trainer_pid" 2>/dev/null; then
        break
    fi
    sleep 1
done
if [ "$ready" -ne 1 ]; then
    printf 'TRAINER_BOOT_FAILED %s %s\n' "$LABEL" "$LOG"
    exit 1
fi

cd "$ROOT/lps1062"
"$PYTHON" "$DRIVER" \
    --label "$LABEL" \
    --seq-len 8192 \
    --num-gpus 1 \
    --datums 1 \
    --warmup-datums 1 \
    --repeats 3

test -s "$RESULT_DIR/$LABEL.json"
stop_trainer
printf 'BASELINE_COMPLETE\n'
