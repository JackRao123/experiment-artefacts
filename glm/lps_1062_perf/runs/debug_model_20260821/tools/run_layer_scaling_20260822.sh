#!/bin/sh
set -eu

ROOT=/root/.cache/user_artifacts
DEVBOX="$ROOT/devboxes/qkpox9w/trainers"
SERVER="$DEVBOX/server-megatron-bridge"
PYTHON="$SERVER/.venv/bin/python"
TORCHRUN="$SERVER/.venv/bin/torchrun"
CONFIG_DIR="$ROOT/lps1062_layer_scaling"
RESULT_DIR="$ROOT/lps1062_bench"
SERVER_CONFIG="$CONFIG_DIR/server-config.json"
DRIVER="$CONFIG_DIR/bench_driver2c.py"

stop_trainer() {
    pids=$(ps -ef | grep trainers_server_main.main | grep -v grep | tr -s ' ' | cut -d ' ' -f 2 || true)
    if [ -n "$pids" ]; then
        kill -TERM $pids 2>/dev/null || true
        sleep 5
    fi
}

run_arm() {
    config=$1
    label=$2
    log="/tmp/${label}.trainer.log"

    stop_trainer
    rm -rf /tmp/checkpoints/profiles/torch_trace
    cd "$SERVER"
    nohup env \
        CUDA_VISIBLE_DEVICES=0 \
        PORT=8001 \
        BT_WARMUP_SEQ=8192 \
        BT_TRAINER_CONFIG_PATH="$CONFIG_DIR/$config" \
        BT_TRAINER_SERVER_CONFIG_PATH="$SERVER_CONFIG" \
        NUM_NODES=1 \
        NUM_GPUS=1 \
        BT_NODE_RANK=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        NVTE_CPU_OFFLOAD_V1=1 \
        BT_OFFLOAD_VALVE_TELEMETRY=0 \
        "$TORCHRUN" --standalone --nproc_per_node=1 \
        -m trainers_server_main.main --backend megatron_bridge >"$log" 2>&1 &
    trainer_pid=$!

    ready=0
    for _ in $(seq 1 900); do
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
        printf 'TRAINER_BOOT_FAILED %s %s\n' "$label" "$log"
        exit 1
    fi

    cd "$CONFIG_DIR"
    "$PYTHON" "$DRIVER" \
        --label "$label" \
        --seq-len 8192 \
        --num-gpus 1 \
        --datums 1 \
        --warmup-datums 1 \
        --repeats 3

    test -s "$RESULT_DIR/$label.json"
    stop_trainer
}

trap stop_trainer EXIT INT TERM
run_arm 02-none.json lps1062-depth-02-none-v1
run_arm 02-four-groups.json lps1062-depth-02-four-groups-v1
run_arm 04-none.json lps1062-depth-04-none-v1
run_arm 04-four-groups.json lps1062-depth-04-four-groups-v1
run_arm 06-none.json lps1062-depth-06-none-v1
run_arm 06-four-groups.json lps1062-depth-06-four-groups-v1
printf 'LAYER_SCALING_COMPLETE\n'
