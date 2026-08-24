#!/bin/sh
# Statistical A/B for the backward-reload knobs + expandable_segments probe.
# 5 arms x 10 untraced windows, fresh process each. Timing arm — telemetry
# and profiler both OFF.
set -eu

ROOT=/root/.cache/user_artifacts
DEVBOX="$ROOT/devboxes/qkpox9w/trainers"
SERVER="$DEVBOX/server-megatron-bridge"
PYTHON="$SERVER/.venv/bin/python"
TORCHRUN="$SERVER/.venv/bin/torchrun"
CONFIG_DIR="$ROOT/lps1062_layer_scaling"
RESULT_DIR="$ROOT/lps1062_bench"
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
    depth=$3
    unchained=$4
    allocconf=$5
    log="/tmp/${label}.trainer.log"

    stop_trainer
    cd "$SERVER"
    nohup env \
        CUDA_VISIBLE_DEVICES=0 \
        PORT=8001 \
        BT_WARMUP_SEQ=8192 \
        BT_TRAINER_CONFIG_PATH="$CONFIG_DIR/$config" \
        BT_TRAINER_SERVER_CONFIG_PATH="$CONFIG_DIR/server-config.json" \
        NUM_NODES=1 \
        NUM_GPUS=1 \
        BT_NODE_RANK=0 \
        PYTORCH_CUDA_ALLOC_CONF="$allocconf" \
        NVTE_CPU_OFFLOAD_V1=1 \
        BT_OFFLOAD_VALVE_TELEMETRY=0 \
        BT_OFFLOAD_PREFETCH_DEPTH="$depth" \
        BT_OFFLOAD_H2D_UNCHAINED="$unchained" \
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
        --repeats 10

    test -s "$RESULT_DIR/$label.json"
    stop_trainer
}

trap stop_trainer EXIT INT TERM
ES=expandable_segments:True
NOES=expandable_segments:False
run_arm 04-none.json        stat-04-base-es-v1      0 0 "$ES"
run_arm 04-four-groups.json stat-04-off-knobsoff-v1 0 0 "$ES"
run_arm 04-four-groups.json stat-04-off-k6u-v1      6 1 "$ES"
run_arm 04-four-groups.json stat-04-off-k6u-noes-v1 6 1 "$NOES"
run_arm 04-none.json        stat-04-base-noes-v1    0 0 "$NOES"
printf 'STAT_AB_COMPLETE\n'
