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
        printf 'TRAINER_BOOT_FAILED %s %s\n' "$label" "$log"
        exit 1
    fi

    cd "$ROOT/lps1062"
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

run_arm 01-core-attn.json lps1062-ladder-01-core-attn
run_arm 02-core-attn-moe-act.json lps1062-ladder-02-core-attn-moe-act
run_arm 03-core-attn-moe-act-qkv-linear.json lps1062-ladder-03-core-attn-moe-act-qkv-linear
run_arm 04-core-attn-moe-act-qkv-linear-attn-proj.json lps1062-ladder-04-core-attn-moe-act-qkv-linear-attn-proj
run_arm 05-core-attn-moe-act-qkv-linear-attn-proj-moe-router.json lps1062-ladder-05-core-attn-moe-act-qkv-linear-attn-proj-moe-router
run_arm 06-all.json lps1062-ladder-06-all

printf 'OFFLOAD_LADDER_COMPLETE\n'
