#!/bin/sh
# Gap-ledger trace capture for the within-2% mission (2026-08-23).
# Arms at the mission config: ES off both; offload = 4-group + fix knobs.
# Hard deadline on every wait (bounded-waits rule): boot 900s, driver runs
# under its own HTTP timeouts; the whole script is expected < 15 min.
set -eu

ROOT=/root/.cache/user_artifacts
DEVBOX="$ROOT/devboxes/qkpox9w/trainers"
SERVER="$DEVBOX/server-megatron-bridge"
PYTHON="$SERVER/.venv/bin/python"
TORCHRUN="$SERVER/.venv/bin/torchrun"
CONFIG_DIR="$ROOT/lps1062_layer_scaling"
TRACE_ROOT="$ROOT/lps1062_traces"
DRIVER="$TRACE_ROOT/trace_driver_20260822.py"

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
    log="/tmp/${label}.trainer.log"

    stop_trainer
    rm -rf /tmp/checkpoints/profiles/torch_trace
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
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
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
        tail -40 "$log" || true
        exit 1
    fi

    "$PYTHON" "$DRIVER" --label "$label" --seq-len 8192 --pre-windows 3

    mkdir -p "$TRACE_ROOT/$label"
    cp -v /tmp/checkpoints/profiles/torch_trace/*.pt.trace.json "$TRACE_ROOT/$label/" || {
        printf 'TRACE_COPY_FAILED %s\n' "$label"
        exit 1
    }
    cp "$log" "$TRACE_ROOT/$label/trainer.log"
    stop_trainer
}

trap stop_trainer EXIT INT TERM
run_arm 04-none.json ledger-04-baseline-noes-v1 0 0
run_arm 04-four-groups.json ledger-04-offload-noes-v1 6 1
printf 'LEDGER_CAPTURE_COMPLETE\n'
