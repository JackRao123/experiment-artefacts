#!/bin/sh
# Telemetry run: 4-layer 4-group offload arm with valve/reload telemetry ON.
# Timing from this run is NOT usable (telemetry perturbs the tiny proxy);
# only the cumulative prefetch_hits / demand_misses / commit counters are.
set -eu

ROOT=/root/.cache/user_artifacts
DEVBOX="$ROOT/devboxes/qkpox9w/trainers"
SERVER="$DEVBOX/server-megatron-bridge"
PYTHON="$SERVER/.venv/bin/python"
TORCHRUN="$SERVER/.venv/bin/torchrun"
CONFIG_DIR="$ROOT/lps1062_layer_scaling"
TRACE_ROOT="$ROOT/lps1062_traces"

stop_trainer() {
    pids=$(ps -ef | grep trainers_server_main.main | grep -v grep | tr -s ' ' | cut -d ' ' -f 2 || true)
    if [ -n "$pids" ]; then
        kill -TERM $pids 2>/dev/null || true
        sleep 5
    fi
}

trap stop_trainer EXIT INT TERM
stop_trainer
log=/tmp/telemetry-04-four-groups.trainer.log
cd "$SERVER"
nohup env \
    CUDA_VISIBLE_DEVICES=0 \
    PORT=8001 \
    BT_WARMUP_SEQ=8192 \
    BT_TRAINER_CONFIG_PATH="$CONFIG_DIR/04-four-groups.json" \
    BT_TRAINER_SERVER_CONFIG_PATH="$CONFIG_DIR/server-config.json" \
    NUM_NODES=1 \
    NUM_GPUS=1 \
    BT_NODE_RANK=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    NVTE_CPU_OFFLOAD_V1=1 \
    BT_OFFLOAD_VALVE_TELEMETRY=1 \
    BT_OFFLOAD_VALVE_TELEMETRY_EVERY=1 \
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
    printf 'TRAINER_BOOT_FAILED %s\n' "$log"
    exit 1
fi

"$PYTHON" "$TRACE_ROOT/trace_driver_20260822.py" \
    --label telemetry-04-four-groups-v1 --seq-len 8192 --pre-windows 3 || true

grep "BT_OFFLOAD_VALVE_TELEMETRY" "$log" \
    > "$TRACE_ROOT/telemetry-04-four-groups-v1.telemetry.txt" || true
cp "$log" "$TRACE_ROOT/telemetry-04-four-groups-v1.trainer.log"
stop_trainer
printf 'TELEMETRY_RUN_COMPLETE\n'
