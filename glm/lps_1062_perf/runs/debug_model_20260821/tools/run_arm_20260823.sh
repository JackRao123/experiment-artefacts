#!/bin/sh
# Parameterized single-arm runner for the within-2% mission.
# Usage: run_arm_20260823.sh <config.json> <label> <depth> <unchained> \
#            <chunk_bytes> <alloc_conf> <repeats> <trace:0|1> [paced:0|1] [seqlen]
# Boots fresh, runs bench_driver2c for <repeats> windows; if trace=1, runs
# trace_driver (3 pre-windows + 1 traced) instead. Copies trainer log and
# any trace out. Bounded waits throughout.
set -eu

ROOT=/root/.cache/user_artifacts
DEVBOX="$ROOT/devboxes/qkpox9w/trainers"
SERVER="$DEVBOX/server-megatron-bridge"
PYTHON="$SERVER/.venv/bin/python"
TORCHRUN="$SERVER/.venv/bin/torchrun"
CONFIG_DIR="$ROOT/lps1062_layer_scaling"
RESULT_DIR="$ROOT/lps1062_bench"
TRACE_ROOT="$ROOT/lps1062_traces"

config=$1
label=$2
depth=$3
unchained=$4
chunk=$5
allocconf=$6
repeats=$7
trace=$8
paced=${9:-0}
seqlen=${10:-8192}
log="/tmp/${label}.trainer.log"

stop_trainer() {
    pids=$(ps -ef | grep trainers_server_main.main | grep -v grep | tr -s ' ' | cut -d ' ' -f 2 || true)
    if [ -n "$pids" ]; then
        kill -TERM $pids 2>/dev/null || true
        sleep 5
    fi
}

trap stop_trainer EXIT INT TERM
stop_trainer
rm -rf /tmp/checkpoints/profiles/torch_trace
cd "$SERVER"
nohup env \
    CUDA_VISIBLE_DEVICES=0 \
    PORT=8001 \
    BT_WARMUP_SEQ="$seqlen" \
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
    BT_OFFLOAD_COPY_CHUNK_BYTES="$chunk" \
    BT_OFFLOAD_D2H_PACED="$paced" \
    BT_MOE_DTOH_HIGH_PRIORITY="${BT_MOE_DTOH_HIGH_PRIORITY:-0}" \
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

if [ "$trace" = "1" ]; then
    "$PYTHON" "$TRACE_ROOT/trace_driver_20260822.py" \
        --label "$label" --seq-len "$seqlen" --pre-windows 3
    mkdir -p "$TRACE_ROOT/$label"
    cp -v /tmp/checkpoints/profiles/torch_trace/*.pt.trace.json "$TRACE_ROOT/$label/"
    cp "$log" "$TRACE_ROOT/$label/trainer.log"
else
    cd "$CONFIG_DIR"
    "$PYTHON" "$CONFIG_DIR/bench_driver2c.py" \
        --label "$label" \
        --seq-len "$seqlen" \
        --num-gpus 1 \
        --datums 1 \
        --warmup-datums 1 \
        --repeats "$repeats"
    test -s "$RESULT_DIR/$label.json"
fi
stop_trainer
printf 'ARM_COMPLETE %s\n' "$label"
