# Reproduce the GLM-5.3 262K all-rank Nsight trace

This reproduces the configuration and capture protocol used for:

`glm53-full-b300-262k-nvtx-all-ranks-gpu-metrics.nsys-rep`

The new report will not be byte-identical because scheduling, routing, and
profiling timestamps vary between runs.

Start with a ready B300 profiler pod from `README.md`.

## Captured configuration

- trainers base: `27a688197879d2d45f2d46971c639eecf93cacfb`
- instrumentation: `nvtx_ranges_pr1298.patch`
- model: `zai-org/GLM-5.3` snapshot
  `187fb9fff6319062325ff825627ef6db084d9bc6`
- one 262,144-token synthetic datum, RNG seed `0xB300`
- 8×B300, TP1/PP1/CP8/EP8/ETP1/DP1
- HybridEP, native-FP8 experts, FlashAttention, LoRA rank 32
- full uniform one-layer recompute
- Nsight Systems 2025.3.2

## 1. Set local and remote variables

Use the job ID whose shared worktree was prepared when the profiler pod was
created.

```bash
export KUBECONFIG="$HOME/.kube/ali-apse7-prod-1.yaml"
NS=org-99340d71961343c28c5c567d705ab0c0
POD=jackrao-profiler-b300
JOB=<job-id>

ROOT="$HOME/Documents/trainers/experiment_artefacts/glm/lps_1062_perf"
SOURCE="$ROOT/runs/glm53_nsys_gpu_metrics_262k_20260902"
POD_TOOLS="$ROOT/tools/profiler_pod"
REMOTE_RUN=/root/.cache/user_artifacts/lps1062_bench/glm53_262k_nsys_repro
BOX=/root/.cache/user_artifacts/devboxes/$JOB
```

If using the currently running historical pod, set:

```bash
POD=jackrao-glm-nsys-b300
JOB=w7xrmo3
```

## 2. Stage the capture files

```bash
kubectl exec -n "$NS" "$POD" -- mkdir -p "$REMOTE_RUN"

kubectl cp "$SOURCE/capture_one.py" \
  "$NS/$POD:$REMOTE_RUN/capture_one.py"
kubectl cp "$SOURCE/trainer-server-config.json" \
  "$NS/$POD:$REMOTE_RUN/trainer-server-config.json"
kubectl cp "$SOURCE/nvtx_ranges_pr1298.patch" \
  "$NS/$POD:$REMOTE_RUN/nvtx_ranges_pr1298.patch"
kubectl cp "$ROOT/tools/profile_driver.py" \
  "$NS/$POD:$REMOTE_RUN/profile_driver.py"
kubectl cp "$ROOT/tools/mfu.py" \
  "$NS/$POD:$REMOTE_RUN/mfu.py"
kubectl cp "$POD_TOOLS/run_trainer_node_nsys.sh" \
  "$NS/$POD:$REMOTE_RUN/run_trainer_node_nsys.sh"
```

Create a fresh trainer config:

```bash
kubectl exec -n "$NS" "$POD" -- bash -lc \
  "cat >'$REMOTE_RUN/trainer-config.json' <<'EOF'
{
  \"base_model\": \"/root/.cache/team_artifacts/huggingface/hub/models--zai-org--GLM-5.3/snapshots/187fb9fff6319062325ff825627ef6db084d9bc6\",
  \"checkpoint_dir\": \"/root/.cache/user_artifacts/lps1062_bench/glm53_262k_nsys_repro/checkpoints\",
  \"max_seq_len\": 262144,
  \"tensor_parallel_size\": 1,
  \"pipeline_parallel_size\": 1,
  \"expert_parallel_size\": 8,
  \"context_parallel_size\": 8,
  \"expert_tensor_parallel_size\": 1,
  \"trust_remote_code\": true,
  \"attention_backend\": \"flash\",
  \"lora_rank\": 32,
  \"lora_alpha\": 32,
  \"recompute\": {
    \"granularity\": \"full\",
    \"method\": \"uniform\",
    \"num_layers\": 1
  },
  \"moe_flex_dispatcher_backend\": \"hybridep\",
  \"expert_weight_storage\": \"native_fp8\",
  \"weight_sync\": {\"type\": \"disabled\"}
}
EOF"
```

## 3. Restore the exact profiled code

The worktree must be clean. This intentionally leaves it detached and dirty
with the preserved NVTX patch.

```bash
kubectl exec -n "$NS" "$POD" -- env JOB="$JOB" REMOTE_RUN="$REMOTE_RUN" bash -lc '
  set -euo pipefail
  BOX=/root/.cache/user_artifacts/devboxes/$JOB
  source "$BOX/env.sh"
  cd "$BOX/trainers"

  test -z "$(git status --porcelain)"
  git cat-file -e 27a688197879d2d45f2d46971c639eecf93cacfb^{commit} ||
    git fetch origin 27a688197879d2d45f2d46971c639eecf93cacfb
  git checkout --detach 27a688197879d2d45f2d46971c639eecf93cacfb
  git submodule update --init --recursive
  git apply "$REMOTE_RUN/nvtx_ranges_pr1298.patch"
  git diff --check

  make megatron-bridge-venv CUDA_FLAVOR=cu13
  /root/.cache/user_artifacts/bin/uv pip install \
    --python /root/.devbox-venvs/server/bin/python pybind11
'
```

## 4. Install the Nsight-aware lifecycle launcher

```bash
kubectl exec -n "$NS" "$POD" -- \
  env JOB="$JOB" REMOTE_RUN="$REMOTE_RUN" bash -lc '
    set -euo pipefail
    BOX=/root/.cache/user_artifacts/devboxes/$JOB
    D="$BOX/.devbox_up"
    chmod +x "$REMOTE_RUN/run_trainer_node_nsys.sh"
    test -e "$D/run_trainer_node.stock.sh" ||
      mv "$D/run_trainer_node.sh" "$D/run_trainer_node.stock.sh"
    ln -sfn "$REMOTE_RUN/run_trainer_node_nsys.sh" "$D/run_trainer_node.sh"
  '
```

## 5. Start the trainer under an idle Nsight session

`nsys launch` instruments the process tree. Collection begins later, after
model startup and warmup.

```bash
SESSION=glm53full262nvtx

kubectl exec -n "$NS" "$POD" -- \
  env JOB="$JOB" REMOTE_RUN="$REMOTE_RUN" SESSION="$SESSION" bash -lc '
    set -euo pipefail
    BOX=/root/.cache/user_artifacts/devboxes/$JOB
    source "$BOX/env.sh"
    export BOX
    export RUN="$REMOTE_RUN"
    export NSYS_SESSION_NAME="$SESSION"
    export BT_TRAINER_CONFIG_PATH="$REMOTE_RUN/trainer-config.json"
    export BT_TRAINER_SERVER_CONFIG_PATH="$REMOTE_RUN/trainer-server-config.json"
    bash "$BOX/.devbox_up/start_trainer.sh"
  '
```

Run the generated waiter. If it exits at an intermediate checkpoint, inspect
its output and run it again until the service is healthy.

```bash
kubectl exec -n "$NS" "$POD" -- \
  env JOB="$JOB" bash -lc \
  'bash "/root/.cache/user_artifacts/devboxes/$JOB/.devbox_up/wait_trainer_health.sh"'
```

## 6. Warm up and record the unprofiled control

Do not capture the first step. It includes compilation, autotuning, allocator,
and collective setup.

```bash
kubectl exec -n "$NS" "$POD" -- env REMOTE_RUN="$REMOTE_RUN" bash -lc '
  PYTHONPATH="$REMOTE_RUN" /root/.devbox-venvs/server/bin/python \
    "$REMOTE_RUN/profile_driver.py" \
    --label glm53-full-pre-nsys-262k \
    --seq-len 262144 \
    --datums 1 \
    --num-gpus 8 \
    --control-repeats 1
'
```

The original clean control forward/backward was approximately 26.3–26.7
seconds. Large differences should be investigated before collecting.

## 7. Capture exactly one step

```bash
REPORT=glm53-full-b300-262k-nvtx-all-ranks-gpu-metrics-repro

kubectl exec -n "$NS" "$POD" -- \
  env SESSION="$SESSION" REMOTE_RUN="$REMOTE_RUN" REPORT="$REPORT" bash -lc '
    nsys start \
      --session="$SESSION" \
      --sample=process-tree \
      --cpuctxsw=process-tree \
      --backtrace=lbr \
      --gpu-metrics-devices=all \
      --gpu-metrics-frequency=10000 \
      --gpuctxsw=true \
      --output="$REMOTE_RUN/$REPORT" \
      --force-overwrite=true \
      --stats=false
  '

kubectl exec -n "$NS" "$POD" -- \
  env REMOTE_RUN="$REMOTE_RUN" REPORT="$REPORT" bash -lc '
    PYTHONPATH="$REMOTE_RUN" /root/.devbox-venvs/server/bin/python \
      "$REMOTE_RUN/capture_one.py" \
      --label "$REPORT" \
      --seq-len 262144 \
      --num-gpus 8 \
      --output "$REMOTE_RUN/$REPORT.json"
  '

kubectl exec -n "$NS" "$POD" -- \
  env SESSION="$SESSION" bash -lc 'nsys stop --session="$SESSION"'
```

The original profiled forward/backward was about 27.7 seconds with loss
approximately 12.327. The report was roughly 300 MB.

## 8. Validate and copy the report

```bash
kubectl exec -n "$NS" "$POD" -- \
  env REMOTE_RUN="$REMOTE_RUN" REPORT="$REPORT" bash -lc '
    nsys stats --report cuda_gpu_kern_sum "$REMOTE_RUN/$REPORT.nsys-rep"
  '

kubectl cp --retries=10 \
  "$NS/$POD:$REMOTE_RUN/$REPORT.nsys-rep" \
  "$ROOT/runs/glm53_nsys_gpu_metrics_262k_20260902/$REPORT.nsys-rep"

kubectl cp \
  "$NS/$POD:$REMOTE_RUN/$REPORT.json" \
  "$ROOT/runs/glm53_nsys_gpu_metrics_262k_20260902/$REPORT.json"
```

## 9. Stop the trainer and restore the lifecycle launcher

```bash
kubectl exec -n "$NS" "$POD" -- env JOB="$JOB" bash -lc '
  set -euo pipefail
  BOX=/root/.cache/user_artifacts/devboxes/$JOB
  bash "$BOX/.devbox_up/stop_trainer.sh"

  D="$BOX/.devbox_up"
  if test -e "$D/run_trainer_node.stock.sh"; then
    rm "$D/run_trainer_node.sh"
    mv "$D/run_trainer_node.stock.sh" "$D/run_trainer_node.sh"
  fi
'
```

Delete the profiler pod using the provisioning guide when no further captures
are needed.
