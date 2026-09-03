# Reproducing the captures

## What requires special access

The NVIDIA driver on these nodes has `RmProfilingAdminOnly=1`. The process that starts nsys GPU metrics therefore needs `CAP_SYS_ADMIN`. A normal devbox cannot add that capability after creation. Use a short-lived internal profiler pod on a dedicated/reserved node, as in `profiler-pod.yaml`, and delete it immediately after collecting the reports.

## Capture protocol

1. Create the profiler pod and wait for it to become ready.
2. Install the devbox build dependencies and the full `nsight-systems-2025.3.2` package. The CUDA image's target-side nsys alone can record `.qdstrm` but cannot import `.nsys-rep`.
3. Build `server-megatron-bridge/.venv` with `CUDA_FLAVOR=cu13`.
4. Launch the trainer through `run_trainer_node_nsys.sh`. `nsys launch` instruments all eight torchrun children but does not collect yet.
5. Use `profile_driver.py` for an uncollected warmup and control at the exact 262144-token shape.
6. Start the interactive collection with CUDA/NVTX/cuBLAS/cuDNN/OS-runtime tracing, CPU sampling, context switches, GPU context switches, and 10 kHz GPU metrics on all devices.
7. Run `capture_one.py` once, then immediately call `nsys stop`.
8. Validate the report with `nsys stats`, copy it to the laptop, and delete the profiler pod.

Exact commands and observed outputs are recorded in `WORKLOG.md` after each completed stage.

## Commands used for this run

Set the local constants:

```bash
export KUBECONFIG="$HOME/.kube/ali-apse7-prod-1.yaml"
NS=org-99340d71961343c28c5c567d705ab0c0
POD=jackrao-glm-nsys-b300
LOCAL_RUN="$HOME/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/runs/glm_nsys_gpu_metrics_262k_20260902"
REMOTE_RUN=/root/.cache/user_artifacts/lps1062_bench/glm_nsys_gpu_metrics_262k_20260902
BOX=/root/.cache/user_artifacts/devboxes/3yylmv3
```

The manifest restricts scheduling to B300 nodes carrying the correct customer reservation. Create the pod; it remains pending if no complete eight-GPU node is currently free:

```bash
kubectl apply --dry-run=server -f "$LOCAL_RUN/profiler-pod.yaml"
kubectl apply -f "$LOCAL_RUN/profiler-pod.yaml"
kubectl wait --for=condition=Ready "pod/$POD" -n "$NS" --timeout=600s
```

Install prerequisites and the complete nsys package:

```bash
kubectl exec -n "$NS" "$POD" -- bash -lc \
  'export DEBIAN_FRONTEND=noninteractive; apt-get update -qq; apt-get install -y -qq python3 python3.12 python3.12-dev python3.12-venv ninja-build curl git nsight-systems-2025.3.2'
```

The trainer venv must be node-local. A venv on the mounted CPFS volume makes eight-rank imports unusably slow:

```bash
kubectl exec -n "$NS" "$POD" -- bash -lc \
  'mkdir -p /root/.devbox-venvs; /root/.cache/user_artifacts/bin/uv venv /root/.devbox-venvs/server --python /usr/bin/python3.12; test -L /root/.cache/user_artifacts/devboxes/3yylmv3/trainers/server-megatron-bridge/.venv || ln -s /root/.devbox-venvs/server /root/.cache/user_artifacts/devboxes/3yylmv3/trainers/server-megatron-bridge/.venv'
```

Restore the local-golden submodule mappings described by the devbox-up skill, initialize submodules, then build:

```bash
kubectl exec -n "$NS" "$POD" -- bash -lc \
  'git config --global protocol.file.allow always; git config --global url."/root/.cache/user_artifacts/trainers_main/loops".insteadOf "git@github.com:basetenlabs/baseten-loops.git"; git config --global url."/root/.cache/user_artifacts/trainers_main/server-automodel/vendor/automodel".insteadOf "https://github.com/NVIDIA-NeMo/Automodel.git"; git config --global url."/root/.cache/user_artifacts/trainers_main/server-megatron-bridge/vendor/megatron-bridge".insteadOf "https://github.com/NVIDIA-NeMo/Megatron-Bridge.git"; git config --global url."/root/.cache/user_artifacts/trainers_main/server-megatron-bridge/vendor/megatron-bridge/3rdparty/Megatron-LM".insteadOf "https://github.com/basetenlabs/Megatron-LM.git"; ln -sf /usr/bin/python3.12-config /usr/local/bin/python3-config; source /root/.cache/user_artifacts/devboxes/3yylmv3/env.sh; cd /root/.cache/user_artifacts/devboxes/3yylmv3/trainers; git submodule update --init --recursive; make megatron-bridge-venv CUDA_FLAVOR=cu13; /root/.cache/user_artifacts/bin/uv pip install --python /root/.devbox-venvs/server/bin/python pybind11'
```

Copy this run bundle and the reusable driver into the persistent remote directory:

```bash
kubectl exec -n "$NS" "$POD" -- mkdir -p "$REMOTE_RUN"
kubectl cp "$LOCAL_RUN/." "$NS/$POD:$REMOTE_RUN"
kubectl cp "$HOME/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/tools/profile_driver.py" "$NS/$POD:$REMOTE_RUN/profile_driver.py"
kubectl cp "$HOME/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/tools/mfu.py" "$NS/$POD:$REMOTE_RUN/mfu.py"
```

Temporarily point the generated lifecycle launcher at the nsys-aware node script:

```bash
kubectl exec -n "$NS" "$POD" -- bash -lc \
  "D='$BOX/.devbox_up'; test -e \"\$D/run_trainer_node.stock.sh\" || mv \"\$D/run_trainer_node.sh\" \"\$D/run_trainer_node.stock.sh\"; ln -sfn '$REMOTE_RUN/run_trainer_node_nsys.sh' \"\$D/run_trainer_node.sh\"; chmod +x '$REMOTE_RUN/run_trainer_node_nsys.sh'"
```

Launch nsys without collecting, wait for health, and warm the exact shape. Set `CONFIG` and `SESSION` to either the 1d1m or full-model values:

```bash
CONFIG=trainer-config-full-glm53.json
SESSION=glm53full262metrics

kubectl exec -n "$NS" "$POD" -- bash -lc \
  "source '$BOX/env.sh'; export BT_TRAINER_CONFIG_PATH='$REMOTE_RUN/$CONFIG' BT_TRAINER_SERVER_CONFIG_PATH='$REMOTE_RUN/trainer-server-config.json' NSYS_SESSION_NAME='$SESSION'; bash '$BOX/.devbox_up/start_trainer.sh'"
kubectl exec -n "$NS" "$POD" -- bash "$BOX/.devbox_up/wait_trainer_health.sh"
kubectl exec -n "$NS" "$POD" -- bash -lc \
  "PYTHONPATH='$REMOTE_RUN' /root/.devbox-venvs/server/bin/python '$REMOTE_RUN/profile_driver.py' --label pre-nsys-warm --seq-len 262144 --datums 1 --num-gpus 8 --control-repeats 1"
```

Collect one step. The 10 kHz setting is intentional; lower it to 1000 for a substantially smaller report when sub-millisecond counter resolution is unnecessary:

```bash
REPORT=glm53-full-b300-262k-golden-all-ranks-gpu-metrics

kubectl exec -n "$NS" "$POD" -- bash -lc \
  "nsys start --session='$SESSION' --sample=process-tree --cpuctxsw=process-tree --backtrace=lbr --gpu-metrics-devices=all --gpu-metrics-frequency=10000 --gpuctxsw=true --output='$REMOTE_RUN/$REPORT' --force-overwrite=true --stats=false"
kubectl exec -n "$NS" "$POD" -- bash -lc \
  "PYTHONPATH='$REMOTE_RUN' /root/.devbox-venvs/server/bin/python '$REMOTE_RUN/capture_one.py' --label '$REPORT' --seq-len 262144 --num-gpus 8 --output '$REMOTE_RUN/$REPORT.json'"
kubectl exec -n "$NS" "$POD" -- nsys stop --session="$SESSION"
```

Validate and retrieve the report, then remove the privileged pod:

```bash
kubectl exec -n "$NS" "$POD" -- nsys stats --report cuda_gpu_kern_sum "$REMOTE_RUN/$REPORT.nsys-rep"
kubectl cp --retries=10 "$NS/$POD:$REMOTE_RUN/$REPORT.nsys-rep" "$LOCAL_RUN/$REPORT.nsys-rep"
shasum -a 256 "$LOCAL_RUN/$REPORT.nsys-rep"
kubectl exec -n "$NS" "$POD" -- bash -lc \
  "D='$BOX/.devbox_up'; test ! -e \"\$D/run_trainer_node.stock.sh\" || { rm \"\$D/run_trainer_node.sh\"; mv \"\$D/run_trainer_node.stock.sh\" \"\$D/run_trainer_node.sh\"; }"
kubectl delete -f "$LOCAL_RUN/profiler-pod.yaml" --wait=true
```

Do not leave the profiler pod running. `SYS_ADMIN` is deliberately scoped to this short-lived internal pod.

## Analysis toolchain (added 2026-09-03)

The `.nsys-rep` is only the raw capture. All numbers in `ANALYSIS.md` come from a sqlite export plus the scripts in this folder, run on the pod (the laptop has no `nsys` CLI). Requires the trainer venv python (numpy).

```bash
# one-time per report (~5 min for a 320 MB report; produces a ~3.5 GB sqlite)
nsys export --type sqlite --force-overwrite true --output $REMOTE_RUN/full-glm53-nvtx.sqlite $REMOTE_RUN/<report>.nsys-rep
PY=/root/.devbox-venvs/server/bin/python
cd $REMOTE_RUN
# step-level attribution: categories, phases, per-layer, sync waits, host syncs, GPU idle, GPU metrics per phase
$PY nsys_attrib.py full-glm53-nvtx.sqlite --out analysis --window nvtx:forward_backward --gpu-metrics
# where do blocking host syncs come from (needs --cudabacktrace=sync in the launch flags)
$PY sync_callchains.py full-glm53-nvtx.sqlite --out analysis
# per collective: who arrived last at each HybridEP device_sync, and was its GPU idle (host stall) or busy (more work)
$PY sync_arrivals.py full-glm53-nvtx.sqlite --out analysis
# kernel category x innermost NVTX range pivot (who owns the copies / elementwise / gemm time)
$PY category_by_range.py full-glm53-nvtx.sqlite --out analysis
# per-instance per-rank GPU time of a repeated range (expert load imbalance etc.)
$PY range_imbalance.py full-glm53-nvtx.sqlite --out analysis
```

Notes:
- The NVTX ranges the scripts key on come from trainers PR #1298 (`nvtx_ranges.py`). Without them only `nsys_attrib.py --window auto` works (kernel categories, sync-wait laggards, host sync counts).
- nsys records every CUDA runtime call twice (`cudaStreamSynchronize` and `cudaStreamSynchronize_v3020`); only the unversioned row carries the callchain. Scripts dedupe on that.
- `--pytorch=autograd-nvtx` emits millions of per-op ranges named `aten::x, op_id = N`; the scripts strip the id so they collapse into per-op names (useful: `aten::cat`, `aten::copy_` GPU time per step).
- Python frames in the CUDA callchains are unresolved addresses even with `--python-backtrace=cuda`; the libtorch frames are enough to name the op (`nonzero` from `index_Tensor`, `_local_scalar_dense` from `.item()`), and the enclosing NVTX range names the module.
