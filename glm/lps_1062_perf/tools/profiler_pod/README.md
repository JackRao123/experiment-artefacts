# Get a new B300 profiler pod

This guide stops when a fresh 8×B300 pod has:

- `CAP_SYS_ADMIN` for Nsight GPU metrics
- the shared project and team caches
- Nsight Systems 2025.3.2
- a trainer worktree and node-local Megatron Bridge venv

It does not start a trainer or collect a profile.

## 1. Create the shared devbox scaffold

`devbox-up` creates the worktree, environment file, and trainer lifecycle
scripts. Do not build its venv because the venv is node-local and would need to
be rebuilt in the profiler pod.

```bash
devbox-up 8 b300
```

Record the job ID printed at the end:

```bash
JOB=<job-id>
BOX=/root/.cache/user_artifacts/devboxes/$JOB
```

Initialize the shared submodules before releasing the devbox:

```bash
ssh "tj-$JOB" \
  "source '$BOX/env.sh' && cd '$BOX/trainers' && git submodule update --init --recursive"
```

## 2. Queue the profiler pod and release the devbox

The manifest is specific to Jack's B300 project on `ali-apse7-prod-1`. Before
using it, verify that the `charles--parsed--com` reservation, namespace, PVC,
and `dq47r1q/user_artifacts` project subpath are still correct.

```bash
export KUBECONFIG="$HOME/.kube/ali-apse7-prod-1.yaml"
NS=org-99340d71961343c28c5c567d705ab0c0
POD=jackrao-profiler-b300
HERE="$HOME/Documents/trainers/experiment_artefacts/glm/lps_1062_perf/tools/profiler_pod"

kubectl apply --dry-run=server -f "$HERE/profiler-pod.yaml"
kubectl apply -f "$HERE/profiler-pod.yaml"
truss train stop --remote baseten --job-id "$JOB"
kubectl wait -n "$NS" --for=condition=Ready "pod/$POD" --timeout=3600s
```

The pod may run on a different physical node. That does not matter: the
worktree and caches are on the mounted persistent filesystem.

## 3. Install Nsight and build dependencies

```bash
kubectl exec -n "$NS" "$POD" -- bash -lc \
  'export DEBIAN_FRONTEND=noninteractive
   apt-get update -qq
   apt-get install -y -qq \
     python3 python3.12 python3.12-dev python3.12-venv \
     ninja-build curl git nsight-systems-2025.3.2'
```

## 4. Build the pod-local trainer venv

```bash
kubectl exec -n "$NS" "$POD" -- env JOB="$JOB" bash -lc '
  set -euo pipefail
  BOX=/root/.cache/user_artifacts/devboxes/$JOB
  test -f "$BOX/env.sh"
  test -d "$BOX/trainers"
  test -x /root/.cache/user_artifacts/bin/uv

  mkdir -p /root/.devbox-venvs
  /root/.cache/user_artifacts/bin/uv venv \
    /root/.devbox-venvs/server --python /usr/bin/python3.12
  test -L "$BOX/trainers/server-megatron-bridge/.venv" ||
    ln -s /root/.devbox-venvs/server \
      "$BOX/trainers/server-megatron-bridge/.venv"
  ln -sf /usr/bin/python3.12-config /usr/local/bin/python3-config

  source "$BOX/env.sh"
  cd "$BOX/trainers"
  git submodule update --init --recursive
  make megatron-bridge-venv CUDA_FLAVOR=cu13
  /root/.cache/user_artifacts/bin/uv pip install \
    --python /root/.devbox-venvs/server/bin/python pybind11
'
```

## 5. Verify and enter the pod

```bash
kubectl exec -n "$NS" "$POD" -- bash -lc '
  nsys --version
  nsys profile --gpu-metrics-devices=help
  /root/.devbox-venvs/server/bin/python -c \
    "import torch; print(torch.__version__, torch.cuda.device_count())"
'

kubectl exec -it -n "$NS" "$POD" -- bash
```

Inside the pod:

```bash
JOB=<job-id>
BOX=/root/.cache/user_artifacts/devboxes/$JOB
source "$BOX/env.sh"
cd "$BOX/trainers"
```

Use the generated scripts in `$BOX/.devbox_up/` to start, wait for, and stop
the trainer. Do not manually launch trainer processes.

## Delete the profiler pod

Copy any node-local output you need into `/root/.cache/user_artifacts` first.

```bash
kubectl delete -n "$NS" -f "$HERE/profiler-pod.yaml" --wait=true
```

Deleting the pod removes its apt packages, processes, and
`/root/.devbox-venvs`. The shared worktree, model cache, and artifacts remain.
