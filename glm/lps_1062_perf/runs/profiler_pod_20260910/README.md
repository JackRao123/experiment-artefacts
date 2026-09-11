# B300 profiler pod replacement

Provisioned using `tools/profiler_pod/README.md` and its original manifest.
No trainer source changes or benchmark runs are part of this setup.

- Pod: `jackrao-profiler-b300`
- Namespace: `org-99340d71961343c28c5c567d705ab0c0`
- Cluster/kubeconfig: `/Users/jackrao/.kube/ali-apse7-prod-1.yaml`
- Node: `e02-sg-bvv4vhk6101`
- GPUs: 8 B300 (reported as NVIDIA L20D), driver 580.105.08
- Image: `nvidia/cuda:13.0.3-devel-ubuntu24.04`
- Nsight Systems: 2025.3.2.474
- Nsight Compute: 2025.3.1.0
- Capability: SYS_ADMIN, as specified by the profiler-pod manifest

## Access

```bash
KUBECONFIG=/Users/jackrao/.kube/ali-apse7-prod-1.yaml \
  kubectl exec -it -n org-99340d71961343c28c5c567d705ab0c0 \
  jackrao-profiler-b300 -- bash
```

An authenticated SSH endpoint is also available via a localhost-only
`kubectl port-forward` on port 2227. It uses Jack's existing public key;
password authentication is disabled. Restart the forward if needed:

```bash
KUBECONFIG=/Users/jackrao/.kube/ali-apse7-prod-1.yaml \
  kubectl port-forward --address 127.0.0.1 \
  -n org-99340d71961343c28c5c567d705ab0c0 \
  pod/jackrao-profiler-b300 2227:22
```

```bash
ssh -i /Users/jackrao/.ssh/id_ed25519 -o IdentitiesOnly=yes \
  -p 2227 root@127.0.0.1
```

## Files and environment

Inside the pod:

```bash
source /root/.cache/user_artifacts/devboxes/32vj99q/env.sh
cd /root/glm53-pr1355-repro-20260910/trainers
```

The existing project and team PVC mounts have identical paths and subpaths.
The shared cache is readable but still returns ENOSPC on nonempty writes
despite reporting free space. Do not store new run output there until fixed.

`migrate.sh` pulls the old pod's selected node-local experiment directories,
debug checkpoints, staged Qwen weights, and Python environment directly to
the same absolute paths on the replacement pod. It then compares every
selected file by checksum. No files are deleted by migration.

The attempted temporary NetworkPolicy in this directory was rejected by RBAC
and never created. The successful transfer uses existing permitted ingress
to the old pod's SSH service. No cluster network policies were changed.

The original devbox scaffold remains under
`/root/.cache/user_artifacts/devboxes/32vj99q`. Its user-edited worktree is
preserved. The clean PR #1355 checkout is at
`/root/glm53-pr1355-repro-20260910/trainers`; its generated lifecycle copies
are under `/root/glm53-pr1355-repro-20260910/.devbox_up`.
Use generated lifecycle start/wait/stop scripts for future trainer launches.
The old `operate.py` targets `tj-32vj99q`, so its SSH transport must be updated
or its generated lifecycle commands invoked inside this pod before reuse.

## Profiler verification

Installed with Ubuntu/NVIDIA's configured apt repositories:

```bash
apt-get install -y python3 python3.12 python3.12-dev python3.12-venv \
  ninja-build curl git rsync openssh-server \
  nsight-systems-2025.3.2 nsight-compute-2025.3.1
```

The standalone `profiler_probe.cu` is only an installation check, not a model
benchmark or a change to repository tests.

- Nsight Compute collected a kernel report with SM throughput, occupancy,
  memory throughput, launch statistics, and waves per SM.
- Nsight Systems captured 100 CUDA kernel launches and 1,934,268 GPU metric
  rows, including SMs Active and SM Issue.
- Both binary reports are saved here and on the pod under
  `/root/profiler-setup-20260910`.

## Completed checks

- Migration transferred 156,714,517,432 bytes, preserving 152,654 file/directory/
  symlink entries. The second checksum-based pass reported no differences.
  See `migration-pull.log`.
- PR source pins and imports passed: trainers `134851f87`, Bridge
  `60b1570fb`, Core `cf81782b2`, with no tracked source changes.
- PyTorch 2.11.0+cu130, Transformer Engine 2.16.0, and Triton 3.6.0 imported.
  A CUDA operation completed successfully on every one of the eight GPUs.
  See `environment-verification.log`.
- `truss train stop --remote baseten --project-id dq47r1q --job-id 32vj99q
  --non-interactive` returned "Training job stopped successfully."
  See `old-job-stop.log`. The shared user-edited worktree was not removed.

The user subsequently said Qwen is no longer needed. Its already-completed
copy is left untouched; no further Qwen work is being done.

## Requested team-cache staging: blocked

The intended shared destination is
`/root/.cache/team_artifacts/debug-checkpoints/glm53`.
On the new pod, a real nonempty write of the debug model's `config.json`
failed at file close with `No space left on device`. No debug checkpoint was
published into that directory. This is not merely a metadata-preservation
error from `cp -a`: ordinary `cp` fails too.

Verified node-local copies remain at:

- `/root/glm53-1d2m-262k-20260909/model` (23 GiB)
- `/root/glm53-1d3m-262k-20260909/model` (32 GiB)
- `/root/glm53-2d2m-262k-20260909/model` (23 GiB)

Do not delete the profiler pod before these are successfully staged on a
writable persistent filesystem. The underlying team-cache write failure
must be resolved before shared checkpoint staging can finish.
