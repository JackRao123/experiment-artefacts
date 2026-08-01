# Devbox ↔ prod trainer environment parity

**Why this doc exists**: LPS-1003 Issue 2 stayed unreproducible on the devbox
for a week because `.devbox_up/run_trainer_node.sh` hand-exports env vars the
prod entrypoint (`server/scripts/launch.sh`) never sets. One of them —
`CUDA_DEVICE_MAX_CONNECTIONS=1` — changes CUDA launch-queue semantics and
masked the production race entirely (devbox 0/10 windows vs prod 6/6).
Environment is numerics-relevant. Treat any devbox-only export as a
potential behavioral fork.

## The divergence found 2026-07-31 (fingerprints: runs/prod_5wolkzw/fingerprint.json vs runs/devbox_arm/fingerprint_rank0.json)

Devbox-only exports (NOT set in prod trainer environ):
- `CUDA_DEVICE_MAX_CONNECTIONS=1`  ← THE one that mattered (see NOTEBOOK.md)
- `MEGATRON_SKIP_GLOO_GROUPS=1`
- `NVTE_FRAMEWORK=pytorch`
- `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600`
- `NCCL_DEBUG=WARN`
- `OMP_NUM_THREADS=1` (equivalent anyway: torchrun defaults workers to 1)

Prod-only:
- `NVTE_CUDA_ARCHS=103a`
- `LD_LIBRARY_PATH=/usr/local/nvidia/lib:...:/usr/local/cuda/lib64`
  (inert in practice: mapped .so sets are 60/73 sha-identical; all CUDA math
  libs resolve to the same pip wheels on both sides)

Identical and verified (so NOT sources of divergence): driver 580.105.08,
kernel, node pool (b300-1-*), python 3.12.3, torch/cudnn/cublas/nccl/TE/
triton/cudnn-frontend wheels (sha256), allocator conf, torchrun topology.

## How to check parity (one command per side)

1. Prod side (any live trainer pod):
   `kubectl -n <ns> cp parity/fingerprint.py <pod>:/tmp/ -c trainer-container`
   `kubectl -n <ns> exec <pod> -c trainer-container -- python3 /tmp/fingerprint.py --python /app/.venv/bin/python --out /tmp/fp_prod.json`
2. Devbox side (while a trainer runs), on the rank-0 node:
   `python3 parity/fingerprint.py --python <venv>/bin/python --out /tmp/fp_devbox.json`
3. Diff: `python3 parity/fingerprint_diff.py fp_prod.json fp_devbox.json --labels PROD DEVBOX`
   — the "numerics-relevant env" section at the bottom is the go/no-go list.

A canonical prod reference (trainer 5wolkzw, image trainer-cuda13-sm103-0e0b65a,
2026-07-31) is checked in at `runs/prod_5wolkzw/fingerprint.json`. The devbox
helper `check_env_parity.sh` (on CPFS `lps1003/parity/`) diffs a live devbox
trainer against it.

## Going-forward rules

1. **Prod-faithful boots**: use `run_trainer_node_prodenv.sh` (this dir) —
   byte-equal to the stock script minus the 6 devbox-only exports, plus
   NVTE_CUDA_ARCHS=103a. Use for any investigation where prod behavior is the
   subject. The stock `run_trainer_node.sh` remains fine for stack bring-up.
2. **Never add an env export to devbox-up without checking prod**: grep the
   live pod environ, not launch.sh — the image ENV + operator injections are
   part of the picture.
3. **After the mitigation PR** (CUDA_DEVICE_MAX_CONNECTIONS=1 in launch.sh),
   regenerate the prod reference fingerprint — the stock devbox env and prod
   will then agree on conn, and prodenv boots must ALSO set it to stay
   faithful (i.e. the repro will then require explicitly unsetting it).
4. Long-term fix (ticketed): derive devbox trainer env from the image's own
   launch path instead of hand-copied exports, so parity holds by
   construction.
