#!/usr/bin/env bash
# Run on the replacement pod with SSH agent forwarding. Pull from the old
# pod over its existing SSH listener and permitted same-namespace ingress.
# No trainer launches or source patches.
set -euo pipefail
source_host=baseten@192.168.53.121
paths=(
  /root/glm53-131k-bf16-20260909
  /root/glm53-131k-ep1-rootcause-20260909
  /root/glm53-131k-fixes-20260909
  /root/glm53-131k-fsdp-cp8ep1-20260909
  /root/glm53-131k-fsdp-parity-20260909
  /root/glm53-1d2m-262k-20260909
  /root/glm53-1d3m-262k-20260909
  /root/glm53-2d2m-262k-20260909
  /root/glm53-full-131k-fsdp-20260909
  /root/glm53-full-131k-fsdp-doublebuf-20260910
  /root/glm53-full-131k-fsdp-headmem-20260910
  /root/glm53-full-131k-fsdp-noprefetch-20260910
  /root/glm53-lmhead-memory-probe-20260910
  /root/glm53-main-262k-20260909
  /root/glm53-pr1355-repro-20260910
  /root/trainers
  /root/qwen06-profile
  /root/lmhead-candidate.py
  /root/.cache/qwen-staging
  /root/.devbox-venvs/server
  /root/.triton
  /root/.tilelang
  /root/.deepep
)
sources=()
for path in "${paths[@]}"; do sources+=("$source_host:$path"); done
transport='ssh -p 2222 -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ControlPath=none'
rsync -aHR --info=stats2 --partial -e "$transport" "${sources[@]}" /
# Read and compare all file contents, not just timestamps and file sizes.
rsync -aHRnc --itemize-changes -e "$transport" "${sources[@]}" / \
  | tee /tmp/profiler-migration-differences.txt
test ! -s /tmp/profiler-migration-differences.txt
echo 'MIGRATION_VERIFIED: all selected files match by checksum'
