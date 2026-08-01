#!/usr/bin/env bash
# LPS-1003 devbox bring-up: run ON THE LEADER of the 2x8 B300 devbox.
# Steps are idempotent; rerun freely. Stops before trainer start — start/wait
# are driven interactively via .devbox_up scripts.
set -euo pipefail

DU=/root/.cache/user_artifacts/.devbox_up
source /root/.cache/user_artifacts/env.sh   # HF_HOME, TRAINERS, PATH

step() { printf '\n=== %s ===\n' "$*"; }

# B300 stack lives on branch trainer-cuda13-sm103 (cu130), NOT main (cu128 —
# torch import fails on these nodes). Default ref = 0e0b65a, the commit the
# patched prod image trainer-cuda13-sm103-0e0b65a was built against (#814
# vendored wheel included via the 98cd395 main-merge) — the exact stack the
# nspvxlhu bump run executed.
REF="${TRAINERS_REF:-0e0b65a}"
step "1. trainers_main -> $REF"
cd "$TRAINERS"
git fetch origin
git checkout -q "$REF"
git log --oneline -1

step "2. rebuild server venv (applies cuDNN #814 patch)"
bash "$DU/server_venv.sh"
echo "server_venv rc=$?"

step "3. verify import on BOTH nodes (CPFS readdir staleness pitfall)"
# bump mtimes so worker readdir caches refresh
find "$TRAINERS/server/.venv/lib" -maxdepth 4 -name cuda -type d -exec touch {} + 2>/dev/null || true
srun --overlap --nodes="$BT_GROUP_SIZE" --ntasks="$BT_GROUP_SIZE" --ntasks-per-node=1 bash -lc '
  h=$(hostname)
  if ibv_devinfo >/dev/null 2>&1; then echo "$h: ibv ok ($(ibv_devinfo | grep -c PORT_ACTIVE) active ports)"; else echo "$h: IBV MISSING"; fi
  cd '"$TRAINERS"' && ./server/.venv/bin/python -c "
import torch, cuda.bindings
import megatron.core
print(\"$h: torch \" + torch.__version__ + \" ok, megatron ok\")" || echo "$h: PYTHON IMPORT FAILED"'

step "4. verify cuDNN FE patch present in venv"
"$TRAINERS/server/.venv/bin/python" - <<'EOF'
import importlib.metadata as md
v = md.version("nvidia-cudnn-frontend")
print("nvidia-cudnn-frontend:", v)
assert "dsatopk" in v, f"EXPECTED PATCHED WHEEL (…+dsatopk1), got {v}"
EOF

step "5. all nodes clean (no stray trainers)"
srun --overlap --nodes="$BT_GROUP_SIZE" --ntasks="$BT_GROUP_SIZE" --ntasks-per-node=1 bash -lc \
  'printf "%s procs=%s gpu_max=%sMiB\n" "$(hostname)" \
    "$(pgrep -fc "[d]p_worker.main|[t]orchrun" || true)" \
    "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)"'

step "6. GLM weights present"
ls /root/.cache/team_artifacts/huggingface/hub/models--zai-org--GLM-5.2-FP8/snapshots/
echo "BRINGUP PREP COMPLETE — stage configs then use $DU/start_trainer.sh"
