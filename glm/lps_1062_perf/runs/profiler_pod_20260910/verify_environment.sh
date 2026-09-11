#!/usr/bin/env bash
set -euo pipefail
source /root/.cache/user_artifacts/devboxes/32vj99q/env.sh
export PYTHONDONTWRITEBYTECODE=1
repo=/root/glm53-pr1355-repro-20260910/trainers
bridge="$repo/server-megatron-bridge/vendor/megatron-bridge"
core="$bridge/3rdparty/Megatron-LM"
for checkout in "$repo" "$bridge" "$core"; do
  git -C "$checkout" rev-parse HEAD
  git -C "$checkout" diff --exit-code --quiet
done
export PYTHONPATH="$repo/server-megatron-bridge/src:$bridge/src:$core${PYTHONPATH:+:$PYTHONPATH}"
/root/.devbox-venvs/server/bin/python - <<'PY'
import torch
import transformer_engine
import triton
from megatron.core.extensions import frozen_grouped_mm
from megatron.bridge.models.hf_pretrained import state

print("torch:", torch.__version__)
print("Transformer Engine:", transformer_engine.__version__)
print("Triton:", triton.__version__)
print("grouped-MM implementation:", frozen_grouped_mm.__file__)
print("checkpoint reader:", state.__file__)
assert torch.cuda.device_count() == 8
for index in range(torch.cuda.device_count()):
    value = torch.ones(1, device=f"cuda:{index}") * 2
    assert value.item() == 2
    properties = torch.cuda.get_device_properties(index)
    print(index, properties.name, properties.multi_processor_count, properties.total_memory)
print("ENVIRONMENT_VERIFIED")
PY
nsys --version
ncu --version
