#!/usr/bin/env python3
"""Duplicate the 0D1M snapshot's MoE layer to build a two-layer proxy."""

import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


source = Path(sys.argv[1])
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)

tensors = {}
with safe_open(source / "model.safetensors", framework="pt", device="cpu") as handle:
    keys = list(handle.keys())
    for index, key in enumerate(keys, 1):
        tensor = handle.get_tensor(key)
        tensors[key] = tensor
        if key.startswith("model.layers.0."):
            tensors[key.replace("model.layers.0.", "model.layers.1.", 1)] = tensor.clone()
        if index % 200 == 0:
            print(f"[{index}/{len(keys)}]", flush=True)

destination = output / "model.safetensors"
total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
print(
    f"writing {len(tensors)} tensors, {total_bytes / 2**30:.2f} GiB -> {destination}",
    flush=True,
)
save_file(tensors, str(destination), metadata={"format": "pt"})

for auxiliary in (
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
):
    source_file = source / auxiliary
    if source_file.exists():
        (output / auxiliary).write_bytes(source_file.read_bytes())

print("DONE", flush=True)
