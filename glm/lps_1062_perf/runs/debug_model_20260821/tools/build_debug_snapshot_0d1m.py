#!/usr/bin/env python3
"""Build a one-MoE-layer GLM-5.2 snapshot with random bf16 weights."""

import json
import struct
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

snapshot = Path(sys.argv[1])
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)

dtype_map = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I64": torch.int64,
    "I32": torch.int32,
    "U8": torch.uint8,
}

weight_map = json.loads(
    (snapshot / "model.safetensors.index.json").read_text()
)["weight_map"]


def keep(key: str) -> bool:
    if "weight_scale_inv" in key:
        return False
    if not key.startswith("model.layers."):
        return True
    return key.startswith("model.layers.6.")


selected = [key for key in weight_map if keep(key)]
print(f"selected {len(selected)} of {len(weight_map)} keys", flush=True)

headers: dict[str, dict] = {}


def shard_header(shard: str) -> dict:
    if shard not in headers:
        with (snapshot / shard).open("rb") as handle:
            (header_length,) = struct.unpack("<Q", handle.read(8))
            headers[shard] = json.loads(handle.read(header_length))
    return headers[shard]


tensors: dict[str, torch.Tensor] = {}
total_bytes = 0
for index, key in enumerate(selected, 1):
    info = shard_header(weight_map[key])[key]
    source_dtype = info["dtype"]
    output_dtype = (
        torch.bfloat16 if source_dtype.startswith("F8") else dtype_map[source_dtype]
    )
    output_key = key.replace("model.layers.6.", "model.layers.0.", 1)
    tensor = (torch.randn(info["shape"], dtype=torch.float32) * 0.02).to(
        output_dtype
    )
    tensors[output_key] = tensor
    total_bytes += tensor.numel() * tensor.element_size()
    if index % 200 == 0:
        print(
            f"[{index}/{len(selected)}] {total_bytes / 2**30:.2f} GiB",
            flush=True,
        )

destination = output / "model.safetensors"
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
    source = snapshot / auxiliary
    if source.exists():
        (output / auxiliary).write_bytes(source.read_bytes())

print("DONE", flush=True)
