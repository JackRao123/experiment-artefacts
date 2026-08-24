#!/usr/bin/env python3
"""Build a tiny GLM-5.2 debug snapshot (1 dense + 1 MoE layer) with random bf16 weights.

Reads the real GLM-5.2-FP8 snapshot's index + safetensors headers (never the
weight data) to get exact key names/shapes/dtypes, keeps:
  - model.embed_tokens.weight, model.norm.weight, lm_head.weight
  - model.layers.0.*   (dense layer)
  - model.layers.3.*   (first MoE layer)  -> renamed to model.layers.1.*
Skips: *.weight_scale_inv (fp8 block scales; debug model is plain bf16),
       model.layers.78.* (MTP layer; debug config sets num_nextn_predict_layers=0).
FP8 dtypes become bf16; everything else keeps its real dtype.
"""
import json
import struct
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

SNAP = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

DTYPE_MAP = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I64": torch.int64,
    "I32": torch.int32,
    "U8": torch.uint8,
}

idx = json.load(open(SNAP / "model.safetensors.index.json"))
wm = idx["weight_map"]


def keep(key: str) -> bool:
    if "weight_scale_inv" in key:
        return False
    if not key.startswith("model.layers."):
        return True
    return key.startswith("model.layers.0.") or key.startswith("model.layers.3.")


selected = [k for k in wm if keep(k)]
print(f"selected {len(selected)} of {len(wm)} keys", flush=True)

# --- read shard headers only (a few KB per shard) for shape/dtype ---
header_cache: dict[str, dict] = {}


def shard_header(shard: str) -> dict:
    if shard not in header_cache:
        with open(SNAP / shard, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header_cache[shard] = json.loads(f.read(hlen))
        print(f"  header {shard}: {len(header_cache[shard])} tensors", flush=True)
    return header_cache[shard]


tensors: dict[str, torch.Tensor] = {}
total_bytes = 0
for i, key in enumerate(selected):
    info = shard_header(wm[key])[key]
    shape = info["shape"]
    dt = info["dtype"]
    out_dt = torch.bfloat16 if dt.startswith("F8") else DTYPE_MAP[dt]
    new_key = key.replace("model.layers.3.", "model.layers.1.", 1)
    t = (torch.randn(shape, dtype=torch.float32) * 0.02).to(out_dt)
    tensors[new_key] = t
    total_bytes += t.numel() * t.element_size()
    if (i + 1) % 200 == 0:
        print(f"  [{i + 1}/{len(selected)}] {total_bytes / 2**30:.2f} GiB so far", flush=True)

print(f"writing {len(tensors)} tensors, {total_bytes / 2**30:.2f} GiB -> {OUT}/model.safetensors", flush=True)
save_file(tensors, str(OUT / "model.safetensors"), metadata={"format": "pt"})

for aux in ["tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"]:
    src = SNAP / aux
    if src.exists():
        (OUT / aux).write_bytes(src.read_bytes())
        print(f"  copied {aux}", flush=True)

print("DONE", flush=True)
