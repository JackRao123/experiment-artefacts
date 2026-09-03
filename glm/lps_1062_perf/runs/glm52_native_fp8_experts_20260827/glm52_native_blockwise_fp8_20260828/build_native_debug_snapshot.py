#!/usr/bin/env python3
"""Build a one-MoE-layer GLM-5.2 snapshot from native checkpoint tensors."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    weight_map = json.loads(
        (args.source / "model.safetensors.index.json").read_text()
    )["weight_map"]
    source_prefix = f"model.layers.{args.layer}."

    selected_by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        if not key.startswith("model.layers.") or key.startswith(source_prefix):
            selected_by_shard[shard].append(key)

    tensors = {}
    dtype_counts: dict[str, int] = defaultdict(int)
    total_bytes = 0
    for shard_name, keys in sorted(selected_by_shard.items()):
        with safe_open(
            args.source / shard_name,
            framework="pt",
            device="cpu",
        ) as shard:
            for key in keys:
                tensor = shard.get_tensor(key)
                output_key = key.replace(source_prefix, "model.layers.0.", 1)
                tensors[output_key] = tensor
                dtype_counts[str(tensor.dtype)] += 1
                total_bytes += tensor.numel() * tensor.element_size()

    args.output.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(args.output / "model.safetensors"),
        metadata={"format": "pt"},
    )
    shutil.copy2(args.config, args.output / "config.json")
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
    ):
        source = args.source / name
        if source.exists():
            shutil.copy2(source, args.output / name)

    manifest = {
        "source": str(args.source),
        "source_layer": args.layer,
        "tensor_count": len(tensors),
        "tensor_bytes": total_bytes,
        "dtype_counts": dict(sorted(dtype_counts.items())),
    }
    (args.output / "native_debug_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
