#!/usr/bin/env python3
"""Build a real-weight GLM-5.2 FP8 snapshot with one dense and one MoE layer."""

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
    args = parser.parse_args()

    source: Path = args.source
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    weight_map = json.loads(
        (source / "model.safetensors.index.json").read_text()
    )["weight_map"]

    selected_by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        if not key.startswith("model.layers."):
            selected_by_shard[shard].append(key)
        elif key.startswith("model.layers.0.") or key.startswith("model.layers.6."):
            selected_by_shard[shard].append(key)

    tensors = {}
    total_bytes = 0
    selected_count = sum(len(keys) for keys in selected_by_shard.values())
    loaded_count = 0
    for shard, keys in sorted(selected_by_shard.items()):
        with safe_open(str(source / shard), framework="pt", device="cpu") as handle:
            for key in keys:
                output_key = key.replace("model.layers.6.", "model.layers.1.", 1)
                tensor = handle.get_tensor(key)
                tensors[output_key] = tensor
                total_bytes += tensor.numel() * tensor.element_size()
                loaded_count += 1
        print(
            f"loaded {loaded_count}/{selected_count} tensors "
            f"({total_bytes / 2**30:.2f} GiB)",
            flush=True,
        )

    destination = output / "model.safetensors"
    print(
        f"writing {len(tensors)} tensors ({total_bytes / 2**30:.2f} GiB) "
        f"to {destination}",
        flush=True,
    )
    save_file(tensors, str(destination), metadata={"format": "pt"})

    config = json.loads((source / "config.json").read_text())
    config.update(
        {
            "num_hidden_layers": 2,
            "first_k_dense_replace": 1,
            "mlp_layer_types": ["dense", "sparse"],
            "indexer_types": ["full", "full"],
            "num_nextn_predict_layers": 0,
        }
    )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
    ):
        source_file = source / name
        if source_file.exists():
            shutil.copy2(source_file, output / name)

    print("done", flush=True)


if __name__ == "__main__":
    main()
