#!/usr/bin/env python3
"""Clone a 0D1M GLM snapshot's MoE layer to an arbitrary layer count."""

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--config-template", type=Path, required=True)
    args = parser.parse_args()

    if args.layers < 1:
        raise ValueError("--layers must be at least 1")
    if (args.output / "model.safetensors").exists():
        raise FileExistsError(f"snapshot already exists: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    tensors = {}
    source_model = args.source / "model.safetensors"
    with safe_open(source_model, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        layer_indices = {
            int(key.split(".")[2]) for key in keys if key.startswith("model.layers.")
        }
        if layer_indices != {0}:
            raise ValueError(f"expected only source layer 0, found {sorted(layer_indices)}")

        for index, key in enumerate(keys, 1):
            tensor = handle.get_tensor(key)
            tensors[key] = tensor
            if key.startswith("model.layers.0."):
                for layer in range(1, args.layers):
                    target = key.replace("model.layers.0.", f"model.layers.{layer}.", 1)
                    tensors[target] = tensor.clone()
            if index % 200 == 0:
                print(f"[{index}/{len(keys)}]", flush=True)

    destination = args.output / "model.safetensors"
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
        source_file = args.source / auxiliary
        if source_file.exists():
            (args.output / auxiliary).write_bytes(source_file.read_bytes())

    config = json.loads(args.config_template.read_text())
    config["num_hidden_layers"] = args.layers
    config["indexer_types"] = ["full"] * args.layers
    config["mlp_layer_types"] = ["sparse"] * args.layers
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
