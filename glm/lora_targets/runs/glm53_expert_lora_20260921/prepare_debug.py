"""Extract one dense and one routed-expert layer from a GLM DSA checkpoint."""

import argparse
import json
import re
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    config = json.loads((args.source / "config.json").read_text())
    assert config["model_type"] == "glm_moe_dsa"
    moe_layer = config["first_k_dense_replace"]
    layers = {0: 0, moe_layer: 1}
    index = json.loads((args.source / "model.safetensors.index.json").read_text())
    selected = {}
    for key, shard in index["weight_map"].items():
        match = re.match(r"model.layers.(\d+)\.", key)
        if match:
            old = int(match[1])
            if old not in layers:
                continue
            renamed = key.replace(
                f"model.layers.{old}.", f"model.layers.{layers[old]}.", 1
            )
        else:
            renamed = key
        selected.setdefault(shard, []).append((key, renamed))
    for shard in selected:
        if not (args.source / shard).is_file():
            raise FileNotFoundError(args.source / shard)
    args.destination.mkdir(parents=True, exist_ok=False)
    result_index = {}
    total_size = 0
    for number, (shard, keys) in enumerate(sorted(selected.items())):
        output_name = f"debug-{number:05d}.safetensors"
        with safe_open(args.source / shard, framework="pt", device="cpu") as source:
            tensors = {renamed: source.get_tensor(key) for key, renamed in keys}
        save_file(tensors, args.destination / output_name, metadata={"format": "pt"})
        total_size += sum(t.numel() * t.element_size() for t in tensors.values())
        result_index.update({key: output_name for key in tensors})
        del tensors
    for key, value in list(config.items()):
        if isinstance(value, list) and len(value) == config["num_hidden_layers"]:
            config[key] = [value[layer] for layer in layers]
    config.update(
        num_hidden_layers=2, first_k_dense_replace=1, num_nextn_predict_layers=0
    )
    (args.destination / "config.json").write_text(json.dumps(config, indent=2))
    (args.destination / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": result_index},
            indent=2,
        )
    )
    for path in args.source.iterdir():
        if path.name.startswith(
            ("tokenizer", "special_tokens", "chat_template", "generation_config")
        ):
            shutil.copyfile(path, args.destination / path.name)
    (args.destination / "debug_provenance.json").write_text(
        json.dumps({"source": str(args.source), "layer_map": layers}, indent=2)
    )


if __name__ == "__main__":
    main()
