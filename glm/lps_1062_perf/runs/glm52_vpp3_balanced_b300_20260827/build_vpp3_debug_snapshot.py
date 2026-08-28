#!/usr/bin/env python3
"""Hardlink the six-layer GLM debug snapshot and disable DSA index sharing."""

import argparse
import json
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"snapshot already exists: {args.output}")
    args.output.mkdir(parents=True)

    for source_file in args.source.iterdir():
        output_file = args.output / source_file.name
        if source_file.name == "config.json":
            shutil.copy2(source_file, output_file)
        elif source_file.is_file():
            try:
                os.link(source_file, output_file)
            except OSError:
                shutil.copy2(source_file, output_file)

    config_path = args.output / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("num_hidden_layers") != 6:
        raise ValueError("expected a six-layer debug snapshot")
    if config.get("indexer_types") != ["full"] * 6:
        raise ValueError("every debug layer must carry a full DSA indexer")
    config["index_topk_freq"] = 1
    config["index_skip_topk_offset"] = 0
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
