"""Read safetensors headers only; no model load or tensor payload reads."""

import argparse
import json
import math
import re
import struct
from collections import Counter
from pathlib import Path


def inventory(snapshot: Path) -> dict:
    tensors = {}
    for shard in sorted(snapshot.glob("*.safetensors")):
        with shard.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_size))
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"Duplicate tensor {name}")
            tensors[name] = {"shape": spec["shape"], "dtype": spec["dtype"]}
    if not tensors:
        raise ValueError(f"No safetensors in {snapshot}")
    groups = Counter()
    for name, spec in tensors.items():
        pattern = re.sub(r"\.(layers|experts)\.\d+\.", r".\1.*.", name)
        groups[(pattern, tuple(spec["shape"]), spec["dtype"])] += 1
    return {
        "snapshot": str(snapshot),
        "config": json.loads((snapshot / "config.json").read_text()),
        "tensor_count": len(tensors),
        "checkpoint_parameters": sum(math.prod(t["shape"]) for t in tensors.values()),
        "groups": [
            {"pattern": p, "shape": s, "dtype": d, "count": n}
            for (p, s, d), n in sorted(groups.items())
        ],
        "tensors": tensors,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = inventory(args.snapshot)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "tensors"}, indent=2))
