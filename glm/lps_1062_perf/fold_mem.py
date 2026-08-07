#!/usr/bin/env python3
"""Fold per-GPU max memory (from poll_gpu_mem.sh CSVs) into a bench result json."""

import csv
import json
import sys
from pathlib import Path

OUT_DIR = Path("/root/.cache/user_artifacts/lps1062_bench")


def main() -> None:
    label = sys.argv[1]
    result_path = OUT_DIR / f"{label}.json"
    mem_dir = OUT_DIR / f"{label}_mem"

    per_gpu: dict[str, int] = {}
    for csv_path in sorted(mem_dir.glob("mem.*.csv")):
        node = csv_path.stem.replace("mem.", "")
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                key = f"{node}:gpu{int(row['idx']):d}"
                mib = int(row["used_mib"])
                if mib > per_gpu.get(key, -1):
                    per_gpu[key] = mib

    data = json.loads(result_path.read_text())
    data["aggregates"]["per_gpu_max_used_mib"] = per_gpu
    data["aggregates"]["max_gpu_used_mib"] = max(per_gpu.values(), default=0)
    result_path.write_text(json.dumps(data, indent=2))
    vals = sorted(per_gpu.values())
    print(
        f"MEM {label}: max {vals[-1]} MiB, min {vals[0]} MiB across {len(vals)} GPUs "
        f"(cap 275040)",
        flush=True,
    )


if __name__ == "__main__":
    main()
