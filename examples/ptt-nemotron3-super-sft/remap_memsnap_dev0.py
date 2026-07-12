#!/usr/bin/env python3
"""Rewrite per-rank CUDA memory snapshots so their data sits on device 0.

Each trainer rank runs on GPU index = LOCAL_RANK (rank % 8), so its
`_dump_snapshot()` pickle records segments/traces under that device id.
PyTorch's memory_viz shows one device at a time and defaults to device 0, so
snapshots for ranks whose data is on device 1..7 look blank. This moves each
file's populated device to index 0 (data preserved) so every file opens
directly in https://docs.pytorch.org/memory_viz .

Usage: python remap_memsnap_dev0.py <dir_with_memory.rankN.pickle> [--inplace]
Default writes to <dir>_dev0/; --inplace overwrites the originals.
Only depends on the stdlib pickle (no torch needed).
"""

import glob
import os
import pickle
import sys


def remap(path, out_path):
    with open(path, "rb") as f:
        s = pickle.load(f)
    dt = s.get("device_traces", [])
    if dt:
        idx = max(range(len(dt)), key=lambda i: len(dt[i]))
        s["device_traces"] = [dt[idx]]
    for seg in s.get("segments", []):
        seg["device"] = 0
    with open(out_path, "wb") as f:
        pickle.dump(s, f, protocol=2)


def main():
    d = sys.argv[1]
    inplace = "--inplace" in sys.argv[2:]
    out_dir = d if inplace else d.rstrip("/") + "_dev0"
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(d, "memory.rank*.pickle")))
    for f in files:
        remap(f, os.path.join(out_dir, os.path.basename(f)))
    print(f"remapped {len(files)} files -> {out_dir}")


if __name__ == "__main__":
    main()
