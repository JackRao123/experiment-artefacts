#!/usr/bin/env python3
"""mem_plateau.py — read the plateau, not the whole-run max (conway, LPS-1062).

The poller samples nvidia-smi reserved memory every 2 s on every GPU of every
node. Two rules from the campaign shape how it must be read:

  1. Torch-reserved creeps ~+21 GiB over a run's early steps before
     flattening, so an early read understates the peak. The plateau is the
     max over the LAST TWO step-length windows, and those two windows must
     agree within ~2 GiB or the run has not plateaued.
  2. nvidia-smi reserved and torch-reserved are DIFFERENT metrics, differing
     by ~12 GiB of NCCL and driver overhead. They are never mixed. Everything
     here is the nvidia-smi number, which is the one that decides an OOM.

Usage:
    mem_plateau.py mem.node0.csv [mem.node1.csv ...] [--window 24] [--tail 2]
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

GIB = 1024.0  # MiB per GiB


def read(path: Path) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append((int(r["ts"]), int(r["idx"]), int(r["used_mib"])))
            except (ValueError, TypeError, KeyError):
                continue
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--window", type=float, default=24.0,
                    help="step length in seconds (d2 @131k measured ~24 s)")
    ap.add_argument("--tail", type=int, default=2,
                    help="how many trailing windows form the plateau read")
    args = ap.parse_args()

    print(f"# GPU memory plateau — nvidia-smi reserved, {args.tail} trailing "
          f"windows of {args.window:g} s\n")
    print("| node | GPUs | whole-run max GiB | last window max | prev window max | "
          "agree within 2 GiB? | per-GPU spread in last window |")
    print("|---|---|---|---|---|---|---|")

    for c in args.csvs:
        p = Path(c)
        rows = read(p)
        if not rows:
            print(f"| {p.name} | - | (no rows) | | | | |")
            continue
        t_end = max(r[0] for r in rows)
        gpus = sorted({r[1] for r in rows})

        def win_max(lo: float, hi: float) -> float:
            vals = [r[2] for r in rows if lo <= r[0] <= hi]
            return max(vals) / GIB if vals else float("nan")

        last = win_max(t_end - args.window, t_end)
        prev = win_max(t_end - 2 * args.window, t_end - args.window)
        allmax = max(r[2] for r in rows) / GIB
        agree = abs(last - prev) <= 2.0

        per_gpu = collections.defaultdict(float)
        for ts, idx, mib in rows:
            if ts >= t_end - args.window:
                per_gpu[idx] = max(per_gpu[idx], mib / GIB)
        spread = (max(per_gpu.values()) - min(per_gpu.values())) if per_gpu else float("nan")

        node = p.name.replace("mem.", "").replace(".csv", "")
        print(f"| {node} | {len(gpus)} | {allmax:.1f} | {last:.1f} | {prev:.1f} | "
              f"{'YES' if agree else 'NO — not plateaued, extend the run'} | {spread:.1f} |")

    print("\nThe plateau read is the 'last window max' column, and it is only "
          "valid where the agreement column says YES.")
    print("Effective ceiling ~247.7 GiB = the 267.7 GiB card less the ~20 GiB "
          "cold-allocator burst.")


if __name__ == "__main__":
    main()
