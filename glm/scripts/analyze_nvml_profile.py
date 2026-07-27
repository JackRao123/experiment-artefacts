#!/usr/bin/env python3
"""Summarize high-resolution NVML samples around a trainer operation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

GIB = 2**30


def median_between(
    samples: list[tuple[int, int]], start_ns: int, end_ns: int
) -> float | None:
    values = [used for timestamp, used in samples if start_ns <= timestamp <= end_ns]
    return statistics.median(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--baseline-ms", type=float, default=1000.0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    start_ns = int((args.result_dir / "driver_start_ns").read_text())
    end_ns = int((args.result_dir / "driver_end_ns").read_text())
    baseline_ns = int(args.baseline_ms * 1_000_000)
    by_gpu: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)

    for path in sorted((args.result_dir / "nvml").glob("*.csv")):
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                by_gpu[(path.stem, int(row["gpu"]))].append(
                    (int(row["time_ns"]), int(row["used_bytes"]))
                )

    rows = []
    for (host, gpu), samples in sorted(by_gpu.items()):
        samples.sort()
        active = [(timestamp, used) for timestamp, used in samples if start_ns <= timestamp <= end_ns]
        if not active:
            continue
        peak_time_ns, peak_bytes = max(active, key=lambda sample: sample[1])
        pre_bytes = median_between(samples, start_ns - baseline_ns, start_ns)
        post_bytes = median_between(samples, end_ns, end_ns + baseline_ns)
        active_min_bytes = min(used for _, used in active)
        settled_bytes = max(
            value for value in (pre_bytes, post_bytes) if value is not None
        )
        row = {
            "host": host,
            "gpu": gpu,
            "pre_gib": pre_bytes / GIB if pre_bytes is not None else None,
            "active_min_gib": active_min_bytes / GIB,
            "peak_gib": peak_bytes / GIB,
            "post_gib": post_bytes / GIB if post_bytes is not None else None,
            "peak_above_pre_gib": (
                (peak_bytes - pre_bytes) / GIB if pre_bytes is not None else None
            ),
            "transient_above_settled_gib": (peak_bytes - settled_bytes) / GIB,
            "peak_offset_s": (peak_time_ns - start_ns) / 1e9,
        }
        rows.append(row)

    if not rows:
        raise SystemExit("no samples overlap the driver interval")

    hottest = max(rows, key=lambda row: row["peak_gib"])
    largest_growth = max(rows, key=lambda row: row["peak_above_pre_gib"])
    largest_transient = max(rows, key=lambda row: row["transient_above_settled_gib"])
    summary = {
        "driver_duration_s": (end_ns - start_ns) / 1e9,
        "gpu_count": len(rows),
        "hottest": hottest,
        "largest_growth": largest_growth,
        "largest_transient": largest_transient,
        "peak_gib_min": min(row["peak_gib"] for row in rows),
        "peak_gib_median": statistics.median(row["peak_gib"] for row in rows),
        "peak_gib_max": max(row["peak_gib"] for row in rows),
        "rows": rows,
    }
    print(json.dumps(summary, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
