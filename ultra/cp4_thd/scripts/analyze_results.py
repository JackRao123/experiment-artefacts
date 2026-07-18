"""Fold sft_driver JSONs + NVML CSVs into the compact profiling tables.

    python analyze_results.py --out-dir <dir with *.json and nvml_*.csv> \
        --profile-glob "ultra_cp4_prof_*.json" --emit results.csv

Produces one row per (label, step): loss, tokens, fb/optim seconds, TPS/GPU,
max-rank allocator peaks from /memory_stats, and the all-GPU NVML window
(hottest / min / median of per-GPU max) between step start and end when NVML
CSVs cover the window.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from pathlib import Path


def load_nvml(out_dir: Path) -> list[tuple[float, str, int, int]]:
    rows: list[tuple[float, str, int, int]] = []
    for path in glob.glob(str(out_dir / "nvml_*.csv")):
        with open(path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 4:
                    continue
                try:
                    rows.append(
                        (float(parts[0]), parts[1], int(parts[2]), int(parts[3]))
                    )
                except ValueError:
                    continue
    rows.sort()
    return rows


def nvml_window(
    nvml: list[tuple[float, str, int, int]], t0: float, t1: float
) -> dict | None:
    per_gpu: dict[tuple[str, int], int] = {}
    for ts, host, idx, mem in nvml:
        if t0 <= ts <= t1:
            key = (host, idx)
            per_gpu[key] = max(per_gpu.get(key, 0), mem)
    if not per_gpu:
        return None
    values = sorted(per_gpu.values())
    return {
        "nvml_gpus": len(values),
        "nvml_max_mib": values[-1],
        "nvml_min_mib": values[0],
        "nvml_median_mib": statistics.median(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--profile-glob", default="*.json")
    parser.add_argument("--gpus", type=int, default=32)
    parser.add_argument("--emit", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    nvml = load_nvml(out_dir)
    rows = []
    for path in sorted(glob.glob(str(out_dir / args.profile_glob))):
        r = json.loads(Path(path).read_text())
        doc_tokens = sum(r["doc_lens"])
        for s in r["steps"]:
            row = {
                "label": r["label"],
                "step": s["step"],
                "doc_tokens": doc_tokens,
                "loss": round(s["loss"], 6),
                "loss_tokens": s.get("optim_metrics", {}).get("num_loss_tokens"),
                "grad_norm": s.get("optim_metrics", {}).get("grad_norm"),
                "fb_seconds": round(s["fb_seconds"], 2),
                "optim_seconds": round(s.get("optim_seconds", 0.0), 2),
                "tps_per_gpu": round(
                    doc_tokens / max(s["fb_seconds"], 1e-9) / args.gpus, 1
                ),
            }
            mem = s.get("memory")
            if mem:
                row["max_allocated_gib"] = round(mem["max_allocated_gib"], 1)
                row["max_reserved_gib"] = round(mem["max_reserved_gib"], 1)
            wall = s.get("wall_window")
            if wall and nvml:
                w = nvml_window(nvml, wall[0], wall[1])
                if w:
                    row.update(w)
            rows.append(row)

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "label", k))
    if args.emit:
        with open(args.emit, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.emit} ({len(rows)} rows)")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
