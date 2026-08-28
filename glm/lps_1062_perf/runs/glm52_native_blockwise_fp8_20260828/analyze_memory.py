#!/usr/bin/env python3
"""Summarize allocator snapshots from one profile_driver memory window."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def summarize(path: Path) -> dict:
    with path.open("rb") as handle:
        snapshot = pickle.load(handle)

    segments = snapshot["segments"]
    final_reserved = sum(segment["total_size"] for segment in segments)
    final_allocated = sum(segment["allocated_size"] for segment in segments)
    events = [event for trace in snapshot["device_traces"] for event in trace]
    net_delta = sum(
        event["size"]
        if event["action"] == "alloc"
        else -event["size"]
        if event["action"] == "free_completed"
        else 0
        for event in events
    )
    current = final_allocated - net_delta
    peak = current
    for event in events:
        if event["action"] == "alloc":
            current += event["size"]
        elif event["action"] == "free_completed":
            current -= event["size"]
        peak = max(peak, current)

    return {
        "file": path.name,
        "event_count": len(events),
        "start_allocated_bytes": final_allocated - net_delta,
        "peak_allocated_bytes": peak,
        "final_allocated_bytes": final_allocated,
        "final_reserved_bytes": final_reserved,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()

    ranks = [summarize(path) for path in sorted(args.directory.glob("memory.rank*.pickle"))]
    result = {
        "directory": str(args.directory),
        "rank_count": len(ranks),
        "max_peak_allocated_bytes": max(rank["peak_allocated_bytes"] for rank in ranks),
        "max_final_reserved_bytes": max(rank["final_reserved_bytes"] for rank in ranks),
        "ranks": ranks,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
