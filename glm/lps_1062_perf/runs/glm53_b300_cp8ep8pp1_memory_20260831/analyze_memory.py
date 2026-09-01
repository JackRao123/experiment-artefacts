#!/usr/bin/env python3
"""Summarize CUDA allocator snapshots from the two GLM-5.2 profile windows."""

from __future__ import annotations

import argparse
import collections
import json
import pickle
from pathlib import Path


def allocation_site(event: dict) -> str:
    frames = event.get("frames") or []
    if not frames:
        return "<no Python stack>"
    frame = frames[0]
    return f"{frame.get('name')}@{Path(frame.get('filename', '')).name}:{frame.get('line')}"


def category(event: dict) -> str:
    stack = " ".join(
        f"{frame.get('name', '')} {frame.get('filename', '')}"
        for frame in event.get("frames") or []
    ).lower()
    if "chunked_lm_head" in stack or "loss.py" in stack or "cross_entropy" in stack:
        return "logits_and_loss"
    if "expert" in stack or "moe" in stack or "grouped" in stack:
        return "moe_experts"
    if "attention" in stack or "indexer" in stack or "dsa" in stack:
        return "attention_and_dsa"
    if "lora" in stack or "peft" in stack:
        return "lora"
    if "checkpoint" in stack or "recomput" in stack:
        return "recompute"
    if "optimizer" in stack or "optim" in stack:
        return "optimizer"
    if "distributed" in stack or "c10d" in stack or "nccl" in stack:
        return "distributed"
    if "packing" in stack or "datum" in stack or "batch" in stack:
        return "input_and_packing"
    return "other"


def summarize(path: Path) -> dict:
    rank = int(path.stem.removeprefix("memory.rank"))
    with path.open("rb") as handle:
        snapshot = pickle.load(handle)

    segments = snapshot.get("segments", [])
    blocks = [block for segment in segments for block in segment.get("blocks", [])]
    final_active = sum(
        block.get("requested_size", block["size"])
        for block in blocks
        if block["state"] in {"active_allocated", "active_awaiting_free"}
    )
    final_reserved = sum(block["size"] for block in blocks)
    final_inactive = sum(block["size"] for block in blocks if block["state"] == "inactive")

    traces = snapshot.get("device_traces", [])
    events = traces[rank] if len(traces) > rank else []
    cumulative = 0
    peak_cumulative = 0
    peak_index = -1
    active_window_allocs: dict[int, dict] = {}
    peak_window_allocs: dict[int, dict] = {}
    first_time = events[0].get("time_us", 0) if events else 0

    for index, event in enumerate(events):
        action = event.get("action")
        if action == "alloc":
            cumulative += event["size"]
            active_window_allocs[event["addr"]] = event
        elif action == "free_requested":
            cumulative -= event["size"]
            active_window_allocs.pop(event["addr"], None)
        if cumulative > peak_cumulative:
            peak_cumulative = cumulative
            peak_index = index
            peak_window_allocs = dict(active_window_allocs)

    initial_active = final_active - cumulative
    peak_active = initial_active + peak_cumulative
    peak_event = events[peak_index] if peak_index >= 0 else None

    by_category = collections.Counter()
    by_site = collections.Counter()
    for event in peak_window_allocs.values():
        by_category[category(event)] += event["size"]
        by_site[allocation_site(event)] += event["size"]

    return {
        "file": path.name,
        "rank": rank,
        "initial_active_bytes": initial_active,
        "peak_active_bytes": peak_active,
        "peak_growth_bytes": peak_cumulative,
        "final_active_bytes": final_active,
        "final_reserved_bytes": final_reserved,
        "final_inactive_bytes": final_inactive,
        "final_inactive_fraction": final_inactive / final_reserved if final_reserved else 0,
        "peak_time_seconds": (
            (peak_event.get("time_us", first_time) - first_time) / 1_000_000
            if peak_event
            else None
        ),
        "event_count": len(events),
        "oom_events": sum(event.get("action") == "oom" for event in events),
        "peak_transient_categories": dict(by_category.most_common()),
        "peak_transient_sites": [
            {"site": site, "bytes": size} for site, size in by_site.most_common(15)
        ],
        "allocator_settings": snapshot.get("allocator_settings"),
    }


def aggregate(rows: list[dict]) -> dict:
    keys = (
        "initial_active_bytes",
        "peak_active_bytes",
        "peak_growth_bytes",
        "final_active_bytes",
        "final_reserved_bytes",
        "final_inactive_bytes",
    )
    result = {"ranks": len(rows)}
    for key in keys:
        values = [row[key] for row in rows]
        result[key] = {
            "min": min(values),
            "mean": sum(values) / len(values),
            "max": max(values),
        }
    peak_row = max(rows, key=lambda row: row["peak_active_bytes"])
    result["max_peak_rank"] = peak_row["rank"]
    result["max_peak_transient_categories"] = peak_row["peak_transient_categories"]
    result["max_peak_transient_sites"] = peak_row["peak_transient_sites"]
    result["oom_events"] = sum(row["oom_events"] for row in rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {}
    for label in ("131k-memory", "262k-memory"):
        rows = [summarize(path) for path in sorted((args.root / label).glob("memory.rank*.pickle"))]
        result[label] = {"aggregate": aggregate(rows), "per_rank": rows}

    args.output.write_text(json.dumps(result, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
