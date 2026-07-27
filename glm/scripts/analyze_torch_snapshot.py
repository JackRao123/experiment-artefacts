#!/usr/bin/env python3
"""Attribute allocations in a torch CUDA memory snapshot by subsystem."""

from __future__ import annotations

import argparse
import collections
import json
import pickle
from pathlib import Path

GIB = 2**30


def classify(frames: list[dict]) -> str:
    names = {frame.get("name", "") for frame in frames}
    files = {frame.get("filename", "") for frame in frames}
    if "gather_from_sequence_parallel_region" in names and any(
        filename.endswith("/experimental_attention_variant/dsa.py")
        for filename in files
    ):
        return "dsa_cp_allgather"
    if any(
        name.startswith("_indexer_topk")
        or name in {"cute_dsl_topk_wrapper", "indexer_top_k_wrapper"}
        for name in names
    ):
        return "dsa_indexer_topk"
    if names & {
        "token_dispatch",
        "moe_chunk_sort_forward",
        "moe_chunk_sort_backward",
        "sort_chunks_by_map",
        "routed_experts_compute",
    }:
        return "moe_dispatch_and_experts"
    if any("attention" in filename and "experimental_attention_variant" in filename for filename in files):
        return "dsa_other"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    with args.snapshot.open("rb") as file:
        snapshot = pickle.load(file)  # noqa: S301 - trusted local profiler output

    trace = snapshot["device_traces"][0]
    live: dict[int, tuple[int, str]] = {}
    live_by_category: collections.Counter[str] = collections.Counter()
    peak_live_by_category: collections.Counter[str] = collections.Counter()
    allocation_count: collections.Counter[str] = collections.Counter()
    allocated_total: collections.Counter[str] = collections.Counter()
    max_single: collections.Counter[str] = collections.Counter()
    stack_totals: collections.Counter[tuple[str, str]] = collections.Counter()

    for event in trace:
        action = event["action"]
        address = event.get("addr")
        if action == "alloc":
            size = event["size"]
            category = classify(event.get("frames", []))
            live[address] = (size, category)
            live_by_category[category] += size
            peak_live_by_category[category] = max(
                peak_live_by_category[category], live_by_category[category]
            )
            allocation_count[category] += 1
            allocated_total[category] += size
            max_single[category] = max(max_single[category], size)
            frames = event.get("frames", [])
            leaf = frames[0].get("name", "<unknown>") if frames else "<unknown>"
            stack_totals[(category, leaf)] += size
        elif action == "free_requested" and address in live:
            size, category = live.pop(address)
            live_by_category[category] -= size

    categories = sorted(allocation_count)
    result = {
        "trace_event_count": len(trace),
        "trace_truncated": len(trace) == 500_000,
        "trace_duration_s": (
            (trace[-1]["time_us"] - trace[0]["time_us"]) / 1e6 if trace else 0.0
        ),
        "categories": {
            category: {
                "allocation_count": allocation_count[category],
                "allocated_total_gib": allocated_total[category] / GIB,
                "max_single_allocation_gib": max_single[category] / GIB,
                "peak_live_from_observed_allocations_gib": (
                    peak_live_by_category[category] / GIB
                ),
            }
            for category in categories
        },
        "largest_leaf_totals": [
            {
                "category": category,
                "leaf": leaf,
                "allocated_total_gib": size / GIB,
            }
            for (category, leaf), size in stack_totals.most_common(20)
        ],
    }
    print(json.dumps(result, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
