#!/usr/bin/env python3

import json
import sys


def merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def intersection(left, right):
    total = 0
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if start < end:
            total += end - start
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


for trace_path in sys.argv[1:]:
    with open(trace_path) as trace_file:
        trace = json.load(trace_file)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    communication = []
    compute = []
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "kernel":
            continue
        start = event["ts"]
        interval = (start, start + event["dur"])
        if "hybrid_ep::" in event.get("name", ""):
            communication.append(interval)
        else:
            compute.append(interval)
    communication = merge(communication)
    compute = merge(compute)
    communication_us = sum(end - start for start, end in communication)
    overlap_us = intersection(communication, compute)
    print(
        f"{trace_path}: communication={communication_us / 1000:.3f} ms "
        f"overlap={overlap_us / 1000:.3f} ms "
        f"hidden={100 * overlap_us / communication_us:.2f}%"
    )
