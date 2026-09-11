"""Attribute standalone GEMM GPU work to NVTX phases using CUDA correlations."""
import argparse
import bisect
import collections
import json
import sqlite3
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("trace", type=Path)
args = parser.parse_args()
db = sqlite3.connect(args.trace)
ranges = db.execute("SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'CP8EP%/%' AND end IS NOT NULL ORDER BY start").fetchall()
runtime = db.execute("SELECT start,end,correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start").fetchall()
starts = [r[0] for r in runtime]
kernels = db.execute("SELECT rowid,start,end,correlationId,streamId FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
by_correlation = collections.defaultdict(list)
for kernel in kernels:
    by_correlation[kernel[3]].append(kernel)


def union_duration(intervals):
    total, left, right = 0, None, None
    for start, end in sorted(intervals):
        if left is None:
            left, right = start, end
        elif start <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = start, end
    return total + (right - left if left is not None else 0)


records, assigned = [], set()
for start, end, name in ranges:
    launches = runtime[bisect.bisect_left(starts, start):bisect.bisect_right(starts, end)]
    selected = {k[0]: k for r in launches if r[1] <= end for k in by_correlation[r[2]]}
    assert selected and not (assigned & selected.keys()), name
    assigned.update(selected)
    intervals = [(k[1], k[2]) for k in selected.values()]
    lo, hi = min(x[0] for x in intervals), max(x[1] for x in intervals)
    busy = union_duration(intervals)
    hardware = {}
    entry_hardware = {}
    for label in ("SMs Active [Throughput %]", "Tensor Active [Throughput %]"):
        samples = [r[0] for r in db.execute(
            "SELECT g.value FROM GPU_METRICS g JOIN TARGET_INFO_GPU_METRICS m "
            "ON g.typeId=m.typeId AND g.metricId=m.metricId "
            "WHERE m.metricName=? AND g.timestamp BETWEEN ? AND ?", (label, lo, hi))]
        hardware[label] = {"mean": statistics.mean(samples) if samples else None, "samples": len(samples)}
        # Forward begins after a preceding device synchronization in this probe.
        # Backward CPU submission can overlap pending forward GPU work, so do
        # not interpret its CPU-entry window as an isolated backward interval.
        if name.endswith("/forward"):
            entry_samples = [r[0] for r in db.execute(
                "SELECT g.value FROM GPU_METRICS g JOIN TARGET_INFO_GPU_METRICS m "
                "ON g.typeId=m.typeId AND g.metricId=m.metricId "
                "WHERE m.metricName=? AND g.timestamp BETWEEN ? AND ?", (label, start, hi))]
            entry_hardware[label] = {"mean": statistics.mean(entry_samples) if entry_samples else None, "samples": len(entry_samples)}
    records.append({"range": name, "kernels": len(selected), "streams": len({k[4] for k in selected.values()}),
                    "cpu_enqueue_ms": (end-start)/1e6, "gpu_envelope_ms": (hi-lo)/1e6,
                    "gpu_kernel_busy_ms": busy/1e6, "gpu_kernel_idle_ms": (hi-lo-busy)/1e6,
                    "summed_kernel_ms": sum(b-a for a,b in intervals)/1e6, "hardware": hardware,
                    "cpu_entry_to_first_kernel_ms": (lo-start)/1e6,
                    "cpu_entry_to_last_kernel_ms": (hi-start)/1e6,
                    "forward_entry_window_hardware": entry_hardware})
assert len(assigned) == len(kernels), (len(assigned), len(kernels), "unattributed kernels")
grouped = collections.defaultdict(list)
for record in records:
    grouped[record["range"]].append(record)
summary = {}
for name, items in grouped.items():
    row = {key: statistics.mean(r[key] for r in items) for key in
           ("kernels", "streams", "cpu_enqueue_ms", "gpu_envelope_ms", "gpu_kernel_busy_ms", "gpu_kernel_idle_ms", "summed_kernel_ms", "cpu_entry_to_first_kernel_ms", "cpu_entry_to_last_kernel_ms")}
    for label in items[0]["hardware"]:
        samples = [r["hardware"][label] for r in items if r["hardware"][label]["samples"]]
        row[label] = sum(s["mean"]*s["samples"] for s in samples)/sum(s["samples"] for s in samples) if samples else None
    summary[name] = row
output = {"trace": str(args.trace), "kernels": len(kernels), "attributed_kernels": len(assigned), "ranges": records, "summary": summary}
args.trace.with_suffix(".analysis.json").write_text(json.dumps(output, indent=2))
print(json.dumps(summary, indent=2))
