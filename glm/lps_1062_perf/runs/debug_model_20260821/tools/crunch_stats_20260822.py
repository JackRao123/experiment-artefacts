#!/usr/bin/env python3
"""Crunch the stat A/B result JSONs (bench_driver2c format)."""
import json
import statistics as st
import sys
from pathlib import Path

labels = [
    "stat-04-base-es-v1",
    "stat-04-off-knobsoff-v1",
    "stat-04-off-k6u-v1",
    "stat-04-off-k6u-noes-v1",
    "stat-04-base-noes-v1",
]
d = Path(sys.argv[1])
rows = {}
for lab in labels:
    p = d / f"{lab}.json"
    if not p.exists():
        print(f"{lab}: MISSING")
        continue
    j = json.loads(p.read_text())
    mains = [w for w in j["windows"] if w["phase"] == "main"]
    fb = [w["fb_elapsed_s"] for w in mains]
    tps = [w["fb_tps_per_gpu"] for w in mains]
    losses = [w["loss"] for w in j["windows"]]
    agg = j["aggregates"]
    rows[lab] = dict(fb=fb, tps=tps, losses=losses,
                     peak_alloc=agg["peak_gpu_memory_allocated_bytes"],
                     peak_res=agg["peak_gpu_memory_reserved_bytes"])
    print(f"{lab}:")
    print(f"  fb windows (s): {' '.join(f'{x:.3f}' for x in fb)}")
    print(f"  median {st.median(fb):.3f}s  mean {st.mean(fb):.3f}s  "
          f"stdev {st.stdev(fb):.3f}s  min {min(fb):.3f}  max {max(fb):.3f}")
    print(f"  median TPS {st.median(tps):.0f}  peak_alloc "
          f"{rows[lab]['peak_alloc']/2**30:.3f} GiB  peak_res "
          f"{rows[lab]['peak_res']/2**30:.3f} GiB")

def cmp(a, b, note):
    if a in rows and b in rows:
        ma, mb = st.median(rows[a]["fb"]), st.median(rows[b]["fb"])
        print(f"{note}: {ma:.3f} vs {mb:.3f} -> {100*(mb-ma)/ma:+.1f}% fb; "
              f"dpeak_alloc {(rows[b]['peak_alloc']-rows[a]['peak_alloc'])/2**30:+.3f} GiB")

print()
cmp("stat-04-base-es-v1", "stat-04-off-knobsoff-v1", "base->offload(legacy)")
cmp("stat-04-base-es-v1", "stat-04-off-k6u-v1", "base->offload(K6+unchained)")
cmp("stat-04-off-knobsoff-v1", "stat-04-off-k6u-v1", "legacy->K6+unchained")
cmp("stat-04-off-k6u-v1", "stat-04-off-k6u-noes-v1", "K6 ES->noES")
cmp("stat-04-base-es-v1", "stat-04-base-noes-v1", "base ES->noES")

# loss parity across arms at same window index
print("\nloss parity (max |delta| vs base-es per window index):")
if "stat-04-base-es-v1" in rows:
    base = rows["stat-04-base-es-v1"]["losses"]
    for lab in labels[1:]:
        if lab in rows:
            l = rows[lab]["losses"]
            deltas = [abs(a-b) for a, b in zip(base, l) if a is not None and b is not None]
            print(f"  {lab}: max {max(deltas):.2e}")
