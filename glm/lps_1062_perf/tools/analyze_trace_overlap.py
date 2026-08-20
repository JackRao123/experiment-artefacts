"""Kineto trace GPU-busy analysis — pipeline vs serial discriminator.

Question (maxwell): in the traced fb window, do the node's GPUs stay busy
~80% (1F1B steady state, bubble 1/m at PP2/d4) or ~50% (stages strictly
alternate = serial)? Node-0-only trace: stage-0 busy fraction answers it.

Prints: window bounds, per-cat busy union, busy fraction, and a binned
ASCII timeline so alternation is visible by eye.
"""
import json
import sys
from collections import defaultdict

path = sys.argv[1]
with open(path) as f:
    trace = json.load(f)

events = trace["traceEvents"] if isinstance(trace, dict) else trace

# GPU-side events: kernels + memcpy/memset. Kineto cat names: "kernel",
# "gpu_memcpy", "gpu_memset". (cuda_runtime/cuda_driver are host-side.)
busy_cats = {"kernel", "gpu_memcpy", "gpu_memset"}
iv = defaultdict(list)  # cat -> [(start, end)]
nccl = []
for e in events:
    if e.get("ph") != "X":
        continue
    cat = e.get("cat", "")
    if cat in busy_cats:
        s, d = e["ts"], e["dur"]
        iv[cat].append((s, s + d))
        if "nccl" in e.get("name", "").lower():
            nccl.append((s, s + d, e["name"][:60]))

if not iv:
    print("no GPU events found; cats present:", sorted({e.get('cat') for e in events if e.get('ph') == 'X'}))
    sys.exit(1)

def merge(intervals):
    intervals = sorted(intervals)
    out = []
    for s, e in intervals:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out

all_iv = merge([x for v in iv.values() for x in v])
t0, t1 = all_iv[0][0], all_iv[-1][1]
window = t1 - t0
busy = sum(e - s for s, e in all_iv)
print(f"window: {window/1e6:.2f}s  GPU-busy union: {busy/1e6:.2f}s  busy fraction: {busy/window*100:.1f}%")
for cat, v in sorted(iv.items()):
    m = merge(v)
    print(f"  {cat}: {sum(e-s for s,e in m)/1e6:.2f}s busy over {len(v)} events")
if nccl:
    m = merge([(s, e) for s, e, _ in nccl])
    print(f"  nccl kernels: {sum(e-s for s,e in m)/1e6:.2f}s busy over {len(nccl)} events")

# Binned timeline (40 bins): % of bin with any GPU busy.
NB = 40
bins = [0.0] * NB
for s, e in all_iv:
    for b in range(NB):
        bs = t0 + window * b / NB
        be = t0 + window * (b + 1) / NB
        ov = min(e, be) - max(s, bs)
        if ov > 0:
            bins[b] += ov
print("timeline (bin = %.0f ms, %% = GPU-busy share):" % (window / NB / 1e3))
for b in range(NB):
    pct = bins[b] / (window / NB) * 100
    print(f"  {b:2d} |{'#' * int(pct / 2):50s}| {pct:5.1f}%")
