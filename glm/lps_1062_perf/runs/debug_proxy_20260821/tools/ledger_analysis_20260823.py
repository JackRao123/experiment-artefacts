#!/usr/bin/env python3
"""Gap-ledger analysis for the within-2% mission.

For one arm (gpu.csv + cpuops.csv dumped by trace_processor), itemize the
traced window: phase walls, main-stream idle gaps >1.5 ms each attributed
to the co-located CPU state, host-blocking CUDA calls, offload-machinery
CPU time, and copy-stream metrics. Run on both arms and diff per class.

Usage: ledger_analysis.py <gpu.csv> <cpuops.csv> <label>
"""

import csv
import sys
from collections import defaultdict


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["ts"] = int(r["ts"])
            r["dur"] = int(r["dur"])
            if "depth" in r:
                r["depth"] = int(r["depth"])
            rows.append(r)
    return rows


def union(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def tl(iv):
    return sum(e - s for s, e in iv)


def isect(a, b):
    i = j = 0
    tot = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if s < e:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def clip(iv, a, b):
    return [[max(s, a), min(e, b)] for s, e in iv if s < b and e > a]


def main(gpu_path, cpu_path, label):
    gpu = load(gpu_path)
    cpu = load(cpu_path)
    ps = [r for r in cpu if r["name"].startswith("ProfilerStep")]
    t0 = min(r["ts"] for r in ps)

    def rel(ts):
        return (ts - t0) / 1e6

    ce = sorted(r["ts"] for r in cpu if r["name"] == "CrossEntropyFunction")
    eng = [r for r in cpu if r["name"].startswith("autograd::engine")]
    bwd0 = min(r["ts"] for r in eng)
    bwd1 = max(r["ts"] + r["dur"] for r in eng)
    end = max(r["ts"] + r["dur"] for r in gpu)

    main_rows = [r for r in gpu if r["thread_name"].strip() == "stream 7"]
    miv = union([[r["ts"], r["ts"] + r["dur"]] for r in main_rows])

    phases = [("fwd", t0, ce[0]), ("mid", ce[0], bwd0), ("bwd", bwd0, bwd1),
              ("tail", bwd1, end)]
    print(f"\n######## {label} ########")
    print(f"CE1 {rel(ce[0]):.1f}  CE2 {rel(ce[1]) if len(ce) > 1 else -1:.1f}  "
          f"bwd [{rel(bwd0):.1f},{rel(bwd1):.1f}]  end {rel(end):.1f}")
    for nm, a, b in phases:
        busy = tl(clip(miv, a, b)) / 1e6
        print(f"  {nm:4s} wall={(b - a) / 1e6:7.1f}  busy={busy:7.1f}  "
              f"idle={(b - a) / 1e6 - busy:7.1f}")

    # copy streams
    for nm, st in (("D2H(s44)", "stream 44"), ("H2D(s48)", "stream 48")):
        c = [r for r in gpu if r["thread_name"].strip() == st
             and r["category"] == "gpu_memcpy"]
        if not c:
            continue
        iv = union([[r["ts"], r["ts"] + r["dur"]] for r in c])
        busy = tl(iv) / 1e6
        b = sum(int(r["bytes"]) for r in c if r["bytes"] not in ("", "[NULL]"))
        ov = isect(iv, miv) / 1e6
        print(f"  {nm}: n={len(c)} busy={busy:.1f}ms {b / 2**30:.2f}GiB "
              f"window=[{rel(min(r['ts'] for r in c)):.1f},"
              f"{rel(max(r['ts'] + r['dur'] for r in c)):.1f}] "
              f"main-overlap={100 * ov / busy:.0f}%")

    # attribute every main-stream gap >1.5ms to co-located CPU state
    print("  --- main-stream gaps >1.5ms, attributed ---")
    ledger = defaultdict(float)
    gaps = []
    prev = miv[0]
    for cur in miv[1:]:
        g = cur[0] - prev[1]
        if g > 1.5e6:
            gaps.append((prev[1], cur[0]))
        prev = cur
    cpu_by_thread = defaultdict(list)
    for r in cpu:
        cpu_by_thread[r["thread_name"]].append(r)
    for g0, g1 in gaps:
        glen = (g1 - g0) / 1e6
        phase = next((nm for nm, a, b in phases if a <= g0 < b), "?")
        # find the most-covering blocking slice on any thread
        best = None
        for th, rows in cpu_by_thread.items():
            for r in rows:
                if r["ts"] < g1 and r["ts"] + r["dur"] > g0 and r["dur"] > 1e6:
                    cover = (min(r["ts"] + r["dur"], g1) - max(r["ts"], g0))
                    if best is None or cover > best[0]:
                        best = (cover, r, th)
        if best and best[0] > 0.5 * (g1 - g0):
            r = best[1]
            # innermost long slice covering the gap on that thread
            inner = [x for x in cpu_by_thread[best[2]]
                     if x["ts"] <= max(r["ts"], g0)
                     and x["ts"] + x["dur"] >= min(r["ts"] + r["dur"], g1)]
            deepest = max(inner, key=lambda x: x.get("depth", 0), default=r)
            key = f"{phase}:{deepest['name'][:44]}"
        else:
            key = f"{phase}:<no covering CPU slice>"
        ledger[key] += glen
        print(f"    {rel(g0):8.1f}ms +{glen:6.1f}ms  {key}")
    print("  --- gap ledger by class ---")
    for k, v in sorted(ledger.items(), key=lambda kv: -kv[1]):
        print(f"    {v:7.1f} ms  {k}")

    # host-blocking cuda calls >1ms
    print("  --- host cuda_runtime calls >1ms ---")
    agg = defaultdict(lambda: [0, 0.0])
    for r in cpu:
        if r["category"] == "cuda_runtime" and r["dur"] > 1e6:
            phase = next((nm for nm, a, b in phases if a <= r["ts"] < b), "?")
            agg[(phase, r["name"])][0] += 1
            agg[(phase, r["name"])][1] += r["dur"] / 1e6
    for (phase, nm), (n, d) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"    {d:7.1f} ms n={n:3d} {phase}:{nm}")

    # offload-machinery CPU time on any thread, by phase
    print("  --- offload-machinery cpu_op time ---")
    off = defaultdict(float)
    for r in cpu:
        if r["category"] == "cpu_op" and ("FineGrained" in r["name"]):
            phase = next((nm for nm, a, b in phases if a <= r["ts"] < b), "?")
            off[f"{phase}:{r['name'][:40]}"] += r["dur"] / 1e6
    for k, v in sorted(off.items(), key=lambda kv: -kv[1]):
        if v > 0.5:
            print(f"    {v:7.1f} ms  {k}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
