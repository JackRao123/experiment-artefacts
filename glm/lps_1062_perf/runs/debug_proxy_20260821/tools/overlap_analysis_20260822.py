#!/usr/bin/env python3
"""Overlap analysis for the 4-layer offload traces (LPS-1062).

Reads the GPU-slice CSVs dumped by trace_processor and quantifies:
  - forward/backward windows (from memcpy phase boundaries + kernel activity)
  - per-direction copy busy time, bytes, effective bandwidth
  - copy time overlapped with main-stream compute vs uncovered
  - main-stream idle gaps and where they sit relative to copy bursts
  - CPU-side sync call inventory
"""

import csv
import sys
from collections import defaultdict


def read_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["ts"] = int(r["ts"])
            r["dur"] = int(r["dur"])
            rows.append(r)
    return rows


def union_intervals(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def total_len(iv):
    return sum(e - s for s, e in iv)


def intersect_len(a, b):
    # a, b: sorted merged interval lists
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


def analyze(gpu_csv, cpu_csv, label):
    rows = read_csv(gpu_csv)
    print(f"\n{'='*70}\n{label}\n{'='*70}")

    t0 = min(r["ts"] for r in rows)

    def rel(ts):
        return (ts - t0) / 1e6  # ms

    # Streams
    main = [r for r in rows if r["thread_name"].strip() == "stream 7"]
    d2h = [r for r in rows if r["thread_name"].strip() == "stream 44"
           and r["category"] == "gpu_memcpy"]
    h2d = [r for r in rows if r["thread_name"].strip() == "stream 48"
           and r["category"] == "gpu_memcpy"]
    other_streams = defaultdict(list)
    for r in rows:
        tn = r["thread_name"].strip()
        if tn not in ("stream 7", "stream 44", "stream 48"):
            other_streams[tn].append(r)

    main_kern = [r for r in main if r["category"] == "kernel"]
    main_iv = union_intervals([[r["ts"], r["ts"] + r["dur"]] for r in main])
    # all-compute = main + other compute streams (146-149 etc), excluding
    # the two offload copy streams
    allcomp_rows = main + [r for v in other_streams.values() for r in v]
    allcomp_iv = union_intervals([[r["ts"], r["ts"] + r["dur"]] for r in allcomp_rows])

    span = (max(r["ts"] + r["dur"] for r in rows) - t0) / 1e6
    print(f"window span            : {span:.1f} ms")
    print(f"main stream busy       : {total_len(main_iv)/1e6:.1f} ms "
          f"({len(main)} slices)")
    print(f"all compute busy union : {total_len(allcomp_iv)/1e6:.1f} ms")
    for tn, v in sorted(other_streams.items()):
        print(f"  {tn:12s}: {sum(r['dur'] for r in v)/1e6:8.1f} ms {len(v)} slices")

    for name, copies in (("D2H(s44)", d2h), ("H2D(s48)", h2d)):
        if not copies:
            print(f"{name}: none")
            continue
        iv = union_intervals([[r["ts"], r["ts"] + r["dur"]] for r in copies])
        busy = total_len(iv) / 1e6
        b = sum(int(r["bytes"]) for r in copies if r["bytes"] not in ("", "[NULL]"))
        first = rel(min(r["ts"] for r in copies))
        last = rel(max(r["ts"] + r["dur"] for r in copies))
        ov_main = intersect_len(iv, main_iv) / 1e6
        ov_all = intersect_len(iv, allcomp_iv) / 1e6
        print(f"{name}: n={len(copies)} busy={busy:.1f}ms bytes={b/2**30:.3f}GiB "
              f"bw={b/ (busy/1e3) / 2**30:.1f}GiB/s window=[{first:.1f},{last:.1f}]ms")
        print(f"    overlapped w/ main compute: {ov_main:.1f}ms "
              f"({100*ov_main/busy:.0f}%)  w/ any compute: {ov_all:.1f}ms "
              f"({100*ov_all/busy:.0f}%)  uncovered: {busy-ov_all:.1f}ms")

    # Main-stream gaps > 1ms
    gaps = []
    for a, bb in zip(main_iv, main_iv[1:]):
        g = bb[0] - a[1]
        if g > 1e6:  # >1ms
            gaps.append((rel(a[1]), g / 1e6))
    print(f"main-stream gaps >1ms  : {len(gaps)}, total {sum(g for _, g in gaps):.1f} ms")
    for st, g in gaps[:15]:
        print(f"    gap at {st:8.1f} ms  len {g:7.1f} ms")

    # CPU sync inventory
    crows = read_csv(cpu_csv)
    agg = defaultdict(lambda: [0, 0])
    for r in crows:
        agg[(r["name"], r["thread_name"])][0] += 1
        agg[(r["name"], r["thread_name"])][1] += r["dur"]
    print("top CPU cuda_runtime calls by total time:")
    for (nm, th), (n, d) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:12]:
        print(f"    {d/1e6:9.1f} ms  n={n:5d}  {nm}  [{th}]")
    return dict(t0=t0, main_iv=main_iv, d2h=d2h, h2d=h2d)


if __name__ == "__main__":
    analyze("/tmp/offload_gpu.csv", "/tmp/offload_cpu.csv", "OFFLOAD (4 layers, 4 groups)")
    analyze("/tmp/baseline_gpu.csv", "/tmp/baseline_cpu.csv", "BASELINE (4 layers, no offload)")
