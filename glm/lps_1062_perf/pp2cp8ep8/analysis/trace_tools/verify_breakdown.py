#!/usr/bin/env python3
"""Verify the 0%-overlap finding and decompose stream-7 idle by covering comm class."""
import sys
from bisect import bisect_left

d = sys.argv[1] if len(sys.argv) > 1 else "fe127_r0"

rows = []
with open(f"{d}/kernels.tsv") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        rows.append((float(p[0]), float(p[1]), int(p[3]), p[5]))
rows.sort()
t0 = rows[0][0]
tend = max(r[0] + r[1] for r in rows)

def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out

def total(iv):
    return sum(e - s for s, e in iv)

def inter(a, b):
    """intersection duration of two merged interval lists"""
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0]); e = min(a[i][1], b[j][1])
        if e > s: tot += e - s
        if a[i][1] < b[j][1]: i += 1
        else: j += 1
    return tot

s7 = merge([(ts, ts + dur) for ts, dur, tid, nm in rows if tid == 7])
allb = merge([(ts, ts + dur) for ts, dur, tid, nm in rows])

# comm classes from gpu annotations
cls = {"nccl:all_to_all": [], "nccl:coalesced": [], "nccl:_all_gather_base": [], "nccl:_reduce_scatter_base": []}
with open(f"{d}/ann_gpu.tsv") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        nm = p[4]
        if nm in cls:
            cls[nm].append((float(p[0]), float(p[0]) + float(p[1])))

wall = tend - t0
print(f"window {wall/1e6:.2f}s  s7 busy {total(s7)/1e6:.2f}s  allGPU busy {total(allb)/1e6:.2f}s  pure idle {(wall-total(allb))/1e6:.2f}s")

for nm, iv in cls.items():
    m = merge(iv)
    ov = inter(m, s7)
    print(f"{nm:28s} resident {total(m)/1e6:7.2f}s  overlap-with-s7 {ov/1e6:6.2f}s ({ov/max(total(m),1e-9)*100:5.1f}%)")

# spot check: take 3 a2a windows, count s7 kernels inside directly
a2a = sorted(cls["nccl:all_to_all"])
import random
print("\nspot-check a2a windows (direct scan):")
for k in [100, 2500, 4900]:
    s, e = a2a[k]
    n_in = sum(1 for ts, dur, tid, nm in rows if tid == 7 and ts < e and ts + dur > s)
    print(f"  a2a[{k}] dur={ (e-s)/1e3:.2f}ms  s7 kernels overlapping: {n_in}")

# s7 idle coverage decomposition
idle = []
prev = t0
for s, e in s7:
    if s > prev: idle.append([prev, s])
    prev = max(prev, e)
if tend > prev: idle.append([prev, tend])
tot_idle = total(idle)
cov_a2a = inter(idle, merge(cls["nccl:all_to_all"]))
cov_p2p = inter(idle, merge(cls["nccl:coalesced"]))
cov_ag  = inter(idle, merge(cls["nccl:_all_gather_base"]))
cov_rs  = inter(idle, merge(cls["nccl:_reduce_scatter_base"]))
# other streams (159-162, 31, 39, 27) busy
oth = merge([(ts, ts + dur) for ts, dur, tid, nm in rows if tid not in (7,)])
cov_any = inter(idle, oth)
print(f"\ns7 idle total {tot_idle/1e6:.2f}s")
print(f"  covered by EP a2a        {cov_a2a/1e6:6.2f}s")
print(f"  covered by PP p2p        {cov_p2p/1e6:6.2f}s")
print(f"  covered by CP allgather  {cov_ag/1e6:6.2f}s")
print(f"  covered by CP reducescat {cov_rs/1e6:6.2f}s")
print(f"  covered by ANY other stream {cov_any/1e6:6.2f}s")
print(f"  covered by nothing (pure)   {(tot_idle-cov_any)/1e6:6.2f}s")

# kernel class totals on s7
def klass(nm):
    if "sparse_attn_fwd" in nm: return "attn_fwd(DSA)"
    if "dsa_bwd" in nm: return "attn_bwd(DSA)"
    if "indexer" in nm: return "dsa_indexer"
    if nm.startswith("nvjet"): return "gemm(nvjet)"
    if "cutlass" in nm and "gemm" in nm.lower(): return "gemm(cutlass)"
    if "grouped" in nm.lower(): return "grouped_gemm"
    if "rmsnorm" in nm: return "rmsnorm"
    if "elementwise" in nm or "CUDAFunctor" in nm: return "elementwise"
    if "CatArray" in nm: return "cat"
    if "gather" in nm or "scatter" in nm or "index" in nm or "sort" in nm or "topk" in nm.lower(): return "gather/sort/index"
    if "reduce" in nm.lower(): return "reduce"
    if "memcpy" in nm.lower() or "Memcpy" in nm: return "memcpy"
    return "other"

from collections import defaultdict
kc = defaultdict(lambda: [0, 0.0])
for ts, dur, tid, nm in rows:
    if tid == 7:
        c = klass(nm)
        kc[c][0] += 1; kc[c][1] += dur
print("\nstream7 kernel classes:")
for c, (n, du) in sorted(kc.items(), key=lambda x: -x[1][1]):
    print(f"  {c:20s} n={n:7d}  {du/1e6:7.2f}s")
