#!/usr/bin/env python3
"""Decompose recompute-window wall time by kernel class / module.
Windows: same per-layer recompute spans as recompute_cost.py (prev bwd-group
end -> a2a 3/4 midpoint for MoE layers, fwd-kernel/bwd midpoint for dense)."""
import sys
from bisect import bisect_left, bisect_right

d = sys.argv[1] if len(sys.argv) > 1 else "fe127_r0"
FWD_SIG = "sparse_attn_fwd_kernel"; BWD_SIG = "dsa_bwd_sm100"

kern = []
with open(f"{d}/kernels.tsv") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        kern.append((float(p[0]), float(p[1]), int(p[3]), p[5]))
kern.sort()

attn = [(ts, ts+dur, "F" if FWD_SIG in nm else "B") for ts, dur, tid, nm in kern
        if tid == 7 and (FWD_SIG in nm or BWD_SIG in nm)]
attn.sort()

a2a = []
with open(f"{d}/ann_gpu.tsv") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if p[4] == "nccl:all_to_all":
            a2a.append((float(p[0]), float(p[0])+float(p[1])))
a2a.sort()
a2a_starts = [s for s, e in a2a]

phases, cur, nF, nB = [], [], 0, 0
for ev in attn:
    cur.append(ev); nF += ev[2] == "F"; nB += ev[2] == "B"
    if nB == 0 and nF == 38: phases.append(("F", cur)); cur, nF, nB = [], 0, 0
    elif nB == 152: phases.append(("B", cur)); cur, nF, nB = [], 0, 0

windows = []  # (start, end, is_moe)
for k, evs in phases:
    if k != "B": continue
    groups, run = [], []
    for ev in evs:
        if ev[2] == "B":
            run.append(ev)
            if len(run) == 4: groups.append((run[0][0], run[-1][1])); run = []
    fwd_evs = [ev for ev in evs if ev[2] == "F"]
    prev_end = evs[0][0]
    for gs, ge in groups:
        i, j = bisect_left(a2a_starts, prev_end), bisect_left(a2a_starts, gs)
        win = a2a[i:j]
        if len(win) == 6:
            boundary = (win[2][1] + win[3][0]) / 2
            windows.append((prev_end, boundary, True))
        else:
            fe = [ev for ev in fwd_evs if prev_end <= ev[0] < gs]
            boundary = (max(x[1] for x in fe) + gs) / 2 if fe else (prev_end + gs) / 2
            windows.append((prev_end, boundary, False))
        prev_end = ge
windows.sort()
wstarts = [w[0] for w in windows]

def klass(tid, nm):
    if FWD_SIG in nm: return "attn_core_fwd"
    if "indexer" in nm: return "attn_indexer"
    if "AllGather" in nm: return "cp_allgather"
    if "ReduceScatter" in nm: return "cp_reducescatter"
    if "SendRecv" in nm: return "ep_a2a" if tid == 51 else "pp_p2p"
    if nm.startswith("nvjet"):
        return "gemm_expert(sidestream)" if tid in (159,160,161,162) else "gemm_main(s7)"
    if "rmsnorm" in nm: return "rmsnorm"
    if "CatArray" in nm: return "cat"
    if "sort" in nm or "gather" in nm.lower() or "index" in nm or "topk" in nm.lower() or "scatter" in nm: return "dispatch_gather_sort"
    if "elementwise" in nm or "CUDAFunctor" in nm: return "elementwise"
    if "Memcpy" in nm or "memcpy" in nm.lower(): return "memcpy"
    return "other"

from collections import defaultdict
tot = defaultdict(float); tot_moe = defaultdict(float); tot_dense = defaultdict(float)
cnt = defaultdict(int)
busy_iv = []  # for idle union: collect clipped intervals
for ts, dur, tid, nm in kern:
    i = bisect_right(wstarts, ts) - 1
    if i < 0: continue
    ws, we, ismoe = windows[i]
    if ts >= we: continue
    e = min(ts + dur, we)
    c = klass(tid, nm)
    tot[c] += e - ts; cnt[c] += 1
    (tot_moe if ismoe else tot_dense)[c] += e - ts
    busy_iv.append((ts, e))

busy_iv.sort()
merged = []
for s, e in busy_iv:
    if merged and s <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], e)
    else: merged.append([s, e])
wall = sum(we - ws for ws, we, _ in windows)
busy = sum(e - s for s, e in merged)
n_moe = sum(1 for w in windows if w[2]); n_dense = len(windows) - n_moe
print(f"recompute windows: {len(windows)} ({n_moe} MoE, {n_dense} dense), wall {wall/1e6:.2f}s, any-stream busy {busy/1e6:.2f}s, idle {(wall-busy)/1e6:.2f}s")
print(f"\n{'class':26s} {'total s':>8s} {'ms/MoE-layer-mb':>16s} {'ms/dense-layer-mb':>18s} {'n':>8s}")
for c in sorted(tot, key=lambda x: -tot[x]):
    print(f"{c:26s} {tot[c]/1e6:8.2f} {tot_moe[c]/1e3/max(n_moe,1):16.1f} {tot_dense[c]/1e3/max(n_dense,1):18.1f} {cnt[c]:8d}")
print(f"\nidle (no kernel any stream): {(wall-busy)/1e6:.2f}s = {(wall-busy)/1e3/len(windows):.1f} ms/layer-mb avg")
print(f"wall per layer-mb: MoE { sum(we-ws for ws,we,m in windows if m)/1e3/n_moe:.1f} ms, dense {sum(we-ws for ws,we,m in windows if not m)/1e3/max(n_dense,1):.1f} ms")
