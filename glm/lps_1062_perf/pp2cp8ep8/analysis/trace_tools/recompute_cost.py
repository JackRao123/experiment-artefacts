#!/usr/bin/env python3
"""Measure the wall-clock cost of full activation recomputation in fe127_d16_rank0.

Method: rebuild F blocks / B phases from DSA attention kernels. Inside each B
phase, walk layers in processing order (reverse layer order); each layer's attn
bwd = a group of 4 consecutive dsa_bwd kernels. The window between consecutive
bwd-groups contains that layer's [recompute fwd (attn fwd + MoE fwd + 3 a2a)]
followed by [MoE bwd + 3 a2a] then its attn-bwd group. Split the window's 6 a2a
ops 3/3; the recompute/bwd wall boundary = midpoint between a2a#3 end and
a2a#4 start. Dense layers (no a2a): split at midpoint between last fwd-class
kernel end and first bwd kernel start."""
import sys
from bisect import bisect_left, bisect_right

d = sys.argv[1] if len(sys.argv) > 1 else "fe127_r0"

FWD_SIG = "sparse_attn_fwd_kernel"
BWD_SIG = "dsa_bwd_sm100"

kern = []   # all kernels: (ts, end, tid, name)
attn = []   # (ts, end, kind) kind in F/B on stream 7
idx  = []   # indexer kernels (ts, end)
with open(f"{d}/kernels.tsv") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        ts, dur, tid, nm = float(p[0]), float(p[1]), int(p[3]), p[5]
        kern.append((ts, ts + dur, tid))
        if tid == 7:
            if FWD_SIG in nm:
                attn.append((ts, ts + dur, "F"))
            elif BWD_SIG in nm:
                attn.append((ts, ts + dur, "B"))
            elif "indexer" in nm:
                idx.append((ts, ts + dur))
attn.sort(); idx.sort(); kern.sort()
t0 = kern[0][0]

a2a = []
with open(f"{d}/ann_gpu.tsv") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if p[4] == "nccl:all_to_all":
            a2a.append((float(p[0]), float(p[0]) + float(p[1])))
a2a.sort()
a2a_starts = [s for s, e in a2a]

# ---- group into F blocks (38 F) and B phases (152 B) ----
phases = []  # (kind, list of attn events)
cur, nF, nB = [], 0, 0
for ev in attn:
    cur.append(ev); nF += ev[2] == "F"; nB += ev[2] == "B"
    if nB == 0 and nF == 38:
        phases.append(("F", cur)); cur, nF, nB = [], 0, 0
    elif nB == 152:
        phases.append(("B", cur)); cur, nF, nB = [], 0, 0
assert not cur, f"leftover {nF}/{nB}"
print(f"phases: {''.join(k for k,_ in phases)}")

def a2a_in(a, b):
    i = bisect_left(a2a_starts, a)
    j = bisect_left(a2a_starts, b)
    return a2a[i:j]

# stream-7 busy merged (for kernel-busy attribution)
s7 = []
for ts, e, tid in kern:
    if tid == 7:
        if s7 and ts <= s7[-1][1]:
            s7[-1][1] = max(s7[-1][1], e)
        else:
            s7.append([ts, e])
def busy7(a, b):
    i = bisect_left(s7, [a, a]) - 1
    i = max(i, 0)
    tot = 0.0
    while i < len(s7) and s7[i][0] < b:
        tot += max(0.0, min(s7[i][1], b) - max(s7[i][0], a))
        i += 1
    return tot

# ---- F blocks: fwd wall & a2a ----
f_wall = f_a2a = f_a2a_n = f_busy = 0.0
for k, evs in phases:
    if k != "F":
        continue
    s, e = evs[0][0], max(x[1] for x in evs)
    aa = a2a_in(s, e + 200e3)  # include trailing MoE of last layer (next phase starts later)
    # bound trailing window by next phase start: handled below via cap
    f_wall += e - s
    f_a2a += sum(y - x for x, y in aa)
    f_a2a_n += len(aa)
    f_busy += busy7(s, e)
print(f"F blocks: wall(sum, attn-anchored) {f_wall/1e6:.2f}s, a2a n={f_a2a_n:.0f} t={f_a2a/1e6:.2f}s, s7busy {f_busy/1e6:.2f}s")

# ---- B phases: per-layer split ----
tot_rec_wall = tot_bwd_wall = 0.0
tot_rec_a2a = tot_bwd_a2a = 0.0
n_rec_a2a = n_bwd_a2a = 0
tot_rec_busy = tot_bwd_busy = 0.0
layer_counts = {}
bad = 0
for k, evs in phases:
    if k != "B":
        continue
    # bwd groups: runs of 4 consecutive B events
    groups = []
    run = []
    for ev in evs:
        if ev[2] == "B":
            run.append(ev)
            if len(run) == 4:
                groups.append((run[0][0], run[-1][1]))
                run = []
        else:
            assert not run or len(run) == 0 or True
    assert len(groups) == 38, len(groups)
    fwd_evs = [ev for ev in evs if ev[2] == "F"]
    phase_start = evs[0][0]
    prev_end = phase_start
    for gs, ge in groups:
        win_a2a = a2a_in(prev_end, gs)
        n = len(win_a2a)
        layer_counts[n] = layer_counts.get(n, 0) + 1
        if n == 6:
            boundary = (win_a2a[2][1] + win_a2a[3][0]) / 2
            tot_rec_a2a += sum(e - s for s, e in win_a2a[:3])
            tot_bwd_a2a += sum(e - s for s, e in win_a2a[3:])
            n_rec_a2a += 3; n_bwd_a2a += 3
        elif n == 0:
            # dense layer: boundary = midpoint(last fwd kernel end, bwd group start)
            fe = [ev for ev in fwd_evs if prev_end <= ev[0] < gs]
            boundary = (max(x[1] for x in fe) + gs) / 2 if fe else (prev_end + gs) / 2
        else:
            bad += 1
            boundary = (prev_end + gs) / 2
        tot_rec_wall += boundary - prev_end
        tot_bwd_wall += (ge - boundary)
        tot_rec_busy += busy7(prev_end, boundary)
        tot_bwd_busy += busy7(boundary, ge)
        prev_end = ge

print(f"layer a2a-count distribution per B window: {layer_counts} (bad={bad})")
print(f"\nB phases, RECOMPUTE portion: wall {tot_rec_wall/1e6:.2f}s | s7 kernel-busy {tot_rec_busy/1e6:.2f}s | a2a n={n_rec_a2a} t={tot_rec_a2a/1e6:.2f}s")
print(f"B phases, TRUE-BWD portion:  wall {tot_bwd_wall/1e6:.2f}s | s7 kernel-busy {tot_bwd_busy/1e6:.2f}s | a2a n={n_bwd_a2a} t={tot_bwd_a2a/1e6:.2f}s")

# attn + indexer kernel time split
attn_f_inB = sum(e - s for s, e, kk in attn if kk == "F" and any(p[0]=="B" and p[1][0][0] <= s < max(x[1] for x in p[1]) for p in phases)) if False else None
# cheaper: recompute attn-fwd kernels = F events inside B phases
rec_attn = 0.0
fwd_attn = 0.0
for k, evs in phases:
    tF = sum(e - s for s, e, kk in evs if kk == "F")
    if k == "B":
        rec_attn += tF
    else:
        fwd_attn += tF
print(f"\nattention fwd kernel time: true-fwd {fwd_attn/1e6:.2f}s, recompute {rec_attn/1e6:.2f}s")

# indexer split by phase windows
pw = [(k, evs[0][0], max(x[1] for x in evs)) for k, evs in phases]
idx_f = idx_b = 0.0
for s, e in idx:
    for k, ps, pe in pw:
        if ps <= s < pe:
            if k == "F": idx_f += e - s
            else: idx_b += e - s
            break
print(f"indexer kernel time: in F blocks {idx_f/1e6:.2f}s, in B phases {idx_b/1e6:.2f}s")
