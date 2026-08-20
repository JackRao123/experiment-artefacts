#!/usr/bin/env python3
"""Reconstruct the PP 1F1B schedule on stage 0 from the fe127_d16_rank0 trace
extracts, and measure per-phase exposure."""
import sys
from bisect import bisect_left

d = sys.argv[1] if len(sys.argv) > 1 else "fe127_r0"

# ---- load kernels ----
rows = []  # (ts, dur, tid, name)
with open(f"{d}/kernels.tsv") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        rows.append((float(p[0]), float(p[1]), int(p[3]), p[5]))
rows.sort(key=lambda r: r[0])

t0 = rows[0][0]
tend = max(r[0] + r[1] for r in rows)
wall = tend - t0
print(f"GPU-active window: {wall/1e6:.2f}s")

FWD_SIG = "sparse_attn_fwd_kernel"
BWD_SIG = "dsa_bwd_sm100"

attn = [(ts, dur, "F" if FWD_SIG in nm else "B")
        for ts, dur, tid, nm in rows
        if tid == 7 and (FWD_SIG in nm or BWD_SIG in nm)]
attn.sort()
print(f"attn kernels: {len(attn)} (F={sum(1 for a in attn if a[2]=='F')}, B={sum(1 for a in attn if a[2]=='B')})")

# ---- group into blocks ----
# A pure-F block = run of consecutive F attn kernels (a true forward pass of 38 layers).
# A B block = region containing B kernels (recompute F interleaved with B per layer).
blocks = []
cur = None
for ts, dur, k in attn:
    if cur is None:
        cur = {"kinds": [k], "start": ts, "end": ts + dur, "nF": int(k == "F"), "nB": int(k == "B")}
        continue
    cur["kinds"].append(k)
    cur["end"] = max(cur["end"], ts + dur)
    cur["nF"] += k == "F"
    cur["nB"] += k == "B"
    # close a block when we have a clean boundary:
    # pure-F block closes at 38 F's followed by next kernel starting a new mb;
    # B block closes at 38 B's.
    if cur["nB"] == 0 and cur["nF"] == 38:
        blocks.append(("F", cur["start"], cur["end"]))
        cur = None
    elif cur["nB"] == 38:
        blocks.append(("B", cur["start"], cur["end"]))
        cur = None
if cur is not None:
    blocks.append(("?" + str(cur["nF"]) + "/" + str(cur["nB"]), cur["start"], cur["end"]))

seq = "".join(b[0][0] for b in blocks)
print(f"block sequence ({len(blocks)}): {seq}")

# ---- p2p + a2a events ----
p2p = []
a2a = []
with open(f"{d}/ann_gpu.tsv") as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        ts, dur, nm = float(p[0]), float(p[1]), p[4]
        if nm == "nccl:coalesced":
            p2p.append((ts, dur))
        elif nm == "nccl:all_to_all":
            a2a.append((ts, dur))
p2p.sort(); a2a.sort()
print(f"p2p ops: {len(p2p)}, total {sum(x[1] for x in p2p)/1e6:.2f}s")

# ---- stream7 busy union / gap list ----
s7 = sorted((ts, ts + dur) for ts, dur, tid, nm in rows if tid == 7)
merged = []
for s, e in s7:
    if merged and s <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], e)
    else:
        merged.append([s, e])
busy7 = sum(e - s for s, e in merged)
print(f"stream7 busy {busy7/1e6:.2f}s, idle {(wall-busy7)/1e6:.2f}s over window")

# all-GPU busy union (all streams)
allints = sorted((ts, ts + dur) for ts, dur, tid, nm in rows)
mall = []
for s, e in allints:
    if mall and s <= mall[-1][1]:
        mall[-1][1] = max(mall[-1][1], e)
    else:
        mall.append([s, e])
busyall = sum(e - s for s, e in mall)
print(f"all-GPU busy {busyall/1e6:.2f}s, pure idle {(wall-busyall)/1e6:.2f}s")

def overlap(a, b, ints):
    """total overlap of [a,b] with merged interval list ints"""
    i = bisect_left(ints, [a, a]) - 1
    if i < 0:
        i = 0
    tot = 0.0
    while i < len(ints) and ints[i][0] < b:
        s, e = ints[i]
        tot += max(0.0, min(e, b) - max(s, a))
        i += 1
    return tot

# ---- timeline print ----
print("\n=== schedule timeline (times in s from window start) ===")
print(f"{'blk':>5} {'start':>8} {'end':>8} {'dur':>7} {'gap_before':>10} {'s7idle_in_gap':>13}")
prev_end = t0
for i, (k, s, e) in enumerate(blocks):
    gap = s - prev_end
    idle = (gap - overlap(prev_end, s, merged)) if gap > 0 else 0.0
    print(f"{k}{i:>4} {(s-t0)/1e6:8.2f} {(e-t0)/1e6:8.2f} {(e-s)/1e6:7.2f} {gap/1e6:10.3f} {idle/1e6:13.3f}")
    prev_end = e

print("\n=== PP p2p (nccl:coalesced) ops ===")
for i, (ts, dur) in enumerate(p2p):
    # which block are we inside / after?
    lbl = ""
    for j, (k, s, e) in enumerate(blocks):
        if ts >= s and ts < e:
            lbl = f"inside {k}{j}"
            break
        if ts < s:
            lbl = f"between {blocks[j-1][0]}{j-1} and {k}{j}" if j > 0 else "before first"
            break
    else:
        lbl = "after last"
    # stream7 busy fraction during this op
    ov = overlap(ts, ts + dur, merged)
    print(f"p2p{i:>3} start={(ts-t0)/1e6:8.2f} dur={dur/1e6:7.3f}s  s7busy={ov/dur*100 if dur>0 else 0:5.1f}%  {lbl}")

# ---- exposure accounting ----
# a2a merged intervals
am = []
for s, e in sorted((ts, ts + dur) for ts, dur in a2a):
    if am and s <= am[-1][1]:
        am[-1][1] = max(am[-1][1], e)
    else:
        am.append([s, e])
a2a_tot = sum(e - s for s, e in am)
a2a_hidden = sum(overlap(s, e, merged) for s, e in am)
pm = []
for s, e in sorted((ts, ts + dur) for ts, dur in p2p):
    if pm and s <= pm[-1][1]:
        pm[-1][1] = max(pm[-1][1], e)
    else:
        pm.append([s, e])
p2p_tot = sum(e - s for s, e in pm)
p2p_hidden = sum(overlap(s, e, merged) for s, e in pm)
print(f"\nEP a2a resident {a2a_tot/1e6:.2f}s, overlapped-with-s7-compute {a2a_hidden/1e6:.2f}s ({a2a_hidden/a2a_tot*100:.0f}%), exposed {(a2a_tot-a2a_hidden)/1e6:.2f}s")
print(f"PP p2p resident {p2p_tot/1e6:.2f}s, overlapped-with-s7-compute {p2p_hidden/1e6:.2f}s ({p2p_hidden/p2p_tot*100:.0f}%), exposed {(p2p_tot-p2p_hidden)/1e6:.2f}s")
