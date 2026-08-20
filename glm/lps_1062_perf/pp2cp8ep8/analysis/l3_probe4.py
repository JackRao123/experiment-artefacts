import sys
import numpy as np
from perfetto.trace_processor import TraceProcessor

tp = TraceProcessor(trace=sys.argv[1])
def q(s):
    return tp.query(s).as_pandas_dataframe()

# GPU union
gpu = q("SELECT ts, dur FROM slice WHERE category IN ('kernel','gpu_memcpy','gpu_memset') AND dur>0")
ga = gpu.ts.to_numpy(dtype=np.int64); gb = ga + gpu.dur.to_numpy(dtype=np.int64)
o = np.argsort(ga, kind="stable"); ga, gb = ga[o], gb[o]
us, ue = [], []
cs, ce = int(ga[0]), int(gb[0])
for s_, e_ in zip(ga[1:], gb[1:]):
    if s_ <= ce:
        if e_ > ce: ce = int(e_)
    else:
        us.append(cs); ue.append(ce); cs, ce = int(s_), int(e_)
us.append(cs); ue.append(ce)
us = np.array(us); ue = np.array(ue)
gap_s, gap_e = ue[:-1], us[1:]
m = gap_e > gap_s
gap_s, gap_e = gap_s[m], gap_e[m]
gap_d = gap_e - gap_s

# layout-builder aten::index windows (parents of layout nonzeros)
idx_ops = q("""SELECT p.id, p.ts, p.dur FROM slice p
               WHERE p.name='aten::index' AND p.id IN (
                   SELECT s.parent_id FROM slice s WHERE s.name='aten::nonzero')""")
ws = np.array(sorted(idx_ops.ts.to_numpy(dtype=np.int64)))
we = np.array(sorted((idx_ops.ts + idx_ops.dur).to_numpy(dtype=np.int64)))

# strict: gap start inside a layout window
i = np.searchsorted(we, gap_s, side="left")
i = np.minimum(i, len(ws) - 1)
inside = (ws[i] <= gap_s) & (gap_s < we[i])
sel = inside & (gap_d >= 100_000)
print(f"gaps >0.1ms starting INSIDE a layout-index window: {sel.sum()}")
d = gap_d[sel]
for lo, hi in [(1e5, 1e6), (1e6, 1e7), (1e7, 1e9)]:
    mm = (d >= lo) & (d < hi)
    print(f"  {lo/1e6:.1f}-{hi/1e6:.0f} ms: {int(mm.sum())} gaps, {d[mm].sum()/1e9:.3f}s")
print("top 15 such gaps (dur ms, gap start rel to first gpu activity s):")
t0 = us[0]
order = np.argsort(-gap_d[sel])
gs, gd = gap_s[sel], gap_d[sel]
for j in order[:15]:
    print(f"  {gd[j]/1e6:9.2f} ms  at t={(gs[j]-t0)/1e9:8.3f}s")

# context: what kernel runs right before and after the top gaps
allk = q("SELECT ts, dur, name, track_id FROM slice WHERE category='kernel' AND dur>0 ORDER BY ts")
ks = allk.ts.to_numpy(dtype=np.int64)
ke = (allk.ts + allk.dur).to_numpy(dtype=np.int64)
kn = allk.name.to_numpy()
print("\ntop 5 gaps bracketed by kernels:")
for j in order[:5]:
    g0, g1 = gap_s[sel][j], gap_e[sel][j]
    before = np.searchsorted(ke, g0, side="right") - 1
    after = np.searchsorted(ks, g1, side="left")
    print(f"  gap {(g1-g0)/1e6:.2f}ms at t={(g0-t0)/1e9:.3f}s")
    if before >= 0:
        print(f"    before: {kn[before][:80]}")
    if after < len(kn):
        print(f"    after:  {kn[after][:80]}")
