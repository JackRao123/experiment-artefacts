import sys
import numpy as np
from perfetto.trace_processor import TraceProcessor

tp = TraceProcessor(trace=sys.argv[1])
def q(s):
    return tp.query(s).as_pandas_dataframe()

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
t0 = us[0]

# p2p SendRecv kernels (pipeline handoffs)
p2p = q("""SELECT ts, dur FROM slice WHERE category='kernel' AND name LIKE '%SendRecv%'""")
ps = p2p.ts.to_numpy(dtype=np.int64)
pe = (p2p.ts + p2p.dur).to_numpy(dtype=np.int64)
ps.sort(); pe.sort()

# a2a kernels
a2a = q("""SELECT ts, dur FROM slice WHERE category='kernel' AND name LIKE '%AllToAll%'""")
as_ = a2a.ts.to_numpy(dtype=np.int64); ae = (a2a.ts + a2a.dur).to_numpy(dtype=np.int64)
as_.sort(); ae.sort()

sel = gap_d > 10_000_000
gs, gd = gap_s[sel], gap_d[sel]
print(f"{sel.sum()} long gaps; phase analysis (times rel to first activity):")
# distance to nearest p2p end before gap start, and nearest a2a end
for thresh_ms in [5, 50]:
    n_p2p = 0; n_a2a = 0
    for g in gs:
        i = np.searchsorted(pe, g, side="right")
        if i > 0 and (g - pe[i - 1]) < thresh_ms * 1e6:
            n_p2p += 1
        j = np.searchsorted(ae, g, side="right")
        if j > 0 and (g - ae[j - 1]) < thresh_ms * 1e6:
            n_a2a += 1
    print(f"  gaps starting within {thresh_ms}ms after a p2p kernel end: {n_p2p}/{sel.sum()}; after a2a end: {n_a2a}/{sel.sum()}")

# timeline: print all long gaps with nearest p2p-end delta
print("\nall >10ms gaps: (dur ms, t s, ms since nearest p2p end, ms since nearest a2a end)")
for g, d in sorted(zip(gs, gd), key=lambda x: x[0]):
    i = np.searchsorted(pe, g, side="right")
    dp = (g - pe[i - 1]) / 1e6 if i > 0 else -1
    j = np.searchsorted(ae, g, side="right")
    da = (g - ae[j - 1]) / 1e6 if j > 0 else -1
    print(f"  {d/1e6:8.2f}ms  t={(g-t0)/1e9:8.3f}  dp2p={dp:9.2f}  da2a={da:9.2f}")
