import sys
from collections import Counter
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

# top long gaps (>10ms)
sel = gap_d > 10_000_000
gs, ge = gap_s[sel], gap_e[sel]
print(f"{sel.sum()} gaps >10ms, total {gap_d[sel].sum()/1e9:.2f}s")

# what host API calls happen inside those gaps
rt = q("""SELECT ts, dur, name FROM slice WHERE category='cuda_runtime'""")
rs = rt.ts.to_numpy(dtype=np.int64); re_ = (rt.ts + rt.dur).to_numpy(dtype=np.int64)
rn = rt.name.to_numpy()
cnt = Counter(); tim = Counter()
for a, b in zip(gs, ge):
    lo = np.searchsorted(rs, a, side="left")
    hi = np.searchsorted(rs, b, side="left")
    for n, d in zip(rn[lo:hi], re_[lo:hi] - rs[lo:hi]):
        cnt[n] += 1; tim[n] += int(d)
print("runtime calls inside >10ms gaps (all gaps summed):")
for n, t in tim.most_common(15):
    print(f"  {t/1e9:8.3f}s {cnt[n]:>7}x  {n}")

# and the aten ops active (cpu_op slices overlapping the gaps)
ops = q("""SELECT ts, dur, name FROM slice WHERE category='cpu_op'""")
os_ = ops.ts.to_numpy(dtype=np.int64); oe = (ops.ts + ops.dur).to_numpy(dtype=np.int64)
on = ops.name.to_numpy()
cnt2 = Counter(); tim2 = Counter()
for a, b in zip(gs[:40], ge[:40]):   # top 40 gaps for speed
    # ops that contain the gap start
    cand = (os_ <= a) & (oe >= a)
    for n in on[cand]:
        cnt2[n] += 1
print("\naten ops containing the gap START (first 40 gaps):")
for n, c in cnt2.most_common(12):
    print(f"  {c:>4}x  {n}")
