#!/usr/bin/env python3
"""W3 capture localization: where did the lookahead side-stream kernels land?

The C′-ON baseline carries compute on stream 7 alone; the W3 capture splits
compute into stream 7 (first pass) + stream 103 (kicked recompute). Measure:
 1. stream-103 kernel time overlapping the checkpoint-backward windows
 2. stream-103 kernel time overlapping the >5ms nonzero drain windows
 3. stream-103 kernel time running CONCURRENT with any other GPU kernel
    (true overlap) vs serialized in gaps
 4. stream-103 kernel time inside vs outside the backward PHASE (first
    LookaheadCheckpointFunctionBackward start .. last end)
"""
import sys
import numpy as np
from perfetto.trace_processor import TraceProcessor

path = sys.argv[1]
SIDE = sys.argv[2] if len(sys.argv) > 2 else "103"
tp = TraceProcessor(trace=path)
def q(sql):
    return tp.query(sql).as_pandas_dataframe()

# side-stream kernels
side = q(f"""
    SELECT s.ts, s.dur FROM slice s
    JOIN args st ON st.arg_set_id = s.arg_set_id AND st.key='args.stream'
    WHERE s.category='kernel' AND st.display_value='{SIDE}' AND s.dur>0
""")
ss = side.ts.to_numpy(dtype=np.int64)
se = ss + side.dur.to_numpy(dtype=np.int64)
side_tot = float(side.dur.sum() / 1e9)

# all other GPU work (for concurrency)
oth = q(f"""
    SELECT s.ts, s.dur FROM slice s
    LEFT JOIN args st ON st.arg_set_id = s.arg_set_id AND st.key='args.stream'
    WHERE s.category IN ('kernel','gpu_memcpy','gpu_memset') AND s.dur>0
      AND COALESCE(st.display_value,'(none)') != '{SIDE}'
""")
os_ = oth.ts.to_numpy(dtype=np.int64)
oe = os_ + oth.dur.to_numpy(dtype=np.int64)

def merged(a, b):
    if len(a) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    o = np.argsort(a, kind="stable"); a, b = a[o], b[o]
    ms, me = [], []
    cs, ce = int(a[0]), int(b[0])
    for s_, e_ in zip(a[1:], b[1:]):
        if s_ <= ce:
            if e_ > ce: ce = int(e_)
        else:
            ms.append(cs); me.append(ce); cs, ce = int(s_), int(e_)
    ms.append(cs); me.append(ce)
    return np.array(ms), np.array(me)

import bisect
def cover(win_s, win_e, iv_s, iv_e):
    tot = 0
    for s_, e_ in zip(win_s, win_e):
        i = max(0, bisect.bisect_right(iv_s, s_) - 1)
        while i < len(iv_s) and iv_s[i] < e_:
            lo = max(s_, iv_s[i]); hi = min(e_, iv_e[i])
            if hi > lo: tot += hi - lo
            i += 1
    return tot / 1e9

# bwd windows (lookahead function) + phase extent
bwd = q("SELECT ts, dur FROM slice WHERE name='LookaheadCheckpointFunctionBackward' AND dur>0")
bs = bwd.ts.to_numpy(dtype=np.int64); be = bs + bwd.dur.to_numpy(dtype=np.int64)
bwd_ms, bwd_me = merged(bs.copy(), be.copy())
phase_lo, phase_hi = int(bs.min()), int(be.max())

# drains
dr = q("SELECT ts, dur FROM slice WHERE name='aten::nonzero' AND dur>5e6")
ds = dr.ts.to_numpy(dtype=np.int64); de = ds + dr.dur.to_numpy(dtype=np.int64)

oth_ms, oth_me = merged(os_.copy(), oe.copy())

in_bwd = cover(ss, se, bwd_ms, bwd_me)
in_drain = cover(ss, se, ds, de)
concurrent = cover(ss, se, oth_ms, oth_me)
in_phase = float(side.dur[(ss >= phase_lo) & (se <= phase_hi)].sum() / 1e9)

print(f"trace: {path}  side stream: {SIDE}")
print(f"side-stream kernel total:        {side_tot:8.2f}s  ({len(side)} kernels)")
print(f"  within bwd windows (union):    {in_bwd:8.2f}s  ({100*in_bwd/side_tot:.1f}%)")
print(f"  within drain windows:          {in_drain:8.2f}s  ({100*in_drain/side_tot:.1f}%)")
print(f"  concurrent w/ other GPU work:  {concurrent:8.2f}s  ({100*concurrent/side_tot:.1f}%)")
print(f"  within bwd phase extent:       {in_phase:8.2f}s  ({100*in_phase/side_tot:.1f}%)")
print(f"  serialized (no concurrency):   {side_tot-concurrent:8.2f}s  ({100*(side_tot-concurrent)/side_tot:.1f}%)")
