#!/usr/bin/env python3
"""SendRecv / comm-stream mechanism analysis for one kineto trace.

Prints the numbers needed for the de-staggering verdict, count-normalized so
traces with different microbatch counts compare cleanly:
  - SendRecv total, calls, per-call p10/p50/p90/p99/mean
  - SendRecv total split by step phase (fwd-span vs bwd-span via CheckpointFunction*)
  - comm-stream (SendRecv track) idle: total, and inside vs outside host-block windows
  - host-block coverage (nonzero>5ms + memcpyAsync>1ms + streamSync>1ms, both threads)
  - GPU-union idle in step
Usage: python sendrecv_mechanism.py TRACE
"""
import sys
import numpy as np
from perfetto.trace_processor import TraceProcessor

def main(path):
    tp = TraceProcessor(trace=path)
    def q(s): return tp.query(s).as_pandas_dataframe()

    step = q("SELECT ts, dur FROM slice WHERE name='ProfilerStep#0'")
    S = float(step.ts.iloc[0]); E = S + float(step.dur.iloc[0])
    wall = (E - S) / 1e9

    sr = q("SELECT ts, dur, track_id FROM slice WHERE category='kernel' AND name LIKE '%SendRecv%' AND dur>0")
    d = sr.dur.to_numpy() / 1e6  # ms
    print(f"wall {wall:.2f}s")
    print(f"SendRecv: n={len(sr)} total={sr.dur.sum()/1e9:.2f}s "
          f"mean={d.mean():.2f}ms p10={np.percentile(d,10):.2f} p50={np.percentile(d,50):.2f} "
          f"p90={np.percentile(d,90):.2f} p99={np.percentile(d,99):.2f}")

    # phase split: fwd-span vs bwd-span (overlapping spans; classify by containment)
    cf = q("SELECT ts, dur, name FROM slice WHERE name IN ('CheckpointFunction','CheckpointFunctionBackward')")
    fwd = cf[cf.name == "CheckpointFunction"]
    bwd = cf[cf.name == "CheckpointFunctionBackward"]
    fwd_lo, fwd_hi = fwd.ts.min(), (fwd.ts + fwd.dur).max()
    bwd_lo, bwd_hi = bwd.ts.min(), (bwd.ts + bwd.dur).max()
    in_fwd = (sr.ts >= fwd_lo) & (sr.ts < fwd_hi)
    in_bwd = (sr.ts >= bwd_lo) & (sr.ts < bwd_hi)
    for label, mask in (("fwd-span", in_fwd), ("bwd-span", in_bwd), ("neither", ~(in_fwd | in_bwd))):
        dd = sr.dur.to_numpy()[mask.to_numpy()] / 1e6
        if len(dd):
            print(f"  {label}: n={len(dd)} total={dd.sum()/1e3:.2f}s p50={np.percentile(dd,50):.2f}ms p90={np.percentile(dd,90):.2f}ms")

    # host-block windows (both threads)
    hb = q("""SELECT ts, dur FROM slice
              WHERE (name='aten::nonzero' AND dur>5e6)
                 OR (name IN ('cudaMemcpyAsync','cudaStreamSynchronize') AND dur>1e6)""")
    a = hb.ts.to_numpy(); b = (hb.ts + hb.dur).to_numpy()
    o = np.argsort(a); a, b = a[o], b[o]
    ms, me = [], []
    cs, ce = a[0], b[0]
    for s_, e_ in zip(a[1:], b[1:]):
        if s_ <= ce: ce = max(ce, e_)
        else: ms.append(cs); me.append(ce); cs, ce = s_, e_
    ms.append(cs); me.append(ce)
    ms = np.array(ms); me = np.array(me)
    cover = (me - ms).sum() / 1e9
    print(f"host-block coverage: {cover:.2f}s ({100*cover/wall:.0f}% of wall), {len(ms)} merged windows")

    # comm-stream idle: total, inside vs outside host-block windows
    tr = sr.track_id.iloc[0]
    lo, hi = int(S), int(E)
    srt = sr.ts.to_numpy(); sre = srt + sr.dur.to_numpy()
    def idle_in(windows):
        tot = 0.0
        for ws, we in windows:
            m = (sre > ws) & (srt < we)
            if not m.any():
                tot += we - ws; continue
            aa = np.sort(srt[m]); bb = sre[m][np.argsort(srt[m])]
            busy = 0.0; c2, e2 = aa[0], bb[0]
            for s2_, e2_ in zip(aa[1:], bb[1:]):
                if s2_ <= e2: e2 = max(e2, e2_)
                else: busy += e2 - c2; c2, e2 = s2_, e2_
            busy += e2 - c2
            tot += (we - ws) - min(busy, we - ws)
        return tot / 1e9
    idle_total = idle_in([(lo, hi)])
    idle_in_hb = idle_in(zip(ms, me))
    print(f"comm-stream(SendRecv track {tr}) idle in step: {idle_total:.2f}s "
          f"| inside host-block windows {idle_in_hb:.2f}s | outside {idle_total - idle_in_hb:.2f}s")

    # GPU-union idle
    gpu = q("SELECT ts, dur FROM slice WHERE category IN ('kernel','gpu_memcpy','gpu_memset') AND dur>0")
    g0 = gpu.ts.to_numpy(); g1 = g0 + gpu.dur.to_numpy()
    o = np.argsort(g0); g0, g1 = g0[o], g1[o]
    busy = 0; c, e = int(g0[0]), int(g1[0])
    for s_, e_ in zip(g0[1:], g1[1:]):
        if s_ <= e: e = max(e, int(e_))
        else: busy += e - c; c, e = int(s_), int(e_)
    busy += e - c
    print(f"GPU-union idle: {((E-S)-busy)/1e9:.2f}s")

    # inter-call gaps on the comm stream (consecutive SendRecv end->next start)
    oo = np.argsort(srt); srt_s, sre_s = srt[oo], sre[oo]
    gaps = (srt_s[1:] - sre_s[:-1]) / 1e6
    gaps = gaps[gaps > 0]
    print(f"SendRecv inter-call gaps: n={len(gaps)} total={gaps.sum()/1e3:.2f}s "
          f"p50={np.percentile(gaps,50):.2f}ms p90={np.percentile(gaps,90):.2f}ms max={gaps.max():.2f}ms")

if __name__ == "__main__":
    main(sys.argv[1])
