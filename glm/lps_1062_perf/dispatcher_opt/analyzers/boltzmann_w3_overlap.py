#!/usr/bin/env python3
"""W3 (BT_MOE_LOOKAHEAD_RECOMPUTE) mechanism metrics — baseline + acceptance.

Metrics (minkowski ESTATE_NOTES W3 watch-list amendments, binding at readout):
 1. bwd-window comm overlap: % of CheckpointFunctionBackward union time covered
    by SendRecv kernel intervals (bar: >=50% under W3).
 2. drain-window kernel concurrency: during >5ms aten::nonzero drains, mean
    active-kernel count and % of drain time with >=2 concurrent kernels
    (W3 win = kicked-recompute kernels overlapping INTO these windows; the
    C′-on baseline is ~1.0 / ~0%).
 3. per-stream kernel time (top streams): the lookahead side stream must
    appear as a distinct stream carrying recompute work.
 4. wait-by-cause buckets (NOT totals): drains / eventSyncs (fwd+replay) /
    streamSyncs / memcpys.

Run on the C′-ON baseline (arm-cprime traces) for the reference column, then
on the W3 capture for the readout. v56.1 shell via perf-venv.
"""
import sys
import numpy as np
from perfetto.trace_processor import TraceProcessor


def q(tp, sql):
    return tp.query(sql).as_pandas_dataframe()


def merged(a, b):
    if len(a) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    order = np.argsort(a, kind="stable")
    a, b = a[order], b[order]
    out_s, out_e = [], []
    cs, ce = int(a[0]), int(b[0])
    for s_, e_ in zip(a[1:], b[1:]):
        if s_ <= ce:
            if e_ > ce:
                ce = int(e_)
        else:
            out_s.append(cs); out_e.append(ce); cs, ce = int(s_), int(e_)
    out_s.append(cs); out_e.append(ce)
    return np.array(out_s), np.array(out_e)


def cover(win_s, win_e, iv_s, iv_e):
    """Seconds of [win_s,win_e) covered by the merged intervals (iv_s,iv_e)."""
    import bisect
    tot = 0
    for s_, e_ in zip(win_s, win_e):
        i = max(0, bisect.bisect_right(iv_s, s_) - 1)
        while i < len(iv_s) and iv_s[i] < e_:
            lo = max(s_, iv_s[i]); hi = min(e_, iv_e[i])
            if hi > lo:
                tot += hi - lo
            i += 1
    return tot / 1e9


def main(path):
    tp = TraceProcessor(trace=path)

    # --- backward windows: union of checkpoint-backward slices --------------
    # (W3 replaces the Function: LookaheadCheckpointFunctionBackward; fall back
    # to CheckpointFunctionBackward for pre-W3 traces)
    bwd = q(tp, "SELECT ts, dur FROM slice WHERE name IN ('CheckpointFunctionBackward','LookaheadCheckpointFunctionBackward') AND dur>0")
    bwd_name = q(tp, "SELECT name, COUNT(*) AS n FROM slice WHERE name IN ('CheckpointFunctionBackward','LookaheadCheckpointFunctionBackward') GROUP BY name")
    bw_s = bwd.ts.to_numpy(dtype=np.int64)
    bw_e = bw_s + bwd.dur.to_numpy(dtype=np.int64)
    bwd_ms, bwd_me = merged(bw_s.copy(), bw_e.copy())
    bwd_union_s = float((bwd_me - bwd_ms).sum() / 1e9)

    # --- SendRecv kernels ----------------------------------------------------
    sr = q(tp, "SELECT ts, dur FROM slice WHERE category='kernel' AND name LIKE '%SendRecv%' AND dur>0")
    sr_ms, sr_me = merged(sr.ts.to_numpy(dtype=np.int64).copy(),
                          (sr.ts + sr.dur).to_numpy(dtype=np.int64).copy())
    sr_in_bwd = cover(bwd_ms, bwd_me, sr_ms, sr_me)
    bwd_comm_overlap_pct = 100.0 * sr_in_bwd / bwd_union_s if bwd_union_s else 0.0

    # --- kernel concurrency timeline (all gpu work) --------------------------
    k = q(tp, "SELECT ts, dur FROM slice WHERE category IN ('kernel','gpu_memcpy','gpu_memset') AND dur>0")
    ks = k.ts.to_numpy(dtype=np.int64)
    ke = ks + k.dur.to_numpy(dtype=np.int64)
    ev_t = np.concatenate([ks, ke])
    ev_d = np.concatenate([np.ones(len(ks), dtype=np.int64), -np.ones(len(ke), dtype=np.int64)])
    order = np.argsort(ev_t, kind="stable")
    ev_t, ev_d = ev_t[order], ev_d[order]
    conc = np.cumsum(ev_d)  # active count after each event

    # --- drains ---------------------------------------------------------------
    dr = q(tp, "SELECT ts, dur FROM slice WHERE name='aten::nonzero' AND dur > 5e6 ORDER BY ts")
    import bisect
    drain_tot_s = 0.0
    drain_multi_s = 0.0
    drain_conc_weighted = 0.0
    for s_, e_ in zip(dr.ts.to_numpy(dtype=np.int64), (dr.ts + dr.dur).to_numpy(dtype=np.int64)):
        i0 = bisect.bisect_left(ev_t, s_)
        i1 = bisect.bisect_left(ev_t, e_)
        if i0 == i1:
            c = conc[i0 - 1] if i0 > 0 else 0
            drain_tot_s += (e_ - s_) / 1e9
            drain_conc_weighted += c * (e_ - s_) / 1e9
            if c >= 2:
                drain_multi_s += (e_ - s_) / 1e9
            continue
        # concurrency during [s_, ev_t[i0]) is conc[i0-1] (or 0)
        pts = [int(s_)] + [int(x) for x in ev_t[i0:i1]] + [int(e_)]
        # concurrency value on segment [pts[j], pts[j+1]): index of last event <= pts[j]
        for j in range(len(pts) - 1):
            a_, b_ = pts[j], pts[j + 1]
            if b_ <= a_:
                continue
            idx = bisect.bisect_right(ev_t, a_) - 1
            c = conc[idx] if idx >= 0 else 0
            dt = (b_ - a_) / 1e9
            drain_tot_s += dt
            drain_conc_weighted += c * dt
            if c >= 2:
                drain_multi_s += dt
    drain_mean_conc = drain_conc_weighted / drain_tot_s if drain_tot_s else 0.0
    drain_multi_pct = 100.0 * drain_multi_s / drain_tot_s if drain_tot_s else 0.0

    # --- per-stream kernel time ----------------------------------------------
    st = q(tp, """
        SELECT COALESCE(stv.display_value, '(none)') AS stream, COUNT(*) AS n,
               SUM(s.dur)/1e9 AS tot_s
        FROM slice s LEFT JOIN args stv
          ON stv.arg_set_id = s.arg_set_id AND stv.key = 'args.stream'
        WHERE s.category='kernel' GROUP BY stream ORDER BY tot_s DESC LIMIT 8
    """)

    # --- wait-by-cause buckets -------------------------------------------------
    nz = q(tp, "SELECT SUM(dur)/1e9 AS s, COUNT(*) AS n FROM slice WHERE name='aten::nonzero'")
    es = q(tp, """SELECT p.name AS parent, COUNT(*) AS n, SUM(s.dur)/1e9 AS s
                  FROM slice s JOIN slice p ON s.parent_id = p.id
                  WHERE s.name='cudaEventSynchronize' GROUP BY parent""")
    ssync = q(tp, "SELECT SUM(dur)/1e9 AS s, COUNT(*) AS n FROM slice WHERE name='cudaStreamSynchronize'")
    mc = q(tp, "SELECT SUM(dur)/1e9 AS s, COUNT(*) AS n FROM slice WHERE name='cudaMemcpyAsync'")

    print(f"trace: {path}")
    print(f"== W3 mechanism metrics ==")
    print(f"-- checkpoint backward slices by name --")
    print(bwd_name.to_string(index=False))
    print(f"bwd_window_union_s        {bwd_union_s:10.2f}  (CheckpointFunctionBackward union)")
    print(f"sendrecv_in_bwd_windows_s {sr_in_bwd:10.2f}")
    print(f"bwd_comm_overlap_pct      {bwd_comm_overlap_pct:10.2f}  (bar under W3: >=50)")
    print(f"drain_total_s             {drain_tot_s:10.2f}  ({len(dr)} drains >5ms)")
    print(f"drain_mean_kernel_conc    {drain_mean_conc:10.3f}  (W3 win: rises above ~1.0)")
    print(f"drain_multi_kernel_pct    {drain_multi_pct:10.2f}  (W3 win: rises above ~0)")
    print(f"-- per-stream kernel time (top) --")
    print(st.to_string(index=False))
    # --- SendRecv structure (for the sendrecv_gpu_s question) ----------------
    sr2 = q(tp, """
        SELECT dt.display_value AS dtype, COUNT(*) AS n, SUM(s.dur)/1e9 AS tot_s
        FROM slice s JOIN args dt ON dt.arg_set_id = s.arg_set_id AND dt.key='args.dtype'
        WHERE s.category='kernel' AND s.name LIKE '%SendRecv%' GROUP BY dtype
    """)
    print(f"-- SendRecv by dtype --")
    print(sr2.to_string(index=False))

    print(f"-- wait by cause --")
    print(f"nonzero (drains+layout):  n={int(nz.n.iloc[0]):6d}  {nz.s.iloc[0]:8.2f}s")
    for _, r in es.iterrows():
        print(f"eventSync under {r.parent:34s} n={int(r.n):6d}  {r.s:8.2f}s")
    print(f"streamSync total:         n={int(ssync.n.iloc[0]):6d}  {ssync.s.iloc[0]:8.2f}s")
    print(f"memcpyAsync total:        n={int(mc.n.iloc[0]):6d}  {mc.s.iloc[0]:8.2f}s")


if __name__ == "__main__":
    main(sys.argv[1])
