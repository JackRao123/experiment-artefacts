#!/usr/bin/env python3
"""SM-slack measurement for the W3-on-B300 headroom question (kepler's order).

Question: is there SM slack in the backward phase that ANY second-stream
ordering scheme could fill (W3 v3 has headroom), or do the bwd-phase kernels
saturate the SMs (W3-on-B300 fundamentally bound)?

Evidence: kineto 'est. achieved occupancy %' per kernel, duration-weighted.
Slack-seconds = sum(dur x (1 - occ/100)) — the SM-time a second stream could
have filled. Reported for the bwd phase, the drain windows, and split by
compute vs comm streams.
"""
import sys
import numpy as np
from perfetto.trace_processor import TraceProcessor

def q(tp, sql):
    return tp.query(sql).as_pandas_dataframe()

def analyze(path, bwd_name):
    tp = TraceProcessor(trace=path)
    bwd = q(tp, f"SELECT ts, dur FROM slice WHERE name='{bwd_name}' AND dur>0")
    lo = int(bwd.ts.min()); hi = int((bwd.ts + bwd.dur).max())
    phase_s = (hi - lo) / 1e9

    k = q(tp, """
        SELECT s.ts, s.dur,
               CAST(oc.display_value AS REAL) AS occ,
               COALESCE(stv.display_value, '(none)') AS stream,
               s.name
        FROM slice s
        JOIN args oc ON oc.arg_set_id = s.arg_set_id AND oc.key='args.est. achieved occupancy %'
        LEFT JOIN args stv ON stv.arg_set_id = s.arg_set_id AND stv.key='args.stream'
        WHERE s.category='kernel' AND s.dur>0
    """)
    ks = k.ts.to_numpy(dtype=np.int64); ke = ks + k.dur.to_numpy(dtype=np.int64)
    in_phase = (ks >= lo) & (ks < hi)
    kp = k[in_phase].copy()
    kp["slack"] = kp.dur * (1.0 - kp.occ.clip(0, 100) / 100.0)
    tot_s = kp.dur.sum() / 1e9
    slack_s = kp.slack.sum() / 1e9

    print(f"\n=== {path.split('/')[-1]} (bwd={bwd_name}) ===")
    print(f"bwd phase extent: {phase_s:.2f}s | kernel time in phase: {tot_s:.2f}s "
          f"| kernels with occupancy data: {len(kp)}")
    print(f"SLACK (dur-weighted unused SM capacity): {slack_s:.2f}s "
          f"= {100*slack_s/tot_s:.1f}% of in-phase kernel time")
    buckets = [0, 10, 25, 50, 75, 90, 100.1]
    labels = ["<10%", "10-25%", "25-50%", "50-75%", "75-90%", ">90%"]
    kp["bucket"] = np.digitize(kp.occ, buckets) - 1
    for i, lab in enumerate(labels):
        sub = kp[kp.bucket == i]
        print(f"  occ {lab:8s}: {sub.dur.sum()/1e9:7.2f}s ({100*sub.dur.sum()/kp.dur.sum():5.1f}% of kernel time) "
              f"slack {sub.slack.sum()/1e9:6.2f}s  n={len(sub)}")
    # comm vs compute split
    comm = kp[kp.name.str.contains("SendRecv", na=False)]
    comp = kp[~kp.name.str.contains("SendRecv", na=False)]
    print(f"  SendRecv kernels: {comm.dur.sum()/1e9:.2f}s, mean occ "
          f"{np.average(comm.occ, weights=comm.dur):.1f}%, slack {comm.slack.sum()/1e9:.2f}s")
    print(f"  compute kernels:  {comp.dur.sum()/1e9:.2f}s, mean occ "
          f"{np.average(comp.occ, weights=comp.dur):.1f}%, slack {comp.slack.sum()/1e9:.2f}s")
    # drains
    dr = q(tp, "SELECT ts, dur FROM slice WHERE name='aten::nonzero' AND dur>5e6")
    ds = dr.ts.to_numpy(dtype=np.int64); de = ds + dr.dur.to_numpy(dtype=np.int64)
    in_drain = np.zeros(len(k), dtype=bool)
    ks_all = k.ts.to_numpy(dtype=np.int64)
    for s_, e_ in zip(ds, de):
        in_drain |= (ks_all >= s_) & (ks_all < e_)
    kd = k[in_drain & in_phase].copy()
    kd["slack"] = kd.dur * (1.0 - kd.occ.clip(0, 100) / 100.0)
    print(f"  DRAIN-WINDOW kernels: {kd.dur.sum()/1e9:.2f}s, mean occ "
          f"{np.average(kd.occ, weights=kd.dur):.1f}%, slack {kd.slack.sum()/1e9:.2f}s")

analyze(sys.argv[1], sys.argv[2])
