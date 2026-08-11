#!/usr/bin/env python3
"""C' drain-wait attribution: busy-CPU (a) vs GPU-timeline wait (b).

Compares rank0 cprime (B+F+W1+C') vs rank0 w1-control (B+F+W1, C off):
1. GPU kernel coverage during the >5ms aten::nonzero drains (are the drains
   waiting on a busy GPU?).
2. CPU op total-time diff (did C' add its own busy CPU work?).
3. C'-signature ops (masked_fill/where/topk) under the replay path.
"""
import sys
import numpy as np
from perfetto.trace_processor import TraceProcessor

C = "/Users/jackrao/perf_profiles/lps-1062/round3/arm-cprime-318g61w/rank0_cprime_steady.pt.trace.json"
W = "/Users/jackrao/perf_profiles/lps-1062/round3/arm2-w1-318g61w/rank0_w1_131k-d4-steady.pt.trace.json"


def load(path):
    tp = TraceProcessor(trace=path)
    def q(sql):
        return tp.query(sql).as_pandas_dataframe()
    out = {}
    out["drains"] = q(
        "SELECT ts, dur FROM slice WHERE name='aten::nonzero' AND dur > 5e6 ORDER BY ts"
    )
    out["kern"] = q(
        "SELECT ts, dur FROM slice WHERE category IN ('kernel','gpu_memcpy','gpu_memset') AND dur > 0"
    )
    out["cpu_ops"] = q(
        "SELECT name, COUNT(*) AS n, SUM(dur)/1e9 AS tot_s FROM slice "
        "WHERE category='cpu_op' GROUP BY name ORDER BY tot_s DESC LIMIT 30"
    )
    out["sig"] = q(
        "SELECT s.name, COUNT(*) AS n, SUM(s.dur)/1e9 AS tot_s FROM slice s "
        "JOIN slice p ON s.parent_id = p.id "
        "WHERE (s.name LIKE '%masked_fill%' OR s.name LIKE '%where%' OR s.name LIKE '%topk%') "
        "AND (p.name LIKE '%Checkpoint%' OR p.name LIKE '%Critic%' OR p.name LIKE '%router%' OR p.name LIKE '%Router%') "
        "GROUP BY s.name"
    )
    out["es"] = q("SELECT SUM(dur)/1e9 AS s, COUNT(*) AS n FROM slice WHERE name='cudaEventSynchronize'")
    return out


def coverage(drains, kern):
    """Fraction of drain time during which >=1 GPU kernel/memop was in flight."""
    a = kern.ts.to_numpy(dtype=np.int64)
    b = a + kern.dur.to_numpy(dtype=np.int64)
    order = np.argsort(a, kind="stable")
    a, b = a[order], b[order]
    # merged kernel intervals
    ms, me = [], []
    cs, ce = int(a[0]), int(b[0])
    for s_, e_ in zip(a[1:], b[1:]):
        if s_ <= ce:
            if e_ > ce:
                ce = int(e_)
        else:
            ms.append(cs); me.append(ce); cs, ce = int(s_), int(e_)
    ms.append(cs); me.append(ce)
    ms = np.array(ms); me = np.array(me)
    import bisect
    tot_wait = 0
    tot_cov = 0
    per_drain_cov = []
    for s_, e_ in zip(drains.ts.to_numpy(dtype=np.int64),
                      (drains.ts + drains.dur).to_numpy(dtype=np.int64)):
        tot_wait += e_ - s_
        i = max(0, bisect.bisect_right(ms, s_) - 1)
        cov = 0
        while i < len(ms) and ms[i] < e_:
            lo = max(s_, ms[i]); hi = min(e_, me[i])
            if hi > lo:
                cov += hi - lo
            i += 1
        tot_cov += cov
        per_drain_cov.append(cov / (e_ - s_))
    return tot_wait / 1e9, tot_cov / 1e9, np.array(per_drain_cov)


def main():
    data = {}
    for tag, path in [("cprime", C), ("w1ctrl", W)]:
        print(f"loading {tag} ...", flush=True)
        data[tag] = load(path)

    for tag in ["cprime", "w1ctrl"]:
        d = data[tag]
        tw, tc, pd = coverage(d["drains"], d["kern"])
        print(f"\n[{tag}] drains>5ms: n={len(d['drains'])} total={tw:.2f}s "
              f"gpu-covered={tc:.2f}s ({100*tc/tw:.1f}%)")
        print(f"[{tag}] per-drain gpu-coverage: p10={np.percentile(pd,10):.2f} "
              f"p50={np.percentile(pd,50):.2f} p90={np.percentile(pd,90):.2f}")
        print(f"[{tag}] cudaEventSynchronize: n={int(d['es'].n.iloc[0])} tot={d['es'].s.iloc[0]:.2f}s")

    print("\n== CPU op total-time table (top 15 each) ==")
    for tag in ["cprime", "w1ctrl"]:
        print(f"--- {tag}")
        print(data[tag]["cpu_ops"].head(15).to_string(index=False))

    print("\n== C'-signature ops under replay/router parents ==")
    for tag in ["cprime", "w1ctrl"]:
        print(f"--- {tag}")
        sig = data[tag]["sig"]
        print(sig.to_string(index=False) if len(sig) else "(none)")


if __name__ == "__main__":
    main()
