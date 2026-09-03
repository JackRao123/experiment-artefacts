#!/usr/bin/env python3
"""Why do ranks wait in HybridEP device_sync? Arrival analysis per collective.

The k-th ``hybrid_ep::device_sync_kernel`` on every rank belongs to the same
collective (all ranks run the same layer sequence). For each k:

  arrival_d  = start of the k-th sync kernel on GPU d
  release    = max over d of its end (all ranks leave together)
  wait_d     = release - arrival_d
  laggard    = argmax arrival_d;  lateness = arrival_laggard - median(arrival)

Then, for the laggard, look at the interval between the previous collective's
release and its arrival and split it into GPU-idle time (no kernel resident:
the host was not feeding it) and GPU-busy time (it had more work). The share of
lateness covered by idle time is the host-stall part; the remainder is work
imbalance. This is measured on the trace, not inferred from totals.

Outputs <out>/sync_arrivals.csv (per collective) and a printed summary.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
from collections import defaultdict

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    con = sqlite3.connect(args.sqlite)
    cur = con.cursor()
    S = {i: v for i, v in cur.execute("SELECT id, value FROM StringIds")}

    fb = cur.execute("SELECT start, end FROM NVTX_EVENTS WHERE eventType IN (59,60) AND text='forward_backward'").fetchall()
    W0, W1 = min(r[0] for r in fb), max(r[1] for r in fb)

    rows = cur.execute("SELECT start, end, deviceId, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?", (W0, W1)).fetchall()
    dev_k: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for s, e, d, nm in rows:
        dev_k[d].append((s, e, S.get(nm, "?")))
    devices = sorted(dev_k)
    busy: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    syncs: dict[int, np.ndarray] = {}
    for d in devices:
        arr = sorted(dev_k[d])
        s = np.array([a[0] for a in arr], dtype=np.int64)
        e = np.array([a[1] for a in arr], dtype=np.int64)
        busy[d] = (s, np.maximum.accumulate(e))
        sy = [(a[0], a[1]) for a in arr if "hybrid_ep::device_sync_kernel" in a[2]]
        syncs[d] = np.array(sy, dtype=np.int64).reshape(-1, 2)
    counts = {d: len(syncs[d]) for d in devices}
    print("sync kernels per GPU:", counts)
    n = min(counts.values())

    def idle_in(d: int, a: int, b: int) -> int:
        """ns with no kernel resident on GPU d inside [a, b]."""
        s, e = busy[d]
        lo = np.searchsorted(e, a, side="right")
        hi = np.searchsorted(s, b, side="left")
        t = a
        idle = 0
        for i in range(max(lo - 1, 0), hi):
            if e[i] <= t:
                continue
            if s[i] > t:
                idle += min(s[i], b) - t
            t = max(t, min(e[i], b))
            if t >= b:
                break
        if t < b:
            idle += b - t
        return idle

    # NVTX phase per time for the laggard (which range the laggard sat in when it arrived)
    op_suffix = re.compile(r",\s*(op_id|seq)\s*=\s*\d+")
    want = {"forward", "backward", "attention", "moe", "moe_experts", "hybridep_dispatch", "hybridep_combine",
            "hybridep_dispatch_bwd", "hybridep_combine_bwd", "recompute", "lm_head"}
    pid_of_dev: dict[int, int] = {}
    for gp, d in cur.execute("SELECT DISTINCT globalPid, deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?", (W0, W1)):
        pid_of_dev[d] = gp
    ranges: dict[tuple[int, str], list[tuple[int, int]]] = defaultdict(list)
    for s, e, gtid, text, text_id in cur.execute(
            "SELECT start, end, globalTid, text, textId FROM NVTX_EVENTS WHERE eventType IN (59,60) AND end IS NOT NULL AND start>=? AND end<=?", (W0 - 10**9, W1 + 10**9)):
        nm = op_suffix.sub("", text if text is not None else S.get(text_id, "?")).strip()
        if nm.startswith("layer:"):
            nm = "layer"
        if nm in want or nm == "layer":
            ranges[((gtid >> 24) << 24, nm)].append((s, e))
    for k in ranges:
        ranges[k].sort()

    def where(d: int, t: int) -> str:
        pid = pid_of_dev.get(d)
        hits = []
        for nm in ["forward", "backward", "recompute", "layer", "attention", "moe", "moe_experts", "hybridep_dispatch",
                   "hybridep_combine", "hybridep_dispatch_bwd", "hybridep_combine_bwd", "lm_head"]:
            inst = ranges.get((pid, nm), [])
            if not inst:
                continue
            arr = np.array(inst, dtype=np.int64)
            i = np.searchsorted(arr[:, 0], t, side="right") - 1
            for j in range(i, max(-1, i - 3), -1):
                if arr[j, 0] <= t <= arr[j, 1]:
                    hits.append(nm)
                    break
        return "|".join(hits) if hits else "-"

    out_rows = []
    tot_wait = 0.0
    tot_late = 0.0
    late_idle = 0.0
    late_busy = 0.0
    by_lag = defaultdict(float)
    by_where = defaultdict(lambda: [0.0, 0.0, 0.0])  # lateness, idle part, busy part
    prev_release = {d: W0 for d in devices}
    for k in range(n):
        arr = np.array([syncs[d][k] for d in devices])
        arrivals = arr[:, 0]
        release = int(arr[:, 1].max())
        waits = release - arrivals
        lag_i = int(np.argmax(arrivals))
        lag = devices[lag_i]
        med = float(np.median(arrivals))
        lateness = float(arrivals[lag_i] - med)
        a0 = prev_release[lag]
        a1 = int(arrivals[lag_i])
        idle = idle_in(lag, a0, a1) if a1 > a0 else 0
        idle_part = min(float(idle), lateness)
        busy_part = lateness - idle_part
        wh = where(lag, a1)
        out_rows.append([k, lag, round(lateness / 1e6, 3), round(idle / 1e6, 3), round((a1 - a0) / 1e6, 3),
                         round(float(waits.sum()) / 1e6, 3), wh])
        tot_wait += float(waits.sum()) / 1e9
        tot_late += lateness / 1e9
        late_idle += idle_part / 1e9
        late_busy += busy_part / 1e9
        by_lag[lag] += lateness / 1e9
        w = by_where[wh]
        w[0] += lateness / 1e9
        w[1] += idle_part / 1e9
        w[2] += busy_part / 1e9
        for d in devices:
            prev_release[d] = release
    with open(os.path.join(args.out, "sync_arrivals.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "laggard_gpu", "lateness_ms", "laggard_gpu_idle_since_prev_release_ms", "laggard_interval_ms",
                    "sum_wait_all_gpus_ms", "laggard_nvtx"])
        w.writerows(out_rows)
    nd = len(devices)
    print(f"\ncollectives matched: {n}; total sync wait all GPUs {tot_wait:.2f} s ({tot_wait / nd:.2f} s/GPU)")
    print(f"sum of last-arrival lateness (vs median arrival): {tot_late:.2f} s")
    print(f"  covered by laggard GPU idle time since previous release (host stall): {late_idle:.2f} s ({late_idle / max(tot_late, 1e-9) * 100:.0f}%)")
    print(f"  laggard GPU busy the whole time (more work / slower kernels):        {late_busy:.2f} s ({late_busy / max(tot_late, 1e-9) * 100:.0f}%)")
    print("\nlateness by laggard GPU (s):", {d: round(v, 2) for d, v in sorted(by_lag.items())})
    print("\n| laggard was in | lateness s | host-stall part s | busy part s |\n|---|---|---|---|")
    for wh, (a, b, c) in sorted(by_where.items(), key=lambda kv: -kv[1][0])[:12]:
        print(f"| {wh} | {a:.2f} | {b:.2f} | {c:.2f} |")
    late = sorted(out_rows, key=lambda r: -r[2])[:10]
    print("\nworst 10 collectives (k, laggard, lateness ms, laggard idle ms, interval ms, where):")
    for r in late:
        print("  ", r[0], r[1], r[2], r[3], r[4], r[6])


if __name__ == "__main__":
    main()
