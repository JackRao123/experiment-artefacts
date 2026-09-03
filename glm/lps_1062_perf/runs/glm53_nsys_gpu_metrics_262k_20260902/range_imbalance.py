#!/usr/bin/env python3
"""Per-instance, per-rank GPU time of a repeated NVTX range: how unbalanced is it?

For a range such as ``moe_experts`` (one instance per MoE layer per pass) or
``attention``, the k-th instance on every rank is the same layer/pass. For each k
this reports GPU kernel time per rank (launch-time attribution), the max/mean
ratio and the time the fastest ranks would wait for the slowest. Summed over k
this is an upper bound on what per-layer imbalance of that range costs.

Writes <out>/imbalance_<range>.csv and prints a summary per range.
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
    ap.add_argument("--ranges", default="moe_experts,attention,dsa_core,self_attention,mlp,layer,hybridep_dispatch")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    con = sqlite3.connect(args.sqlite)
    cur = con.cursor()
    S = {i: v for i, v in cur.execute("SELECT id, value FROM StringIds")}
    fb = cur.execute("SELECT start, end FROM NVTX_EVENTS WHERE eventType IN (59,60) AND text='forward_backward'").fetchall()
    W0, W1 = min(r[0] for r in fb), max(r[1] for r in fb)

    launch = {}
    for s, gtid, corr, nid in cur.execute("SELECT start, globalTid, correlationId, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>=? AND start<=?", (W0 - 10**9, W1)):
        if "aunch" in S.get(nid, ""):
            launch[((gtid >> 24) << 24, corr)] = s
    kern = cur.execute("SELECT start, end, deviceId, globalPid, correlationId, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?", (W0, W1)).fetchall()
    per_pid_launch = defaultdict(list)
    per_pid_dur = defaultdict(list)
    per_pid_sync = defaultdict(list)
    pid_dev = {}
    for s, e, d, pid, corr, nid in kern:
        lt = launch.get((pid, corr))
        if lt is None:
            continue
        pid_dev[pid] = d
        per_pid_launch[pid].append(lt)
        per_pid_dur[pid].append(e - s)
        per_pid_sync[pid].append("device_sync_kernel" in S.get(nid, ""))
    pids = sorted(per_pid_launch, key=lambda p: pid_dev[p])
    L = {p: np.array(per_pid_launch[p]) for p in pids}
    D = {p: np.array(per_pid_dur[p]) for p in pids}
    SY = {p: np.array(per_pid_sync[p]) for p in pids}
    for p in pids:
        o = np.argsort(L[p])
        L[p], D[p], SY[p] = L[p][o], D[p][o], SY[p][o]
    cumD = {p: np.concatenate([[0], np.cumsum(D[p])]) for p in pids}
    cumS = {p: np.concatenate([[0], np.cumsum(np.where(SY[p], D[p], 0))]) for p in pids}

    op_suffix = re.compile(r",\s*(op_id|seq)\s*=\s*\d+")
    want = set(args.ranges.split(","))
    inst = defaultdict(lambda: defaultdict(list))
    for s, e, gtid, text, tid in cur.execute("SELECT start, end, globalTid, text, textId FROM NVTX_EVENTS WHERE eventType IN (59,60) AND end IS NOT NULL AND start>=? AND end<=?", (W0, W1)):
        nm = op_suffix.sub("", text if text is not None else S.get(tid, "?")).strip()
        if nm.startswith("layer:"):
            nm = "layer"
        elif nm.endswith("._forward_attention.self_attention"):
            nm = "self_attention"
        elif nm.endswith("._run_mlp.mlp"):
            nm = "mlp"
        if nm in want:
            inst[nm][(gtid >> 24) << 24].append((s, e))

    def gpu_time(p, a, b, exclude_sync=True):
        i = np.searchsorted(L[p], a, side="left")
        j = np.searchsorted(L[p], b, side="right")
        t = cumD[p][j] - cumD[p][i]
        if exclude_sync:
            t -= cumS[p][j] - cumS[p][i]
        return t

    for nm in sorted(inst):
        per = {p: sorted(v) for p, v in inst[nm].items() if p in pid_dev}
        n = min(len(v) for v in per.values())
        if n == 0:
            continue
        rows = []
        tot_wait = 0.0
        tot_mean = 0.0
        ratios = []
        heavy = defaultdict(int)
        for k in range(n):
            times = np.array([gpu_time(p, per[p][k][0], per[p][k][1]) for p in pids], dtype=np.float64) / 1e6
            mx, mean = times.max(), times.mean()
            wait = float((mx - times).sum()) / len(pids)  # per-GPU average wait if everyone waits for the slowest
            tot_wait += wait
            tot_mean += mean
            ratios.append(mx / mean if mean > 0 else 1.0)
            heavy[pid_dev[pids[int(times.argmax())]]] += 1
            rows.append([k, round(mean, 3), round(mx, 3), round(mx / mean if mean > 0 else 1, 3), pid_dev[pids[int(times.argmax())]]] + [round(x, 3) for x in times])
        with open(os.path.join(args.out, f"imbalance_{nm}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["k", "mean_ms", "max_ms", "max_over_mean", "heaviest_gpu"] + [f"gpu{pid_dev[p]}_ms" for p in pids])
            w.writerows(rows)
        r = np.array(ratios)
        print(f"\n## {nm}: {n} instances/rank, GPU kernel time excl. sync kernels")
        print(f"mean per instance {tot_mean / n:.2f} ms; max/mean ratio median {np.median(r):.3f}, p90 {np.quantile(r, .9):.3f}, max {r.max():.3f}")
        print(f"if every rank waited for the slowest at each instance: {tot_wait / 1e3:.2f} s per GPU per step ({tot_wait / max(tot_mean, 1e-9) * 100:.1f}% of the range's mean time)")
        print("heaviest rank count:", dict(sorted(heavy.items())))


if __name__ == "__main__":
    main()
