#!/usr/bin/env python3
"""GPU kernel time by (kernel category) x (innermost NVTX range that launched it).

Answers "who owns the copies / cats / elementwise time": each kernel is placed in
the innermost of a fixed list of ranges (by launch time on its process), e.g.
forward > layer > attention, or backward > recompute > (layer backward, no inner
range). Writes <out>/category_by_range.csv and prints a pivot (s per GPU).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nsys_attrib import categorize  # noqa: E402

# innermost-first priority
INNER = ["lm_head_gemm", "lm_head", "dsa_indexer", "dsa_core", "attention", "hybridep_dispatch_bwd", "hybridep_combine_bwd",
         "hybridep_dispatch", "hybridep_combine", "moe_route", "moe_dispatch", "moe_experts", "moe_combine", "moe_shared_experts",
         "moe", "self_attention", "mlp", "layer", "recompute", "forward", "backward", "optimizer"]


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

    launch = {}
    for s, gtid, corr, nid in cur.execute("SELECT start, globalTid, correlationId, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>=? AND start<=?", (W0 - 10**9, W1)):
        if "aunch" in S.get(nid, ""):
            launch[((gtid >> 24) << 24, corr)] = s
    kern = cur.execute("SELECT start, end, deviceId, globalPid, correlationId, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?", (W0, W1)).fetchall()
    ndev = len({k[2] for k in kern})
    cat_of = {}
    for k in kern:
        if k[5] not in cat_of:
            cat_of[k[5]] = categorize(S.get(k[5], "?"))

    op_suffix = re.compile(r",\s*(op_id|seq)\s*=\s*\d+")
    ranges: dict[tuple[int, str], list] = defaultdict(list)
    for s, e, gtid, text, tid in cur.execute("SELECT start, end, globalTid, text, textId FROM NVTX_EVENTS WHERE eventType IN (59,60) AND end IS NOT NULL AND start>=? AND end<=?", (W0 - 10**9, W1 + 10**9)):
        nm = op_suffix.sub("", text if text is not None else S.get(tid, "?")).strip()
        if nm.startswith("layer:"):
            nm = "layer"
        elif nm.endswith("._forward_attention.self_attention"):
            nm = "self_attention"
        elif nm.endswith("._run_mlp.mlp"):
            nm = "mlp"
        if nm in INNER:
            ranges[((gtid >> 24) << 24, nm)].append((s, e))
    arrs = {k: np.array(sorted(v), dtype=np.int64) for k, v in ranges.items()}

    def innermost(pid: int, t: int) -> str:
        path = []
        for nm in INNER:
            a = arrs.get((pid, nm))
            if a is None:
                continue
            i = np.searchsorted(a[:, 0], t, side="right") - 1
            for j in range(i, max(-1, i - 3), -1):
                if a[j, 0] <= t <= a[j, 1]:
                    path.append(nm)
                    break
        if not path:
            return "-"
        # innermost = first in INNER order; also keep the phase
        phase = next((p for p in path if p in ("forward", "backward", "optimizer")), "-")
        inner = path[0] if path[0] not in ("forward", "backward", "optimizer") else "(phase only)"
        if inner in ("layer", "recompute") and "recompute" in path and phase == "backward":
            inner = "layer_backward_proper" if inner == "recompute" else "layer_recompute_fwd"
        return f"{phase}>{inner}"

    pivot: dict[tuple[str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for s, e, d, pid, corr, nid in kern:
        lt = launch.get((pid, corr))
        where = innermost(pid, lt) if lt is not None else "?"
        c = cat_of[nid]
        pivot[(c, where)] += (e - s) / 1e9 / ndev
        counts[(c, where)] += 1
    rows = sorted(([c, w, round(v, 3), counts[(c, w)] // ndev] for (c, w), v in pivot.items()), key=lambda r: -r[2])
    with open(os.path.join(args.out, "category_by_range.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "phase>innermost_range", "s_per_gpu", "kernels_per_gpu"])
        w.writerows(rows)
    cats = sorted({r[0] for r in rows}, key=lambda c: -sum(r[2] for r in rows if r[0] == c))
    wheres = sorted({r[1] for r in rows}, key=lambda w: -sum(r[2] for r in rows if r[1] == w))
    print("s per GPU; rows = category, cols = phase>innermost range\n")
    print("| category | total | " + " | ".join(wheres) + " |")
    print("|---|---|" + "---|" * len(wheres))
    for c in cats:
        tot = sum(r[2] for r in rows if r[0] == c)
        cells = []
        for w in wheres:
            v = pivot.get((c, w), 0.0)
            cells.append(f"{v:.2f}" if v >= 0.005 else "")
        print(f"| {c} | {tot:.2f} | " + " | ".join(cells) + " |")
    print("| **total** | " + f"{sum(r[2] for r in rows):.2f} | " + " | ".join(f"{sum(r[2] for r in rows if r[1] == w):.2f}" for w in wheres) + " |")


if __name__ == "__main__":
    main()
