#!/usr/bin/env python3
"""Turn glm_prof results (status JSONs + poller CSVs + driver logs) into report tables.

Usage: analyze.py <results_dir> [--poll-dir <dir>] [--gpus-per-node 8]

results_dir layout (written by run_point.sh):
  points.csv          epoch,start|end,seq[,rc=N]
  status_S<seq>.json  /status snapshot after the point (all-rank keys: rankN, rankN:max_reserved)
  driver_S<seq>.log   sft_driver output ([step k/N] loss=... <dt>s lines)

Emits a markdown table + numpy deg-1 fits on (a) max over ranks of peak_alloc
(/status channel) and (b) hottest-GPU nvidia-smi memory.used (poller channel).
"""

import argparse
import glob
import json
import math
import os
import re
import sys

GIB = 2**30


def load_points(results_dir):
    pts = {}  # seq -> {"start": e, "end": e, "rc": n}
    path = os.path.join(results_dir, "points.csv")
    if not os.path.exists(path):
        return pts
    for line in open(path):
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue
        epoch, kind, seq = int(parts[0]), parts[1], int(parts[2])
        d = pts.setdefault(seq, {})
        d[kind] = epoch
        for p in parts[3:]:
            if p.startswith("rc="):
                d["rc"] = int(p[3:])
    return pts


def load_status(results_dir, seq):
    path = os.path.join(results_dir, f"status_S{seq}.json")
    if not os.path.exists(path):
        return None
    try:
        s = json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return None
    ranks_alloc, ranks_res = {}, {}
    for k, v in (s.get("gpu_max_memory_allocated") or {}).items():
        m = re.fullmatch(r"rank(\d+)", k)
        if m:
            ranks_alloc[int(m.group(1))] = v
    for k, v in (s.get("gpu_memory") or {}).items():
        m = re.fullmatch(r"rank(\d+):max_reserved", k)
        if m:
            ranks_res[int(m.group(1))] = v
    return {"alloc": ranks_alloc, "reserved": ranks_res, "raw": s}


def load_driver(results_dir, seq):
    path = os.path.join(results_dir, f"driver_S{seq}.log")
    steps, losses = [], []
    if os.path.exists(path):
        for line in open(path):
            m = re.search(r"\[step (\d+)/(\d+)\] loss=([\d.eE+-]+|nan).* ([\d.]+)s", line)
            if m:
                losses.append(float(m.group(3)))
                steps.append(float(m.group(4)))
    return steps, losses


def load_poll(poll_dir, start, end, gpus_per_node=8):
    """Max memory.used per GPU over [start, end]; returns {(<node_file>, gpu): max_mib}."""
    out = {}
    for f in sorted(glob.glob(os.path.join(poll_dir, "*.csv"))):
        node = os.path.basename(f).replace(".csv", "")
        for line in open(f):
            try:
                ts, idx, used = line.strip().split(",")
                ts, idx, used = int(ts), int(idx), int(used)
            except ValueError:
                continue
            if start <= ts <= end:
                key = (node, idx)
                out[key] = max(out.get(key, 0), used)
    return out


def fit(xs, ys):
    """Deg-1 least squares without numpy dependency."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    m = sxy / sxx
    b = my - m * mx
    ss_res = sum((y - (m * x + b)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return m, b, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--poll-dir", default=None)
    ap.add_argument("--gpus-per-node", type=int, default=8)
    args = ap.parse_args()
    poll_dir = args.poll_dir or os.path.join(
        os.path.dirname(os.path.dirname(args.results_dir.rstrip("/"))), "poll"
    )

    pts = load_points(args.results_dir)
    rows = []
    for seq in sorted(pts):
        d = pts[seq]
        st = load_status(args.results_dir, seq)
        steps, losses = load_driver(args.results_dir, seq)
        pol = (
            load_poll(poll_dir, d.get("start", 0), d.get("end", 0), args.gpus_per_node)
            if "start" in d and "end" in d and os.path.isdir(poll_dir)
            else {}
        )
        alloc = st["alloc"] if st else {}
        res = st["reserved"] if st else {}
        used_vals = sorted(pol.values())
        row = {
            "seq": seq,
            "rc": d.get("rc"),
            "alloc_max": max(alloc.values()) / GIB if alloc else None,
            "alloc_rank0": alloc.get(0, 0) / GIB if alloc else None,
            "res_max": max(res.values()) / GIB if res else None,
            "hottest_used": used_vals[-1] / 1024 if used_vals else None,
            "used_min": used_vals[0] / 1024 if used_vals else None,
            "used_med": used_vals[len(used_vals) // 2] / 1024 if used_vals else None,
            "losses": losses,
            "warm_s": steps[1] if len(steps) > 1 else (steps[0] if steps else None),
            "pol": pol,
        }
        rows.append(row)

    def f(v, nd=2):
        return f"{v:.{nd}f}" if v is not None else "—"

    print(
        "| seq_len | peak_alloc max/all-ranks (GiB) | rank0 | max_reserved max (GiB) | "
        "hottest GPU used (GiB) | used min/med | warm fb+opt (s) | loss steps | status |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        loss_s = " → ".join(f"{x:.3f}" for x in r["losses"]) if r["losses"] else "—"
        status = "ok" if (r["rc"] == 0) else f"rc={r['rc']}"
        print(
            f"| {r['seq']} | {f(r['alloc_max'])} | {f(r['alloc_rank0'])} | {f(r['res_max'])} | "
            f"{f(r['hottest_used'])} | {f(r['used_min'])}/{f(r['used_med'])} | "
            f"{f(r['warm_s'],1)} | {loss_s} | {status} |"
        )

    ok = [r for r in rows if r["rc"] == 0]
    for chan, key in (("peak_alloc(max-rank)", "alloc_max"), ("hottest-used", "hottest_used")):
        pts2 = [(r["seq"], r[key]) for r in ok if r[key] is not None]
        got = fit([p[0] for p in pts2], [p[1] for p in pts2])
        if got:
            m, b, r2 = got
            print(
                f"\nfit[{chan}]: y = {m*1024:.4f} GiB/1k-tok * (S/1024) + {b:.2f} GiB "
                f"(slope {m*GIB/2**20*1:.4f} MiB/tok... = {m:.6f} GiB/tok; R^2={r2:.5f})"
            )
    # per-GPU dump for the largest ok point
    if ok and ok[-1]["pol"]:
        r = ok[-1]
        print(f"\nAll-GPU memory.used (GiB) at seq={r['seq']} (node,gpu sorted):")
        for (node, idx), v in sorted(r["pol"].items()):
            print(f"  {node} gpu{idx}: {v/1024:.1f}")


if __name__ == "__main__":
    main()
