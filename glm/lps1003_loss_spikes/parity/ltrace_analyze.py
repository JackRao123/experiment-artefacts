#!/usr/bin/env python3
"""LPS-1003 layer-bisection analysis over ltrace dumps.

Input: a directory of rank<R>_pid<P>.jsonl.gz + matching _modmap.json files
(one per rank), plus a way to split ops into EVENT and HEALED groups (the
probe timeline gives ts ranges; ops carry ts).

Method: for every (rank, hook, bin) the healed ops give a noise floor
(mean, std of each stat across healed ops with the same call signature).
Event ops are scored as z = |x - mean| / (std + eps). Report the hooks with
the largest excess z, ordered by layer index — the FIRST layer with large z
is where corruption enters.

Call signature matching: ops are grouped by (entries fingerprint, cu hash) so
partition k of the probe payload is compared against partition k of other reps.

Usage:
  ltrace_analyze.py DIR --event-ts T0 T1 --healed-ts T2 T3 [--rank N] [--top 40]
"""
import argparse
import glob
import gzip
import json
import math
import os
import re
from collections import defaultdict

STATS = ["rms", "absmax", "mean", "nnf_or_neg"]


def load_rank(fp):
    ops = []
    try:
        with gzip.open(fp, "rt") as fh:
            for line in fh:
                ops.append(json.loads(line))
    except (EOFError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        pass  # truncated tail (live pull / crash) — keep what parsed
    return ops


def sig_of(op):
    ent = tuple(tuple(e) for e in op["entries"])
    cu = tuple(op.get("cu") or [])
    return hash((ent, cu))


def unpack(op):
    """-> {(hook_id, bin_idx, stat_idx): value}"""
    vals = {}
    i = 0
    data = op["data"]
    for hid, nb, _nc in op["entries"]:
        for b in range(nb):
            for s in range(4):
                vals[(hid, b, s)] = data[i]
                i += 1
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--event-ts", nargs=2, type=float, required=True)
    ap.add_argument("--healed-ts", nargs=2, type=float, required=True)
    ap.add_argument("--rank", default=None)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--zmin", type=float, default=6.0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "rank*_pid*.jsonl.gz")))
    if args.rank is not None:
        files = [f for f in files if re.search(rf"rank{args.rank}_pid", f)]
    report_rows = []
    for fp in files:
        rank = re.search(r"rank(\w+)_pid", os.path.basename(fp)).group(1)
        mm_fp = fp.replace(".jsonl.gz", "_modmap.json")
        modmap = json.load(open(mm_fp)) if os.path.exists(mm_fp) else {}
        ops = load_rank(fp)
        ev = [o for o in ops if args.event_ts[0] <= o["ts"] <= args.event_ts[1]]
        he = [o for o in ops if args.healed_ts[0] <= o["ts"] <= args.healed_ts[1]]
        if not ev or not he:
            continue
        by_sig = defaultdict(list)
        for o in he:
            by_sig[sig_of(o)].append(o)
        for o in ev:
            peers = by_sig.get(sig_of(o))
            if not peers or len(peers) < 2:
                continue
            pv = [unpack(p) for p in peers]
            x = unpack(o)
            for key, v in x.items():
                ref = [p.get(key) for p in pv if p.get(key) is not None]
                if len(ref) < 2:
                    continue
                m = sum(ref) / len(ref)
                sd = math.sqrt(sum((r - m) ** 2 for r in ref) / (len(ref) - 1))
                z = abs(v - m) / (sd + 1e-9)
                if z >= args.zmin and abs(v - m) > 1e-6 + 0.001 * abs(m):
                    hid, b, s = key
                    path = modmap.get(str(hid), {}).get("path", f"hook{hid}")
                    lay = re.search(r"layers\.(\d+)", path)
                    report_rows.append({
                        "rank": rank, "op": o["op"], "ts": o["ts"],
                        "layer": int(lay.group(1)) if lay else -1,
                        "path": path, "bin": b, "stat": STATS[s],
                        "event": v, "healed_mean": m, "healed_std": sd, "z": z,
                    })
    report_rows.sort(key=lambda r: (r["layer"], -r["z"]))
    seen_layers = sorted({r["layer"] for r in report_rows})
    print(f"# {len(report_rows)} anomalous (hook,bin,stat) cells; layers involved: {seen_layers}")
    if report_rows:
        first_layer = report_rows[0]["layer"]
        print(f"# EARLIEST divergent layer: {first_layer}")
    for r in report_rows[:args.top]:
        print(f"L{r['layer']:>3} {r['path']:<55} bin{r['bin']:>2} {r['stat']:<10} "
              f"rank{r['rank']:>2} op{r['op']} z={r['z']:9.1f} "
              f"event={r['event']:.5g} healed={r['healed_mean']:.5g}±{r['healed_std']:.2g}")


if __name__ == "__main__":
    main()
