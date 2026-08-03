#!/usr/bin/env python3
"""Correlate harness6 double-exec records with forward reps.

Reads every rank*.jsonl in a dsa_double dir, clusters records into forwards
per rank by time gap (>GAP s), and prints a per-rank x per-forward table of
disagreement counts (indexer: calls with n_diff_elems>0; flashmla: calls with
rows_diff|lse_diff>0) plus totals per forward index across ranks.

Usage: analyze_dd.py DSA_DOUBLE_DIR [GAP]
"""
import glob
import json
import sys
from collections import defaultdict

d = sys.argv[1]
GAP = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0

per_fwd = defaultdict(lambda: {"idx_calls": 0, "idx_bad": 0, "idx_elems": 0,
                               "fm_calls": 0, "fm_bad": 0, "fm_rows": 0})
rank_fwd_counts = {}

for path in sorted(glob.glob(f"{d}/rank*.jsonl")):
    rank = path.split("/")[-1].split("_")[0]
    recs = [json.loads(l) for l in open(path) if l.strip()]
    recs.sort(key=lambda r: r["ts"])
    fwd = -1
    last_ts = None
    for r in recs:
        if last_ts is None or r["ts"] - last_ts > GAP:
            fwd += 1
        last_ts = r["ts"]
        s = per_fwd[fwd]
        if r.get("kernel") == "indexer":
            s["idx_calls"] += 1
            if r.get("n_diff_elems", 0):
                s["idx_bad"] += 1
                s["idx_elems"] += r["n_diff_elems"]
        elif r.get("kernel") == "flashmla":
            s["fm_calls"] += 1
            if r.get("rows_diff", 0) or r.get("lse_diff", 0):
                s["fm_bad"] += 1
                s["fm_rows"] += r.get("rows_diff", 0)
    rank_fwd_counts[rank] = fwd + 1

print("forwards per rank:", sorted(set(rank_fwd_counts.values())),
      f"({len(rank_fwd_counts)} ranks)")
print(f"{'fwd':>4} {'idx_calls':>9} {'idx_bad':>8} {'idx_elems':>10} "
      f"{'fm_calls':>9} {'fm_bad':>7} {'fm_rows':>8}")
for fwd in sorted(per_fwd):
    s = per_fwd[fwd]
    flag = "  <-- DISAGREE" if s["idx_bad"] or s["fm_bad"] else ""
    print(f"{fwd:>4} {s['idx_calls']:>9} {s['idx_bad']:>8} {s['idx_elems']:>10} "
          f"{s['fm_calls']:>9} {s['fm_bad']:>7} {s['fm_rows']:>8}{flag}")
