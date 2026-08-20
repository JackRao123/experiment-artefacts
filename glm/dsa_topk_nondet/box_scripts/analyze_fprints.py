#!/usr/bin/env python3
"""Align fingerprint records across probe repeats and find the FIRST
divergent (site, call-index) per rank.

Records: {"seq": N, "site": "dsa_topk"|"dsa_sparse_fwd"|"moe_router", <name>: sha16|None|"ERR:..."}
Alignment: file line order, split into --repeats equal chunks per rank.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/root/fprints")
ap.add_argument("--repeats", type=int, default=3)
args = ap.parse_args()

files = sorted(glob.glob(os.path.join(args.dir, "rank*.jsonl")))
if not files:
    sys.exit(f"no rank*.jsonl in {args.dir}")

overall_first = {}  # rank -> (call_idx, site, details)

for path in files:
    rank = os.path.basename(path).replace(".jsonl", "")
    lines = [json.loads(l) for l in open(path) if l.strip()]
    n = len(lines)
    print(f"\n=== {rank}: {n} records ===")
    if n % args.repeats != 0:
        print(f"  !! {n} not divisible by {args.repeats} — repeats have UNEQUAL "
              f"record counts; alignment by order is unsafe. Per-site counts:")
        print("  ", Counter(r["site"] for r in lines))
        continue
    per = n // args.repeats
    chunks = [lines[i * per:(i + 1) * per] for i in range(args.repeats)]

    # sanity: sites must align across repeats at every index
    site_mismatch = [i for i in range(per)
                     if len({c[i]["site"] for c in chunks}) > 1]
    if site_mismatch:
        print(f"  !! site sequence differs across repeats at indices "
              f"{site_mismatch[:10]} — alignment broken, results unreliable")
        continue

    # seq contiguity within each chunk (catches interleaving)
    for r, c in enumerate(chunks):
        seqs = [rec["seq"] for rec in c]
        if seqs != list(range(seqs[0], seqs[0] + per)):
            print(f"  !! repeat {r}: seq numbers not contiguous "
                  f"(start {seqs[0]}, {len(seqs)} recs) — possible interleaving")

    first_div = None
    div_by_site = defaultdict(list)  # site -> [call indices]
    site_call_counter = Counter()    # per-site call index for reporting
    site_idx_of = []                 # global idx -> per-site call idx
    for i in range(per):
        s = chunks[0][i]["site"]
        site_idx_of.append(site_call_counter[s])
        site_call_counter[s] += 1

    for i in range(per):
        recs = [c[i] for c in chunks]
        site = recs[0]["site"]
        keys = sorted(set().union(*[set(r) for r in recs]) - {"seq", "site"})
        diffs = {}
        for k in keys:
            vals = [r.get(k) for r in recs]
            if len(set(map(str, vals))) > 1:
                diffs[k] = vals
        if diffs:
            div_by_site[site].append(i)
            if first_div is None:
                first_div = (i, site, diffs)

    total_sites = Counter(chunks[0][i]["site"] for i in range(per))
    print(f"  per-repeat records: {per}  site counts: {dict(total_sites)}")
    if first_div is None:
        print("  ALL SITES BITWISE IDENTICAL across repeats.")
        continue
    i, site, diffs = first_div
    print(f"  FIRST DIVERGENCE: global call {i} = {site} call #{site_idx_of[i]}")
    for k, vals in diffs.items():
        print(f"     {k}: " + " | ".join(str(v) for v in vals))
    print("  divergent-call counts by site: "
          + ", ".join(f"{s}: {len(v)}/{total_sites[s]}" for s, v in sorted(div_by_site.items())))
    for s, v in sorted(div_by_site.items()):
        print(f"     {s}: first at global {v[0]} (site-call #{site_idx_of[v[0]]}), "
              f"indices {v[:12]}{'...' if len(v) > 12 else ''}")
    overall_first[rank] = (i, site)

print("\n=== SUMMARY (first divergent site per rank) ===")
for rank, (i, site) in sorted(overall_first.items()):
    print(f"  {rank}: {site} at global call {i}")
if overall_first:
    firsts = {site for _, site in overall_first.values()}
    print(f"  verdict: first-divergence site(s) = {firsts}")
