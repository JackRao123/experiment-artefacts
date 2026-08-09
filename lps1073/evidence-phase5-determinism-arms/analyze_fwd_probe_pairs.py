#!/usr/bin/env python3
"""Per-pair global-origin analysis for bt_fwd_probe dumps.

For every pair of the last N forward calls, find across ALL ranks the
earliest (by execution seq) module whose fingerprint differs between the
two calls. Reports, per pair: origin seq/module and the set of ranks that
diverge at that seq. If the origin is stable across pairs -> single hot
site; scattered -> distributed mechanism.

Usage: analyze_fwd_probe_pairs.py <probe_dir> [--n-calls 10]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter


def keyed(call):
    seen: dict = {}
    out = {}
    for e in call:
        k = (e["module"], seen.setdefault(e["module"], 0))
        seen[e["module"]] += 1
        out[k] = (e["seq"], e["cls"], tuple(t["sha"] for t in e["tensors"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_dir")
    ap.add_argument("--n-calls", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ranks = {}
    for path in glob.glob(os.path.join(args.probe_dir, "fwdprobe.rank*.json")):
        d = json.load(open(path))
        calls = d["calls"][-args.n_calls:]
        ranks[d["rank"]] = [keyed(c) for c in calls]

    n = args.n_calls
    origin_counter = Counter()
    results = []
    for i in range(n):
        for j in range(i + 1, n):
            best_seq = None
            best = []  # (rank, module, cls)
            for r, kcalls in sorted(ranks.items()):
                a, b = kcalls[i], kcalls[j]
                for k in sorted(a.keys(), key=lambda k: a[k][0]):
                    if k in b and a[k][2] != b[k][2]:
                        seq, cls, _ = a[k]
                        if best_seq is None or seq < best_seq:
                            best_seq, best = seq, [(r, k[0], cls)]
                        elif seq == best_seq:
                            best.append((r, k[0], cls))
                        break
            mods = sorted({(m.split(".decoder.")[-1], c) for _, m, c in best})
            rks = sorted({r for r, _, _ in best})
            origin_counter[tuple(mods)] += 1
            results.append({
                "pair": [i, j], "origin_seq": best_seq,
                "origin_modules": mods, "origin_ranks": rks,
            })
            print(f"pair {i}v{j}: seq {best_seq} {mods} ranks {rks}")

    print("\norigin histogram:")
    for mods, c in origin_counter.most_common():
        print(f"  {c:3d}x {list(mods)}")

    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)
        print("written:", args.out)
    return 0


if __name__ == "__main__":
    main()
