#!/usr/bin/env python3
"""Analyze probe_nll.py output JSONL: cross-repeat wobble per batch and per datum.

Usage: python3 analyze_probe.py out.jsonl [out2.jsonl ...]

Reports, for identical repeated submissions against frozen weights:
  - per-batch: mean_nll per repeat, max-min range, label
  - per-datum: NLL range across repeats, ranked; wobble by label class
  - flags datums whose NLL range exceeds --wobble-thresh (default 0.02)
"""
import json
import sys
from collections import defaultdict

WOBBLE = 0.02

def main(paths):
    batch_reps = defaultdict(dict)   # key -> repeat -> record
    datum_nll = defaultdict(dict)    # (batch,idx) -> repeat -> nll
    datum_meta = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if r.get("kind") == "status":
                continue
            key = tuple(r["key"]) if isinstance(r["key"], list) else r["key"]
            batch_reps[key][r["repeat"]] = r
            for d in r["datums"]:
                dk = (d["batch"], d["idx"])
                datum_nll[dk][r["repeat"]] = d["nll"]
                datum_meta[dk] = d

    print("=== per-op (batch or single datum) across repeats ===")
    print(f"{'key':>12} {'label':<13} {'n_rep':>5} {'mean_nll(rep0..)':<28} {'range':>8}")
    for key in sorted(batch_reps, key=str):
        reps = batch_reps[key]
        vals = [reps[i]["mean_nll"] for i in sorted(reps)]
        lbl = reps[min(reps)]["datums"][0]["label"]
        rng = max(vals) - min(vals)
        flag = "  <-- WOBBLE" if rng > WOBBLE else ""
        print(f"{str(key):>12} {lbl:<13} {len(vals):>5} "
              f"{' '.join(f'{v:.4f}' for v in vals):<28} {rng:>8.4f}{flag}")

    print("\n=== per-datum wobble ranking (top 25 by NLL range across repeats) ===")
    rows = []
    for dk, reps in datum_nll.items():
        if len(reps) < 2:
            continue
        vals = list(reps.values())
        rows.append((max(vals) - min(vals), dk, vals))
    rows.sort(reverse=True)
    print(f"{'batch:idx':>10} {'label':<13} {'range':>8} {'nll values'}")
    for rng, dk, vals in rows[:25]:
        m = datum_meta[dk]
        print(f"{dk[0]:>5}:{dk[1]:<4} {m['label']:<13} {rng:>8.4f} "
              f"{' '.join(f'{v:.4f}' for v in sorted(vals))}")

    print("\n=== wobble by label class ===")
    by_label = defaultdict(list)
    for rng, dk, _ in rows:
        by_label[datum_meta[dk]["label"]].append(rng)
    for lbl, rs in sorted(by_label.items()):
        rs.sort()
        n = len(rs)
        print(f"{lbl:<13} n={n:<4} median={rs[n//2]:.5f} p90={rs[int(n*0.9)]:.5f} "
              f"max={rs[-1]:.5f} n>{WOBBLE}={sum(1 for r in rs if r > WOBBLE)}")

    print("\n=== per-datum absolute NLL ranking (top 20, rep-mean) ===")
    arows = []
    for dk, reps in datum_nll.items():
        vals = list(reps.values())
        arows.append((sum(vals) / len(vals), dk))
    arows.sort(reverse=True)
    for mean, dk in arows[:20]:
        m = datum_meta[dk]
        print(f"{dk[0]:>5}:{dk[1]:<4} {m['label']:<13} mean_nll={mean:.4f} "
              f"n_sup={m['n_sup']} min_lp={m['min_lp']:.2f} n_below_10={m['n_below_10']}")


if __name__ == "__main__":
    main(sys.argv[1:])
