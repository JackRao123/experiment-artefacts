#!/usr/bin/env python3
"""Compare two train_replay arms (identical fresh boots, same seed/data/recipe).

Usage: python3 compare_arms.py trainF_arm0.jsonl trainF_arm1.jsonl

Reports per-step: mean_nll of each arm, delta, plus per-datum divergence stats
(max |nll0 - nll1| per step, count of datums moving > 0.05), quantifying how
fast two "identical" training runs diverge and whether any step shows a
bump-magnitude (>0.3) excursion in one arm only.
"""
import json
import sys


def load(path):
    steps = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("kind") == "train_step":
            steps[r["step"]] = r
    return steps


def main(p0, p1):
    a, b = load(p0), load(p1)
    common = sorted(set(a) & set(b))
    print(f"{'step':>4} {'arm0':>7} {'arm1':>7} {'d_mean':>8} {'max_datum_d':>11} "
          f"{'n>0.05':>6} {'n>0.3':>5}")
    for s in common:
        d0 = {d["idx"]: d["nll"] for d in a[s]["datums"]}
        d1 = {d["idx"]: d["nll"] for d in b[s]["datums"]}
        deltas = [abs(d0[i] - d1[i]) for i in d0]
        dm = a[s]["mean_nll"] - b[s]["mean_nll"]
        flag = "  <-- DIVERGED" if abs(dm) > 0.05 or max(deltas) > 0.3 else ""
        print(f"{s:>4} {a[s]['mean_nll']:>7.4f} {b[s]['mean_nll']:>7.4f} {dm:>+8.4f} "
              f"{max(deltas):>11.4f} {sum(1 for x in deltas if x > 0.05):>6} "
              f"{sum(1 for x in deltas if x > 0.3):>5}{flag}")
    g0 = [a[s].get("optim_metrics") or {} for s in common]
    print("\ngrad_norm pairs (step, arm0, arm1):")
    for s in common[:0]:
        pass


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
