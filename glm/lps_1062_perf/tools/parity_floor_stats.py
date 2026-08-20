#!/usr/bin/env python3
"""parity_floor_stats.py — characterise a parity pair as a NOISE FLOOR
(LPS-1062 activation-placement ladder, conway 2026-08-20).

The parity driver's --compare prints a single max-abs figure and judges it
against absolute tolerances (1e-6 / 1e-3) inherited from a deterministic-path
era. On this path those tolerances always trip, and a lone max-abs is
dominated by one outlier token, so it cannot tell "the arm changed the maths"
from "this is Tuesday".

What a floor needs is a DISTRIBUTION. Two identical legs give the reference
distribution; an arm's leg is then judged by whether its distribution sits
inside it, not by whether some absolute bar was crossed.

Usage:
    parity_floor_stats.py A.json B.json [--label NAME]
    parity_floor_stats.py A.json B.json --floor floor.json   # judge vs a floor
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(p: str) -> dict:
    return json.loads(Path(p).read_text())


def flat_diffs(a: dict, b: dict) -> tuple[list[float], list[tuple[int, int, float]]]:
    def vals(entry):
        # The driver serialises each datum as {"data": [...], "shape": [...],
        # "dtype": "..."} rather than a bare list.
        return entry["data"] if isinstance(entry, dict) else entry

    la = [vals(e) for e in a["per_datum_logprobs"]]
    lb = [vals(e) for e in b["per_datum_logprobs"]]
    if len(la) != len(lb):
        raise SystemExit(f"datum count differs: {len(la)} vs {len(lb)}")
    diffs: list[float] = []
    worst: list[tuple[int, int, float]] = []
    for di, (xa, xb) in enumerate(zip(la, lb)):
        if len(xa) != len(xb):
            raise SystemExit(f"datum {di} length differs: {len(xa)} vs {len(xb)}")
        for ti, (va, vb) in enumerate(zip(xa, xb)):
            d = abs(va - vb)
            diffs.append(d)
            worst.append((di, ti, d))
    worst.sort(key=lambda r: -r[2])
    return diffs, worst[:5]


def pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, max(0, int(math.ceil(q / 100.0 * len(sorted_vals))) - 1))
    return sorted_vals[i]


def summarize(a: dict, b: dict) -> dict:
    diffs, worst = flat_diffs(a, b)
    s = sorted(diffs)
    n = len(s)
    la, lb = a["loss"], b["loss"]
    return {
        "n_tokens": n,
        "loss_a": la,
        "loss_b": lb,
        "loss_rel_diff": abs(la - lb) / max(abs(la), 1e-12),
        "max_abs": s[-1] if n else float("nan"),
        "p99_9": pct(s, 99.9),
        "p99": pct(s, 99.0),
        "p95": pct(s, 95.0),
        "median": pct(s, 50.0),
        "mean": sum(s) / n if n else float("nan"),
        "frac_gt_1e-3": sum(1 for d in s if d > 1e-3) / n if n else float("nan"),
        "frac_gt_1e-2": sum(1 for d in s if d > 1e-2) / n if n else float("nan"),
        "frac_gt_1": sum(1 for d in s if d > 1.0) / n if n else float("nan"),
        "n_exact_zero": sum(1 for d in s if d == 0.0),
        "worst": worst,
    }


def show(tag: str, r: dict) -> None:
    print(f"\n## {tag}")
    print(f"tokens compared      {r['n_tokens']:,}")
    print(f"loss                 {r['loss_a']:.12f} vs {r['loss_b']:.12f}")
    print(f"loss rel diff        {r['loss_rel_diff']:.3e}")
    print(f"per-token |diff|     max {r['max_abs']:.4e}  p99.9 {r['p99_9']:.4e}  "
          f"p99 {r['p99']:.4e}  p95 {r['p95']:.4e}  median {r['median']:.4e}  mean {r['mean']:.4e}")
    print(f"share > 1e-3         {r['frac_gt_1e-3'] * 100:.4f}%")
    print(f"share > 1e-2         {r['frac_gt_1e-2'] * 100:.4f}%")
    print(f"share > 1.0          {r['frac_gt_1'] * 100:.4f}%")
    print(f"bitwise-identical    {r['n_exact_zero']:,} tokens "
          f"({r['n_exact_zero'] / max(r['n_tokens'], 1) * 100:.2f}%)")
    print("worst tokens (datum, pos, |diff|): "
          + ", ".join(f"({d},{t},{v:.3f})" for d, t, v in r["worst"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label", default="pair")
    ap.add_argument("--floor", help="floor JSON written by an earlier run; judge this pair against it")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    r = summarize(load(args.a), load(args.b))
    show(args.label, r)

    if args.floor:
        f = json.loads(Path(args.floor).read_text())
        print(f"\n## judged against the floor in {args.floor}")
        # Noise-relative only. Every comparison is a ratio against the floor's
        # own statistic; no absolute bar appears anywhere.
        for key in ("max_abs", "p99_9", "p99", "median", "frac_gt_1e-3", "loss_rel_diff"):
            fv, rv = f.get(key), r.get(key)
            if not fv:
                print(f"  {key:16s} floor 0 — cannot form a ratio; arm {rv:.4e}")
                continue
            ratio = rv / fv
            verdict = "inside" if ratio <= 1.0 else f"{ratio:.2f}x the floor"
            print(f"  {key:16s} arm {rv:.4e}  floor {fv:.4e}  -> {verdict}")
        print("\nReading: 'inside' on the distribution statistics (p99.9, p99, median,")
        print("share>1e-3) is the signal. A single max_abs above the floor is weak")
        print("evidence on its own — it is one token out of a quarter-million.")

    if args.json_out:
        out = {k: v for k, v in r.items() if k != "worst"}
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f"\n[written] {args.json_out}")


if __name__ == "__main__":
    main()
