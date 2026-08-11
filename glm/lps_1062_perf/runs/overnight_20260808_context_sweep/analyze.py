#!/usr/bin/env python3
"""Analyze GLM-5.2 EP8+CP8 memory sweep results: y = mx + b fit.

Reads the sweep JSONL (one row per context length L, peak MiB per GPU).
Fits peak-max-vs-L by least squares, prints the equation, R^2, per-point
table, and the predicted OOM crossing vs the B300's 275040 MiB capacity.
"""

import json
import sys


def fit(xs, ys):
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    m = (n * sxy - sx * sy) / denom
    b = (sy - m * sx) / n
    ybar = sy / n
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    ss_res = sum((y - (m * x + b)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return m, b, r2


def main(path, capacity_mib=275040):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    idle = next((r for r in rows if r["L"] == 0), None)
    pts = [r for r in rows if r["L"] > 0 and r["ok"]]
    fails = [r for r in rows if r["L"] > 0 and not r["ok"]]

    if idle:
        print(f"idle baseline (weights+ctx resident): max {idle['peak_mib_max']} MiB "
              f"({idle['peak_mib_max']/1024:.1f} GiB), min {idle['peak_mib_min']} MiB")
    print(f"\n{'L (tokens)':>10} {'peak max MiB':>13} {'peak max GiB':>13} "
          f"{'spread MiB':>11} {'wall s':>8}")
    for r in pts:
        spread = r["peak_mib_max"] - r["peak_mib_min"]
        print(f"{r['L']:>10} {r['peak_mib_max']:>13} {r['peak_mib_max']/1024:>13.1f} "
              f"{spread:>11} {r.get('wall_s', 0):>8.1f}")

    xs = [r["L"] for r in pts]
    ys = [r["peak_mib_max"] for r in pts]
    if len(xs) >= 2:
        m, b, r2 = fit(xs, ys)
        print(f"\n[naive fit, all n={len(xs)} points — BIASED by allocator saturation]")
        print(f"  y = {m:.4f}*x + {b:.0f}   (MiB), R^2 = {r2:.4f}")

    # Unsaturated regime only: peak reserved tracks true demand only while the
    # caching allocator is below the device ceiling. Drop points within 2% of
    # capacity — they are clamped, not measured.
    lin = [r for r in pts if r["peak_mib_max"] < 0.98 * capacity_mib]
    xs = [r["L"] for r in lin]
    ys = [r["peak_mib_max"] for r in lin]
    if len(xs) >= 2:
        m, b, r2 = fit(xs, ys)
        print(f"\n[linear-regime fit, n={len(xs)} unsaturated points]")
        print(f"y = m*x + b   (y = peak max GPU memory MiB, x = context length tokens)")
        print(f"  m = {m:.4f} MiB/token  ({m*1024:.0f} B/token)")
        print(f"  b = {b:,.0f} MiB  ({b/1024:.1f} GiB)")
        print(f"  R^2 = {r2:.4f}")
        if idle:
            print(f"  (intercept vs measured idle baseline {idle['peak_mib_max']:,} MiB: "
                  f"{abs(b-idle['peak_mib_max'])/idle['peak_mib_max']*100:.1f}% off)")
        if m > 0:
            x_cap = (capacity_mib - b) / m
            print(f"  naive capacity crossing: x = {x_cap:,.0f} tokens "
                  f"(soft — allocator reuse pushes real completions past this)")
    if fails:
        print(f"\nfirst failure at L = {fails[0]['L']} tokens")
        err = fails[0].get("error", "")
        if err:
            print(f"  error: {err[:300]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results.jsonl")
