#!/usr/bin/env python3
"""Analyze run-2 (golden EP16+CP16, 2-node) sweep results.

Joins results2_windows.jsonl (per-L [t0,t1] windows) with per-node nvidia-smi
poller logs (epoch-prefixed per-GPU MiB rows), computes per-GPU and
cluster-wide peaks per window, and fits y = mx + b on the unsaturated regime.

Usage: analyze2.py <windows.jsonl> <mem_log_1> [mem_log_2 ...]
"""

import json
import sys


def fit(xs, ys):
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    m = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    b = (sy - m * sx) / n
    ybar = sy / n
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    ss_res = sum((y - (m * x + b)) ** 2 for x, y in zip(xs, ys))
    return m, b, (1 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def load_samples(paths):
    """-> list of (epoch, [mib per gpu]) across all nodes."""
    out = []
    for p in paths:
        for line in open(p):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    out.append((float(parts[0]), [int(x) for x in parts[1:]]))
                except ValueError:
                    pass
    return out


def peak_in_window(samples, t0, t1, pad=4.0):
    n_gpu = max((len(v) for _, v in samples), default=0)
    peaks = [0] * n_gpu
    n = 0
    for ts, vals in samples:
        if t0 - pad <= ts <= t1 + pad:
            n += 1
            for i, v in enumerate(vals):
                if v > peaks[i]:
                    peaks[i] = v
    return peaks, n


def main():
    windows_path = sys.argv[1]
    mem_paths = sys.argv[2:]
    capacity = 275040

    windows = [json.loads(l) for l in open(windows_path) if l.strip()]
    samples = load_samples(mem_paths)
    print(f"loaded {len(samples)} memory samples from {len(mem_paths)} node log(s)")

    idle_row = next((r for r in windows if r["L"] == 0), None)
    if idle_row:
        peaks, n = peak_in_window(samples, idle_row["t0"] - 5, idle_row["t0"] + 5, pad=0)
        print(f"idle baseline: max {max(peaks)} MiB ({max(peaks)/1024:.1f} GiB), "
              f"min {min(peaks)} MiB across {len(peaks)} GPUs ({n} samples)")

    print(f"\n{'L':>9} {'peak max MiB':>13} {'GiB':>7} {'spread MiB':>11} {'wall s':>8} {'samples':>8}")
    pts = []
    fail = None
    for r in windows:
        if r["L"] == 0:
            continue
        peaks, n = peak_in_window(samples, r["t0"], r["t1"])
        row = dict(r)
        row["peak_max"] = max(peaks)
        row["peak_min"] = min(peaks)
        row["nsamples"] = n
        if r["ok"]:
            pts.append(row)
            print(f"{r['L']:>9} {max(peaks):>13} {max(peaks)/1024:>7.1f} "
                  f"{max(peaks)-min(peaks):>11} {r['wall_s']:>8.1f} {n:>8}")
        else:
            fail = row
            print(f"{r['L']:>9} {max(peaks):>13} {max(peaks)/1024:>7.1f} "
                  f"{max(peaks)-min(peaks):>11} {r['wall_s']:>8.1f} {n:>8}   FAILED")

    xs = [r["L"] for r in pts]
    ys = [r["peak_max"] for r in pts]
    if len(xs) >= 2:
        m, b, r2 = fit(xs, ys)
        print(f"\n[naive fit, all n={len(pts)} points]  y = {m:.4f}*x + {b:.0f} MiB, R^2 = {r2:.4f}")
    lin = [r for r in pts if r["peak_max"] < 0.98 * capacity]
    if len(lin) >= 2:
        xs = [r["L"] for r in lin]
        ys = [r["peak_max"] for r in lin]
        m, b, r2 = fit(xs, ys)
        print(f"[linear-regime fit, n={len(lin)} unsaturated]")
        print(f"  y = {m:.4f}*x + {b:,.0f} MiB   (m = {m*1024:.0f} B/token/GPU, b = {b/1024:.1f} GiB)")
        print(f"  R^2 = {r2:.4f}")
        if m > 0:
            print(f"  naive capacity crossing: x = {(capacity-b)/m:,.0f} tokens")
    if fail:
        print(f"\nfirst failure at L = {fail['L']}: {fail.get('error', '')[:300]}")


if __name__ == "__main__":
    main()
