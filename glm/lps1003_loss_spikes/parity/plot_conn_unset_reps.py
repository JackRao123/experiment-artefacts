#!/usr/bin/env python3
"""Loss curves for the three devbox conn-unset boots (LPS-1003).

Each point = one probe rep (a /forward of the fixed 32-doc batch-0), y = mean
per-datum NLL, x = minutes since that boot's first probe. Only rep0 of each
boot fires (docs {4,5,6} destroyed); every later rep sits on the clean floor.

Usage:  python3 plot_conn_unset_reps.py [-o conn_unset_reps.png]
Reads runs/ relative to this script; writes the PNG next to it.
"""
import argparse
import glob
import gzip
import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")

BOOTS = [  # (label, glob of rep dumps), palette slots 1-3 in fixed order
    ("boot 1: prodenv + tracer", "d2_prodenv/d2_penv_*rep*.json.gz", "#2a78d6"),
    ("boot 2: prodenv, no tracer", "attrib_archive/prodenv_0731_073248/*rep*.json.gz", "#eb6834"),
    ("boot 3: conn-only arm", "attrib_conn/*rep*.json.gz", "#1baf7a"),
]
CLEAN_LO, CLEAN_HI = 0.760, 0.771  # stock devbox clean band (NOTEBOOK.md)


def load_boot(pattern):
    reps = []
    for path in glob.glob(os.path.join(RUNS, pattern)):
        d = json.load(gzip.open(path))
        mean = sum(d["nlls"]) / len(d["nlls"])
        reps.append((d["ts_start"], mean, sorted(map(int, d.get("destroyed") or {}))))
    reps.sort()
    t0 = reps[0][0]
    return [((ts - t0) / 60.0, mean, destroyed) for ts, mean, destroyed in reps]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default=os.path.join(HERE, "conn_unset_reps.png"))
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    ax.axhspan(CLEAN_LO, CLEAN_HI, color="#0b0b0b", alpha=0.06, lw=0,
               label="clean band (stock devbox)")

    for i, (label, pattern, color) in enumerate(BOOTS):
        pts = load_boot(pattern)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=5, label=label)
        print(f"\n{label}")
        for x, y, destroyed in pts:
            print(f"  +{x:5.1f} min  mean_nll={y:.4f}  destroyed={destroyed or '-'}")
            if destroyed:
                ax.annotate(f"{label.split(':')[0]}: {y:.2f}, destroyed {destroyed}",
                            (x, y), xytext=(10, 6 - 12 * i),
                            textcoords="offset points",
                            fontsize=8, color="#52514e")

    ax.set_xlabel("minutes since first probe of the boot")
    ax.set_ylabel("batch-0 mean per-datum NLL (nats)")
    ax.set_title("Devbox, CUDA_DEVICE_MAX_CONNECTIONS unset: only rep0 fires")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#0b0b0b", alpha=0.08)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
