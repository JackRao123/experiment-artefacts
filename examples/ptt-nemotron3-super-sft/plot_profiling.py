#!/usr/bin/env python3
"""Plot Nemotron-3-Super peak-memory vs sequence length (profiling.md data).

Two single-node 8x B200 layouts, selective recompute (core_attn,moe,moe_act,
layernorm), LoRA rank 16, micro-batch 1:
  - TP8/CP1
  - TP4/CP2  (CP sequence-sharded via the experimental get_batch_on_this_cp_rank
              slice; memory only — CP grads aren't validated, see profiling.md)

Draws each series with a line of best fit, the slope (GiB per 1k tokens, 2 d.p.),
and R^2. Usage:  python plot_profiling.py [--out peak_mem.png]
"""

from __future__ import annotations

import argparse

import numpy as np

# (seq_len tokens, peak_alloc GiB/GPU)
TP8_CP1 = [
    (8192, 39.37),
    (16384, 48.96),
    (24576, 58.40),
    (32768, 67.59),
    (40960, 76.75),
]

TP4_CP2 = [
    (8192, 39.34),
    (16384, 47.34),
    (24576, 55.07),
    (32768, 62.45),
    (40960, 69.85),
    (65536, 92.46),
    (98304, 122.98),
    (99328, 124.00),
    (100352, 125.03),
    (101376, 126.04),
    (102400, 126.82),
    (103424, 127.41),
    (104448, 128.11),
    (105472, 128.84),
    (106496, 129.67),
    (107520, 130.57),
    (108544, 131.55),
    (109568, 132.67),
    (110592, 133.82),
    (111616, 134.97),
    (112640, 136.13),
    (113664, 137.27),
    (114688, 138.38),
    (115712, 139.47),
    (116736, 140.50),
    (117760, 141.49),
    (118784, 142.48),
    (119808, 143.43),
    (120832, 144.44),
    (121856, 145.44),
    (122880, 146.41),
    (123904, 147.48),
    (124928, 148.50),
    (125952, 149.53),
    (126976, 150.66),
    (128000, 151.71),
    (129024, 152.80),
    (130048, 153.83),
    (131072, 154.85),
]

GPU_HBM_GIB = 178.0  # usable HBM per B200


def _fit(points):
    """Return (slope_per_token, intercept, r2) for a degree-1 fit."""
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="peak_mem.png")
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, pts, color in (
        ("TP8/CP1", TP8_CP1, "tab:blue"),
        ("TP4/CP2", TP4_CP2, "tab:orange"),
    ):
        x = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts], dtype=float)
        slope, intercept, r2 = _fit(pts)
        # slope per 1k tokens (GiB), 2 d.p.
        slope_k = slope * 1000.0
        ax.scatter(x / 1000.0, y, s=18, color=color)
        xfit = np.array([x.min(), x.max()])
        ax.plot(
            xfit / 1000.0,
            slope * xfit + intercept,
            color=color,
            label=f"{label}: {slope_k:.2f} GiB/1k tok, "
            f"intercept {intercept:.1f} GiB, R\u00b2={r2:.4f}",
        )

    ax.axhline(
        GPU_HBM_GIB, ls="--", color="red", lw=1, label=f"{GPU_HBM_GIB:.0f} GiB (B200)"
    )
    ax.set_xlabel("sequence length (k tokens)")
    ax.set_ylabel("peak_alloc per GPU (GiB)")
    ax.set_title(
        "Nemotron-3-Super peak memory vs seq len (8x B200, selective recompute)"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
