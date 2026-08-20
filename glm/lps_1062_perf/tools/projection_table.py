#!/usr/bin/env python3
"""projection_table.py — the all-rungs memory projection, computed ONCE from
the rung-1 census (LPS-1062 §8c, conway 2026-08-20).

Every later gate inherits its band from this one table. Computing each rung's
band separately is how per-rung arithmetic drift happens: each gate ends up
resting on slightly different assumptions and nobody notices the
disagreement.

Two rules are baked in and worth stating plainly, because getting either
backwards moves the answer by more than the margin:

  GLUE ALONE is the net-new resident memory. Under full recompute the layer
  inputs (S_ckpt, 0.19 GiB per set) are ALREADY stored — that is what the
  measured base peak includes. Adding glue+input double-counts, which is
  worth 13.3 GiB on rank 0 and 7.6 GiB on rank 8: enough to flip a verdict
  that lands near the line. The glue+input figure is reported alongside as an
  explicitly conservative upper bound, never as the primary.

  THE RANKS ARE NOT SYMMETRIC and are never averaged. Rank 0 is the first
  pipeline stage: 3 dense + 35 MoE layers holding 2 microbatches, so 70 MoE
  sets. Rank 8 is the last stage: 40 MoE layers holding 1, so 40 MoE sets,
  plus the loss and output logits. They bind on different things — rank 0 on
  set count, rank 8 on total peak.

Usage:
  projection_table.py --base0 151.8 --base8 162.5 --glue 0.72 --attn-proj 0.20
"""

from __future__ import annotations

import argparse

CAP_GIB = 267.7          # measured by CUDA on this box
COLD_BURST_GIB = 20.0    # cold-allocator burst the ceiling must leave room for
EFFECTIVE_CEILING = CAP_GIB - COLD_BURST_GIB   # ~247.7
FAIL_MARGIN = 20.0       # "within ~20 GiB of the ceiling" is a gate-#1 FAIL
FAIL_LINE = EFFECTIVE_CEILING - FAIL_MARGIN    # ~227.7
PREFETCH_RESERVE = 5.0   # double-buffering one layer ahead
S_CKPT = 0.19            # GiB/set layer inputs, already in the base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base0", type=float, required=True,
                    help="rank-0 measured post-plateau peak, GiB (nvidia-smi reserved)")
    ap.add_argument("--base8", type=float, required=True,
                    help="rank-8 measured post-plateau peak, GiB")
    ap.add_argument("--glue", type=float, required=True,
                    help="MEASURED glue, GiB per MoE layer per microbatch")
    ap.add_argument("--attn-proj", type=float, default=0.20,
                    help="measured attn_proj, GiB per MoE layer per microbatch")
    ap.add_argument("--sets0", type=int, default=70, help="rank-0 MoE sets")
    ap.add_argument("--sets8", type=int, default=40, help="rank-8 MoE sets")
    args = ap.parse_args()

    ranks = [("rank 0 (first stage)", args.base0, args.sets0),
             ("rank 8 (last stage)", args.base8, args.sets8)]

    # per-set resident delta by arm. moe_act is offloaded in both arms, so it
    # never appears; core attention stays recomputed in both.
    arms = [
        ("3a  selective + moe_act offload", args.glue + args.attn_proj),
        ("3b  3a + attn_proj offload", args.glue),
    ]

    print(f"# All-rungs projection — one pass from the census\n")
    print(f"card {CAP_GIB} GiB | cold burst {COLD_BURST_GIB} | effective ceiling "
          f"{EFFECTIVE_CEILING:.1f} | FAIL at >= {FAIL_LINE:.1f}")
    print(f"measured glue {args.glue:.3f} GiB/MoE-layer-mb | attn_proj "
          f"{args.attn_proj:.3f} | prefetch reserve {PREFETCH_RESERVE:.0f} GiB\n")

    print("| arm | rank | base | + sets x delta | + prefetch | PROJECTED | headroom | verdict | "
          "conservative (glue+input) |")
    print("|---|---|---|---|---|---|---|---|---|")

    worst = {}
    for arm, per_set in arms:
        for name, base, sets in ranks:
            delta = sets * per_set
            proj = base + delta + PREFETCH_RESERVE
            cons = proj + sets * S_CKPT
            head = EFFECTIVE_CEILING - proj
            verdict = "FAIL" if proj >= FAIL_LINE else "clears"
            worst.setdefault(arm, []).append((name, proj, verdict))
            print(f"| {arm} | {name} | {base:.1f} | {delta:.1f} | "
                  f"{PREFETCH_RESERVE:.0f} | **{proj:.1f}** | {head:.1f} | {verdict} | {cons:.1f} |")

    print("\n## Verdict")
    for arm, rows in worst.items():
        tight = max(rows, key=lambda r: r[1])
        fails = [r for r in rows if r[2] == "FAIL"]
        gap = tight[1] - FAIL_LINE
        if fails:
            print(f"- **{arm}: GATE-#1 FAIL** — {tight[0]} projects {tight[1]:.1f} GiB, "
                  f"{gap:+.1f} GiB past the {FAIL_LINE:.1f} fail line "
                  f"({EFFECTIVE_CEILING - tight[1]:.1f} GiB from the effective ceiling).")
        else:
            print(f"- {arm}: clears. Tighter rank is {tight[0]} at {tight[1]:.1f} GiB, "
                  f"{FAIL_LINE - tight[1]:.1f} GiB below the fail line.")
        if tight[1] >= EFFECTIVE_CEILING:
            print(f"    ...and {tight[0]} is ABOVE the effective ceiling "
                  f"({EFFECTIVE_CEILING:.1f} GiB): this arm is predicted to OOM, not merely to run tight.")

    print("\nGlue-alone is the primary projection; the conservative column adds "
          f"{S_CKPT} GiB/set of layer inputs that the base ALREADY contains, and is an "
          "upper bound only.")


if __name__ == "__main__":
    main()
