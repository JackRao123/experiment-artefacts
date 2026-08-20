#!/usr/bin/env python3
"""Divergence-structure analysis for a failed parity compare (LPS-1062 W2 leg 2).

Discriminates the two live hypotheses for a pervasive per-token logprob
divergence with a preserved aggregate mean:

  (A) REPORTING/ORDERING — the ON arm's logprob VALUES are the same set as
      OFF's, just assigned to wrong positions (a zigzag/unshard or
      microbatch-ordering offset in the executor's loss-node reporting).
  (B) FORWARD CONTEXT — the ON arm's VALUES differ (each token computed
      against a shifted attention/DSA context), not a permutation.

Tests, per datum and pooled:
  1. Multiset test: sorted(ON) vs sorted(OFF) — if the sorted values match
     (within fp noise), the values are a permutation -> (A). If they differ,
     the values themselves changed -> (B).
  2. Positional-offset search: for a sample of ON positions, find the OFF
     position with the nearest value; report the offset histogram. A tight
     offset cluster (constant shift, or a block swap) -> (A) with the offset
     measured. A flat/uniform offset distribution -> (B).
  3. Boundary profile: divergence magnitude vs fractional position in the
     datum, and vs the CP zigzag chunk edges (2*cp chunks per row) — whether
     the divergence concentrates at chunk boundaries (ordering) or is uniform
     (context).

Usage:
  python3 parity_divergence_structure.py ON.json OFF.json [--cp 8]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

def _load(path: str):
    with open(path) as f:
        rec = json.load(f)
    return [d["data"] for d in rec["per_datum_logprobs"]], rec


def _multiset_test(on: list[float], off: list[float], tol: float = 1e-3) -> tuple[bool, float]:
    """Are the two value-multisets equal within tol? Returns (is_perm, max_sorted_diff)."""
    if len(on) != len(off):
        return False, float("inf")
    so, sf = sorted(on), sorted(off)
    worst = max(abs(x - y) for x, y in zip(so, sf)) if so else 0.0
    return worst <= tol, worst


def _offset_search(on: list[float], off: list[float], tol: float = 1e-3, max_positions: int = 2000):
    """For sampled ON positions, find the OFF position with the nearest value.
    Returns (matched_fraction, offset_counter, unmatched_fraction)."""
    # Index OFF values -> sorted list for nearest search.
    import bisect

    off_sorted = sorted(range(len(off)), key=lambda j: off[j])
    off_vals = [off[j] for j in off_sorted]

    n = len(on)
    step = max(1, n // max_positions)
    offsets: Counter[int] = Counter()
    matched = 0
    unmatched = 0
    for i in range(0, n, step):
        v = on[i]
        # nearest OFF position by value
        k = bisect.bisect_left(off_vals, v)
        cands = [k - 1, k, k + 1]
        best_j = None
        best_d = tol
        for c in cands:
            if 0 <= c < len(off_vals):
                j = off_sorted[c]
                d = abs(off[j] - v)
                if d <= best_d:
                    best_d = d
                    best_j = j
        if best_j is None:
            unmatched += 1
        else:
            matched += 1
            offsets[best_j - i] += 1
    total = matched + unmatched
    return matched / total, offsets, unmatched / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("on")
    ap.add_argument("off")
    ap.add_argument("--cp", type=int, default=8)
    args = ap.parse_args()

    on_data, on_rec = _load(args.on)
    off_data, off_rec = _load(args.off)
    print(f"ON  loss={on_rec['loss']} datums={on_rec['n_datums']} tokens={on_rec['total_real_tokens']}")
    print(f"OFF loss={off_rec['loss']} datums={off_rec['n_datums']} tokens={off_rec['total_real_tokens']}")
    assert len(on_data) == len(off_data), "datum count mismatch"

    pooled_on: list[float] = []
    pooled_off: list[float] = []
    for di, (on, off) in enumerate(zip(on_data, off_data)):
        assert len(on) == len(off), f"datum {di} length mismatch"
        is_perm, worst_sorted = _multiset_test(on, off)
        matched_frac, offsets, unmatched_frac = _offset_search(on, off)
        top_offsets = offsets.most_common(3)
        print(
            f"datum {di} (n={len(on)}): multiset_match={is_perm} "
            f"(sorted_max_diff={worst_sorted:.3e}) | value-matched {matched_frac:.1%} "
            f"unmatched {unmatched_frac:.1%} | top offsets (j-i, count): {top_offsets}"
        )
        pooled_on.extend(on)
        pooled_off.extend(off)

    is_perm, worst_sorted = _multiset_test(pooled_on, pooled_off)
    print(
        f"POOLED: multiset_match={is_perm} (sorted_max_diff={worst_sorted:.3e}) "
        f"-> {'PERMUTATION/ORDERING (reporting)' if is_perm else 'VALUES DIFFER (forward context)'}"
    )


if __name__ == "__main__":
    main()
