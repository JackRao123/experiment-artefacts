#!/usr/bin/env python3
"""Aggregate a BT_DSA_AUDIT_DIR of per-rank JSONLs into a verdict summary.

Usage: python3 audit_summary.py <audit_dir>
"""

from __future__ import annotations

import collections
import glob
import json
import sys


def main(d: str) -> None:
    files = sorted(glob.glob(f"{d}/rank*_pid*.jsonl"))
    if not files:
        print(f"no rank jsonls under {d}")
        return
    t = collections.Counter()
    k_set, sk_max, rows_min, rows_max = set(), 0, 1 << 62, 0
    tfsc = collections.Counter()
    exits, ctrl = {}, {}
    under_ex, dup_ex, oor_ex = [], [], []
    errors = collections.Counter()
    staged_any = False

    for f in files:
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = r.get("kind")
            if kind == "topk":
                t["calls"] += 1
                for src, dst in (
                    ("rows", "rows"), ("short_rows", "short_rows"), ("radix_rows", "radix_rows"),
                    ("unwritten_slots", "unwritten"), ("unwritten_rows", "unwritten_rows"),
                    ("fixed_slots", "fixed"), ("dup_slots", "dup"), ("dup_rows", "dup_rows"),
                    ("sel_oor", "oor"), ("sel_neg", "neg"), ("inv_finite", "inv_finite"),
                ):
                    t[dst] += r.get(src, 0) or 0
                if r.get("staged"):
                    staged_any = True
                    t["staged_calls"] += 1
                if r.get("unwritten_slots"):
                    t["unwritten_calls"] += 1
                    if len(under_ex) < 12:
                        under_ex.append({"call": r["call"], "rows": r["rows"], "k": r["k"],
                                         "n": r["unwritten_slots"], "ex": r.get("unwritten_detail", [{}])[0]})
                if r.get("dup_slots"):
                    t["dup_calls"] += 1
                    if len(dup_ex) < 8:
                        dup_ex.append({"call": r["call"], "rows": r["rows"],
                                       "n": r["dup_slots"], "rows_dup": r.get("dup_rows")})
                if r.get("sel_oor"):
                    t["oor_calls"] += 1
                    if len(oor_ex) < 8:
                        oor_ex.append({"call": r["call"], "ex": r.get("oor_detail", [{}])[0]})
                if r.get("ctrl_frac") is not None:
                    ctrl[(r["rows"], r["k"])] = r["ctrl_frac"]
                k_set.add(r.get("k"))
                sk_max = max(sk_max, r.get("sk", 0) or 0)
                rows_min = min(rows_min, r.get("rows", 1 << 62))
                rows_max = max(rows_max, r.get("rows", 0))
                for ek in ("pre_audit_error", "post_audit_error", "stage_error"):
                    if r.get(ek):
                        errors[f"{ek}: {r[ek]}"] += 1
                if r.get("ctr"):
                    exits[f] = r["ctr"]
            elif kind == "tfsc":
                tfsc[(r.get("starts"), r.get("ends"), r.get("score_seq_lens"),
                      r.get("brks") is not None)] += 1
            elif kind == "exit":
                exits[f] = {k: v for k, v in r.items() if k not in ("kind", "ts", "ctrl")}
                for kk, v in (r.get("ctrl") or {}).items():
                    a, b = kk.split("x")
                    ctrl[(int(a), int(b))] = v

    print(f"ranks={len(files)} topk_calls={t['calls']} rows_total={t['rows']} "
          f"rows_per_call={rows_min if rows_min < (1<<62) else '-'}..{rows_max} "
          f"k={sorted(x for x in k_set if x is not None)} sk_max={sk_max}")
    print(f"output branch mix: rows with window<=k (trivial full-write)={t['short_rows']}  "
          f"window>k (radix, the suspect)={t['radix_rows']}")
    print(f"prod-shaped launches (rows>148 -> large_occupancy variant): "
          f"{'YES' if rows_max > 148 else 'NO — WRONG KERNEL VARIANT'}")

    print("\n--- E2/E4 under-write detection ---")
    if not staged_any:
        print("staging OFF (audit-only run) — no unwritten-slot evidence either way")
    else:
        bad_ctrl = {k: v for k, v in ctrl.items() if v is None or v <= 0.99}
        print(f"staged calls={t['staged_calls']}  shapes controlled={len(ctrl)}  "
              f"shapes FAILING control={len(bad_ctrl)}")
        if bad_ctrl:
            print("  !! staged block not handed back for these shapes; their counts are INCONCLUSIVE:")
            for k, v in list(bad_ctrl.items())[:10]:
                print(f"     rows={k[0]} k={k[1]} sentinel_frac={v}")
        print(f"UNWRITTEN SLOTS = {t['unwritten']} in {t['unwritten_rows']} rows "
              f"across {t['unwritten_calls']} calls")
        if t["unwritten"]:
            print("  -> UNDER-WRITE CAUGHT. examples:")
            for e in under_ex:
                print("    ", e)
        elif not bad_ctrl:
            print("  -> no unwritten slots, and staging control passed on every shape: "
                  "the radix branch full-wrote every row at these shapes")
        if t["fixed"]:
            print(f"fix arm rewrote {t['fixed']} slots to -1")

    print("\n--- E1 output-index audit ---")
    print(f"duplicates within a row: {t['dup']} slots in {t['dup_rows']} rows "
          f"({t['dup_calls']} calls); negative sentinels={t['neg']}")
    for e in dup_ex:
        print("  DUP", e)
    print(f"selected out of window: {t['oor']} ({t['oor_calls']} calls)")
    for e in oor_ex:
        print("  OOR", e)

    print("\n--- E0 branch audit ---")
    if tfsc:
        print("tfsc kwargs (starts, ends, score_seq_lens, bottom_right_set) -> count:")
        for kk, v in sorted(tfsc.items(), key=lambda x: -x[1]):
            print("  ", kk, v)
    agg = collections.Counter()
    for ctr in exits.values():
        for kk, v in ctr.items():
            if isinstance(v, (int, float)):
                agg[kk] += v
    if agg:
        print(f"counters ({len(exits)} ranks): cudnn_fw={agg['cudnn_fw']} "
              f"torch_scores={agg['torch_scores']} tfsc={agg['tfsc']} topk={agg['topk']}")
        print(f"scores regression watch: sampled {agg['scores_scanned']} calls, "
              f"invalid-region-finite calls={agg['inv_finite_calls']} "
              f"(expect 0 — region is -inf by wheel pre-fill)")
        if agg.get("sk_ge_sentinel_calls"):
            print(f"  !! {agg['sk_ge_sentinel_calls']} calls had sk >= sentinel: raise BT_DSA_STAGE_VAL")
    if errors:
        print("\nAUDIT ERRORS:")
        for msg, c in errors.most_common(10):
            print(f"  {c}x {msg}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
