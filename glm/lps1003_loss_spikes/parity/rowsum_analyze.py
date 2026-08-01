#!/usr/bin/env python3
"""LPS-1003 instrument v2 analysis: DSA top-k selection digests.

Input: rowsum dir (rank<R>_pid<P>.jsonl, one line per _indexer_top_k_one_chunk
call: ts, call#, rows, sk, k, total, sha1).

Given event and healed ts ranges, aligns call sequences positionally within
each range (packing is deterministic; launch order is 4-periodic per rep) and
reports:
  - healed-vs-healed digest stability (the control: ties/nondeterminism level)
  - event-vs-healed digest mismatches, grouped by call geometry (rows, sk)

Usage: rowsum_analyze.py DIR --event-ts A B --healed-ts C D [--reps N]
The healed range is split into reps by call-count = event call-count.
"""
import argparse
import glob
import json
import os
import re
from collections import Counter


def load(fp):
    calls = []
    with open(fp) as fh:
        for line in fh:
            try:
                calls.append(json.loads(line))
            except Exception:
                break
    return calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--event-ts", nargs=2, type=float, required=True)
    ap.add_argument("--healed-ts", nargs=2, type=float, required=True)
    args = ap.parse_args()

    for fp in sorted(glob.glob(os.path.join(args.dir, "rank*_pid*.jsonl"))):
        rank = re.search(r"rank(\w+)_pid", os.path.basename(fp)).group(1)
        calls = load(fp)
        ev = [c for c in calls if args.event_ts[0] <= c["ts"] <= args.event_ts[1]]
        he = [c for c in calls if args.healed_ts[0] <= c["ts"] <= args.healed_ts[1]]
        if not ev or len(he) < len(ev) * 2:
            continue
        n = len(ev)
        heald_reps = [he[i * n:(i + 1) * n] for i in range(len(he) // n)]
        # control: healed rep pairs
        ctrl_mismatch = Counter()
        ctrl_total = 0
        for a, b in zip(heald_reps, heald_reps[1:]):
            for ca, cb in zip(a, b):
                if (ca["rows"], ca["sk"]) != (cb["rows"], cb["sk"]):
                    continue  # misalignment; skip
                ctrl_total += 1
                if ca["sha1"] != cb["sha1"]:
                    ctrl_mismatch[(ca["rows"], ca["sk"])] += 1
        # event vs first healed rep
        ev_mismatch = Counter()
        ev_total = 0
        mis_detail = []
        ref = heald_reps[0]
        for ce, cr in zip(ev, ref):
            if (ce["rows"], ce["sk"]) != (cr["rows"], cr["sk"]):
                continue
            ev_total += 1
            if ce["sha1"] != cr["sha1"]:
                ev_mismatch[(ce["rows"], ce["sk"])] += 1
                if len(mis_detail) < 6:
                    mis_detail.append((ce["call"], ce["rows"], ce["sk"],
                                       ce["total"] - cr["total"]))
        cm = sum(ctrl_mismatch.values())
        em = sum(ev_mismatch.values())
        print(f"rank{rank}: event calls={n} | control mismatch {cm}/{ctrl_total} "
              f"({cm/max(ctrl_total,1):.2%}) | event mismatch {em}/{ev_total} "
              f"({em/max(ev_total,1):.2%})")
        if em:
            top = ev_mismatch.most_common(5)
            print(f"    event mismatch geometries: {top}")
            print(f"    examples (call, rows, sk, d_total): {mis_detail}")
        if cm:
            print(f"    CONTROL NONZERO — digests unstable, interpret with care: "
                  f"{ctrl_mismatch.most_common(3)}")


if __name__ == "__main__":
    main()
