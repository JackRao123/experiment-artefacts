#!/usr/bin/env python3
"""LPS-1003 bisection: rule data and packing-position in or out, offline.

Two questions the docs assert but never verified with numbers:

  1. "Not a data issue" — do the batches/documents that spiked in prod look
     different, or score worse on frozen base weights, than quiet ones?
  2. Is there a deterministic POSITIONAL effect? Prod destruction lands on the
     tail documents of packed THD partitions. If tail position by itself raised
     NLL, that would be a deterministic packing/masking (script) bug and would
     show up on the devbox too, where the memory-window effect never fires.

Reconstructs the partition map by greedy sequential packing to max_seq_len
(validated: reproduces the documented batch-0 map [0-6][7-14][15-22][23-30][31])
and joins it against measured per-datum NLLs from the devbox probes.

usage: python3 data_forensics.py [--bundle probe_bundle.jsonl.gz]
                                 [--probe devbox_artifacts/phaseA_batches.jsonl]
                                 [--max-seq-len 262144]
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import statistics as st


def load_bundle(path):
    batches = collections.defaultdict(list)
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            batches[r["batch"]].append(r)
    for rows in batches.values():
        rows.sort(key=lambda r: r["idx"])
    return batches


def partitions(rows, max_seq_len):
    """Greedy sequential pack, mirroring the trainer's THD partitioning."""
    parts, cur, tot = [], [], 0
    for r in rows:
        L = len(r["ids"])
        if cur and tot + L > max_seq_len:
            parts.append(cur)
            cur, tot = [], 0
        cur.append(r["idx"])
        tot += L
    if cur:
        parts.append(cur)
    return parts


def mean_or_nan(v):
    return st.mean(v) if v else float("nan")


def welch_t(a, b):
    """Welch's t statistic; enough to say whether a split is meaningful."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    return (st.mean(a) - st.mean(b)) / se if se > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="probe_bundle.jsonl.gz")
    ap.add_argument("--probe", default="devbox_artifacts/phaseA_batches.jsonl")
    ap.add_argument("--max-seq-len", type=int, default=262144)
    args = ap.parse_args()

    bundle = load_bundle(args.bundle)

    # measured per-datum NLL on frozen base weights (devbox, no destruction)
    nll = collections.defaultdict(list)
    label = {}
    for line in open(args.probe):
        r = json.loads(line)
        if r.get("kind") != "batches":
            continue
        for d in r["datums"]:
            nll[(d["batch"], d["idx"])].append(d["nll"])
            label[(d["batch"], d["idx"])] = d["label"]

    print(f"bundle={args.bundle} batches={len(bundle)} probe={args.probe} "
          f"measured datums={len(nll)}\n")

    # ---------------- partition geometry + positional join ----------------
    pos_kind = {}
    for b, rows in bundle.items():
        parts = partitions(rows, args.max_seq_len)
        for p_i, part in enumerate(parts):
            for slot, idx in enumerate(part):
                pos_kind[(b, idx)] = {
                    "part": p_i,
                    "slot": slot,
                    "n_in_part": len(part),
                    "is_tail": slot == len(part) - 1,
                    "is_head": slot == 0,
                    "solo": len(part) == 1,
                }

    print("=== Q2: deterministic positional effect on devbox (frozen weights) ===")
    tail, mid = [], []
    for k, v in nll.items():
        pk = pos_kind.get(k)
        if pk is None:
            continue
        (tail if pk["is_tail"] else mid).append(mean_or_nan(v))
    print(f"tail-of-partition datums: n={len(tail)} mean_nll={mean_or_nan(tail):.4f}")
    print(f"mid/head datums:          n={len(mid)} mean_nll={mean_or_nan(mid):.4f}")
    if tail and mid:
        print(f"delta={mean_or_nan(tail)-mean_or_nan(mid):+.4f} nats  "
              f"Welch t={welch_t(tail, mid):+.2f}")
        print("  (prod destruction on tails is 5-11 nats. A deterministic packing/"
              "masking bug would show a large positive delta HERE too.)")

    # slot-position profile
    print("\nmean NLL by slot position within partition:")
    by_slot = collections.defaultdict(list)
    for k, v in nll.items():
        pk = pos_kind.get(k)
        if pk:
            by_slot[pk["slot"]].append(mean_or_nan(v))
    for s in sorted(by_slot):
        v = by_slot[s]
        print(f"  slot {s}: n={len(v):3d} mean={mean_or_nan(v):.4f}")

    # ---------------- Q1: data signal ----------------
    print("\n=== Q1: does content predict the prod label? (frozen-weight NLL) ===")
    by_label = collections.defaultdict(list)
    for k, v in nll.items():
        by_label[label[k]].append(mean_or_nan(v))
    for lab in sorted(by_label):
        v = by_label[lab]
        print(f"  {lab:14s} n={len(v):3d} mean={mean_or_nan(v):.4f} "
              f"median={st.median(v):.4f} max={max(v):.4f}")
    spike = [x for lab, v in by_label.items() if "spike" in lab or "bump" in lab for x in v]
    quiet = by_label.get("quiet", [])
    if spike and quiet:
        print(f"  spike/bump vs quiet: delta={mean_or_nan(spike)-mean_or_nan(quiet):+.4f} "
              f"nats, Welch t={welch_t(spike, quiet):+.2f}")

    # ---------------- structural / tokenizer sanity ----------------
    print("\n=== structural sanity over the whole bundle ===")
    firsts, lasts, lens, nsup, ratio = (collections.Counter(), collections.Counter(), [], [], [])
    bad_prefix = []
    for b, rows in bundle.items():
        for r in rows:
            ids, p = r["ids"], r["prefix_len"]
            firsts[tuple(ids[:3])] += 1
            lasts[tuple(ids[-2:])] += 1
            lens.append(len(ids))
            nsup.append(len(ids) - p)
            ratio.append((len(ids) - p) / len(ids))
            if not (0 < p < len(ids)):
                bad_prefix.append((b, r["idx"], p, len(ids)))
    print(f"datums={len(lens)} len: min={min(lens)} med={int(st.median(lens))} max={max(lens)}")
    print(f"supervised tokens: min={min(nsup)} med={int(st.median(nsup))} max={max(nsup)} "
          f"(supervised fraction med={st.median(ratio):.4f})")
    print(f"malformed prefix_len: {len(bad_prefix)}")
    print(f"distinct 3-token prefixes: {len(firsts)} -> {firsts.most_common(3)}")
    print(f"distinct 2-token suffixes: {len(lasts)} -> {lasts.most_common(6)}")
    print(f"datums shorter than 2048 (would skip the DSA indexer): "
          f"{sum(1 for L in lens if L < 2048)}")

    # do the different endings correlate with NLL / label?
    print("\nmean frozen-weight NLL grouped by final token:")
    by_last = collections.defaultdict(list)
    for b, rows in bundle.items():
        for r in rows:
            k = (b, r["idx"])
            if k in nll:
                by_last[r["ids"][-1]].append(mean_or_nan(nll[k]))
    for tok, v in sorted(by_last.items(), key=lambda x: -len(x[1]))[:6]:
        print(f"  last_id={tok:6d} n={len(v):3d} mean={mean_or_nan(v):.4f}")

    # duplicate documents?
    print("\nduplicate documents across the bundle:")
    seen = collections.defaultdict(list)
    for b, rows in bundle.items():
        for r in rows:
            seen[hash(tuple(r["ids"]))].append((b, r["idx"]))
    dups = {h: v for h, v in seen.items() if len(v) > 1}
    print(f"  {len(dups)} duplicated id-sequences",
          f"e.g. {list(dups.values())[:3]}" if dups else "")


if __name__ == "__main__":
    main()
