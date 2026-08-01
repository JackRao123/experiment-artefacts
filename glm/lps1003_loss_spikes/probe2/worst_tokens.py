#!/usr/bin/env python3
"""Top-k worst-token inspector with decoded context, for LPS-1003 bisection.

Reimplements the IDEA of `debug_worst_tokens` from basetenlabs/subprime-rl
(`src/prime_rl/trainer/megatron/sft_train.py:432-598`,
`sft_metrics.py:174-206`): pick the highest-NLL supervised tokens, and show a
±20-token decoded window around each with the bad token bracketed, so a loss
number becomes something you can read.

Deliberate differences from theirs, because this codebase is not that one:

  * They hook Megatron's unreduced loss vector. Here the per-token logprobs come
    back from the trainer's /forward `loss_fn_outputs[i]["logprobs"]`, already
    split per datum and already gathered across CP — so no stashing, no FIFO
    join, and none of their CP hazard (their `target_position` is a local
    interleaved shard index under cp>1; they run cp=1, we run cp=16).
  * They forbid packing outright ("cross-document attention leakage"). Packing
    is exactly what we are investigating, so every token is additionally tagged
    with its packed-partition slot and whether its document sits at a partition
    TAIL — the position prod destruction lands on.

Alignment (mirrors probe_nll.py, which mirrors the production client): for ids
of length L, targets = ids[1:], logprobs[j] is the NLL of target ids[j+1], and
supervised positions are j >= prefix_len-1. Context windows are cut from the
target stream so that window[relative] IS the bad target token.

usage:
  python3 worst_tokens.py --probe runs/window/probe.jsonl --bundle train_bundle_0_31.jsonl.gz \
      [--lp-dir runs/window/lp] [--topk 32] [--tokenizer <hf-path>] [--nll-threshold 2.0]
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import math
import os
import statistics as st

CTX = 20


def load_bundle(path):
    out = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            out[(r["batch"], r["idx"])] = r
    return out


def partition_map(bundle, batch, max_seq_len):
    rows = sorted([r for (b, i), r in bundle.items() if b == batch], key=lambda r: r["idx"])
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
    slot = {}
    for p_i, part in enumerate(parts):
        for s, idx in enumerate(part):
            slot[idx] = (p_i, s, len(part), s == len(part) - 1)
    return parts, slot


def get_tokenizer(path):
    if not path:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001 — decoding is a nicety, never fatal
        print(f"[warn] tokenizer unavailable ({type(e).__name__}: {e}); ids only")
        return None


def dec(tok, ids):
    if tok is None or not len(ids):
        return ""
    try:
        return tok.decode(list(ids))
    except Exception:  # noqa: BLE001
        return "<?>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="probe_nll.py output jsonl")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--lp-dir", default=None, help="dir of per-token .npz from --dump-logprobs")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--max-seq-len", type=int, default=262144)
    ap.add_argument("--nll-threshold", type=float, default=2.0)
    ap.add_argument("--out", default=None, help="write per-token rows as jsonl")
    args = ap.parse_args()

    bundle = load_bundle(args.bundle)
    tok = get_tokenizer(args.tokenizer)

    # ---- per-datum summary from the probe (always available) ----
    recs = [json.loads(l) for l in open(args.probe)]
    datum_recs = [r for r in recs if r.get("kind") == "batches"]
    print(f"probe ops: {len(datum_recs)}")
    destroyed = []
    for r in datum_recs:
        b = r["key"] if isinstance(r["key"], int) else r["key"]
        parts, slot = partition_map(bundle, b, args.max_seq_len)
        hot = [d for d in r["datums"] if d["nll"] > args.nll_threshold]
        print(f"\nop batch={b} rep={r['repeat']} mean_nll={r['mean_nll']:.4f} "
              f"partitions={[f'{p[0]}-{p[-1]}' for p in parts]}")
        if not hot:
            print(f"  no datum above {args.nll_threshold} nats "
                  f"(max={max(d['nll'] for d in r['datums']):.3f})")
        for d in hot:
            p_i, s, n_in, is_tail = slot.get(d["idx"], (-1, -1, -1, False))
            print(f"  DESTROYED idx={d['idx']} nll={d['nll']:.3f} n_sup={d['n_sup']} "
                  f"partition={p_i} slot={s}/{n_in-1} tail={is_tail} "
                  f"min_lp={d['min_lp']:.2f} n_below_5={d.get('n_below_5')} "
                  f"n_below_10={d.get('n_below_10')}")
            destroyed.append((b, d["idx"], d["nll"], is_tail))
            # the probe already carries the 5 worst positions per datum
            row = bundle.get((b, d["idx"]))
            if row and d.get("worst_pos"):
                ids, p = row["ids"], row["prefix_len"]
                targets = ids[1:]
                print("    worst supervised tokens (from probe worst_pos):")
                for pos, lp in zip(d["worst_pos"], d["worst_lp"]):
                    a, z = max(pos - CTX, 0), min(pos + CTX + 1, len(targets))
                    rel = pos - a
                    win = targets[a:z]
                    bad = targets[pos] if pos < len(targets) else -1
                    marked = (f"{dec(tok, win[:rel])}<<BAD>>{dec(tok, [bad])}<</BAD>>"
                              f"{dec(tok, win[rel+1:])}")
                    frac = (pos - (p - 1)) / max(1, d["n_sup"])
                    print(f"      pos={pos} (sup_frac={frac:.2f}) nll={-lp:7.3f} "
                          f"id={bad} text={dec(tok, [bad])!r}")
                    if tok is not None:
                        print(f"        ctx={marked!r}")

    # ---- full per-token analysis when npz dumps exist ----
    if not args.lp_dir:
        print("\n(no --lp-dir: skipping full per-token top-k; pass probe_nll.py "
              "--dump-logprobs to enable)")
        return
    try:
        import numpy as np
    except ImportError:
        print("numpy unavailable; cannot read npz dumps")
        return

    files = sorted(glob.glob(os.path.join(args.lp_dir, "*.npz")))
    print(f"\n=== per-token analysis over {len(files)} npz dumps ===")
    allrows = []
    mismatch = []
    for f in files:
        base = os.path.basename(f)
        try:
            b = int(base.split("_")[0].lstrip("b"))
        except ValueError:
            continue
        parts, slot = partition_map(bundle, b, args.max_seq_len)
        z = np.load(f)
        for key in z.files:
            idx = int(key.lstrip("d"))
            row = bundle.get((b, idx))
            if row is None:
                continue
            ids, p = row["ids"], row["prefix_len"]
            targets = ids[1:]
            lp = z[key]
            # Guard against pairing a dump with the wrong bundle: the logprob
            # vector must be exactly len(ids)-1 (one per target).
            if len(lp) != len(ids) - 1:
                mismatch.append((b, idx, len(lp), len(ids) - 1))
                continue
            sup = lp[p - 1:]
            nll = -sup.astype("float64")
            p_i, s, n_in, is_tail = slot.get(idx, (-1, -1, -1, False))
            for j in range(len(nll)):
                allrows.append({
                    "batch": b, "idx": idx, "sup_j": j, "n_sup": len(nll),
                    "target_position": p - 1 + j,
                    "sup_frac": j / max(1, len(nll)),
                    "nll": float(nll[j]),
                    "target_id": int(targets[p - 1 + j]) if p - 1 + j < len(targets) else -1,
                    "partition": p_i, "slot": s, "is_tail": is_tail,
                })
    if mismatch:
        print(f"!! {len(mismatch)} datum dumps did not match the bundle by length "
              f"(logprobs != len(ids)-1) — WRONG BUNDLE for these dumps. "
              f"e.g. {mismatch[:3]}")
    if not allrows:
        print("no per-token rows recovered — bundle does not match these dumps")
        return
    nlls = [r["nll"] for r in allrows]
    nlls_sorted = sorted(nlls)

    def q(f):
        return nlls_sorted[min(len(nlls_sorted) - 1, int(f * len(nlls_sorted)))]

    print(f"supervised tokens: {len(nlls)}")
    print(f"loss/p50={q(.5):.4f} p90={q(.9):.4f} p99={q(.99):.4f} "
          f"max={max(nlls):.4f} min={min(nlls):.4f} mean={st.mean(nlls):.4f}")

    print(f"\ntop-{args.topk} worst supervised tokens:")
    worst = sorted(allrows, key=lambda r: -r["nll"])[:args.topk]
    for rank, r in enumerate(worst, 1):
        row = bundle[(r["batch"], r["idx"])]
        targets = row["ids"][1:]
        pos = r["target_position"]
        a, z2 = max(pos - CTX, 0), min(pos + CTX + 1, len(targets))
        rel = pos - a
        win = targets[a:z2]
        marked = (f"{dec(tok, win[:rel])}<<BAD>>{dec(tok, [r['target_id']])}<</BAD>>"
                  f"{dec(tok, win[rel+1:])}")
        print(f" #{rank:3d} nll={r['nll']:8.3f} b{r['batch']}/d{r['idx']:02d} "
              f"pos={pos} sup_frac={r['sup_frac']:.2f} slot={r['slot']} tail={r['is_tail']} "
              f"id={r['target_id']} text={dec(tok,[r['target_id']])!r}")
        if tok is not None:
            print(f"       ctx={marked!r}")

    # --- the bisection questions ---
    print("\n=== is a specific TOKEN ID responsible? ===")
    by_id = collections.defaultdict(list)
    for r in allrows:
        by_id[r["target_id"]].append(r["nll"])
    freq_worst = collections.Counter(r["target_id"] for r in worst)
    print(f"distinct supervised target ids: {len(by_id)}")
    print(f"top ids among the {args.topk} worst: {freq_worst.most_common(8)}")
    common = [(i, v) for i, v in by_id.items() if len(v) >= 20]
    common.sort(key=lambda x: -st.mean(x[1]))
    print("highest mean-NLL ids with n>=20 occurrences:")
    for i, v in common[:8]:
        print(f"  id={i:6d} n={len(v):5d} mean={st.mean(v):.3f} max={max(v):.3f} "
              f"text={dec(tok,[i])!r}")

    print("\n=== is a specific POSITION responsible? ===")
    bins = collections.defaultdict(list)
    for r in allrows:
        bins[min(9, int(r["sup_frac"] * 10))].append(r["nll"])
    for k in sorted(bins):
        v = bins[k]
        print(f"  sup_frac [{k/10:.1f},{(k+1)/10:.1f}) n={len(v):6d} mean={st.mean(v):.4f} "
              f"p99={sorted(v)[int(.99*len(v))]:.3f}")
    first = [r["nll"] for r in allrows if r["sup_j"] == 0]
    rest = [r["nll"] for r in allrows if r["sup_j"] > 0]
    if first and rest:
        print(f"  first supervised token: n={len(first)} mean={st.mean(first):.4f} "
              f"vs rest mean={st.mean(rest):.4f} (a masking off-by-one would spike this)")

    # Is the prompt/completion boundary where we think it is? If prefix_len is
    # off by one, or the chat template is mis-split, the first supervised target
    # is the wrong token — and it would be the SAME wrong token every time.
    print("\n=== first supervised token identity (prefix_len boundary check) ===")
    firstrows = [r for r in allrows if r["sup_j"] == 0]
    fid = collections.Counter(r["target_id"] for r in firstrows)
    print(f"distinct first-supervised target ids over {len(firstrows)} datum-dumps: {len(fid)}")
    for i, c in fid.most_common(6):
        v = [r["nll"] for r in firstrows if r["target_id"] == i]
        print(f"  id={i:6d} n={c:4d} mean_nll={st.mean(v):7.3f} text={dec(tok,[i])!r}")
    if len(fid) == 1:
        print("  -> CONSTANT first supervised token. If its NLL is high, the model is "
              "failing on a token it should have memorised => suspect the boundary.")
    else:
        print("  -> first supervised token VARIES with content (expected when the "
              "completion begins with the answer rather than a fixed template).")
    # what the last prompt token / boundary looks like, decoded
    if tok is not None and firstrows:
        print("\n  boundary context for 3 datums (prompt tail | completion head):")
        for r in firstrows[:3]:
            row = bundle[(r["batch"], r["idx"])]
            ids, p = row["ids"], row["prefix_len"]
            print(f"    b{r['batch']}/d{r['idx']:02d} prefix_len={p} "
                  f"prompt_tail={dec(tok, ids[max(0,p-12):p])!r} "
                  f"completion_head={dec(tok, ids[p:p+12])!r}")

    print("\n=== does partition-TAIL position raise per-token NLL? ===")
    t = [r["nll"] for r in allrows if r["is_tail"]]
    m = [r["nll"] for r in allrows if not r["is_tail"]]
    if t and m:
        va, vb = st.variance(t), st.variance(m)
        se = math.sqrt(va / len(t) + vb / len(m))
        print(f"  tail tokens n={len(t)} mean={st.mean(t):.4f} p99={sorted(t)[int(.99*len(t))]:.3f}")
        print(f"  non-tail    n={len(m)} mean={st.mean(m):.4f} p99={sorted(m)[int(.99*len(m))]:.3f}")
        print(f"  delta={st.mean(t)-st.mean(m):+.4f} nats Welch t={(st.mean(t)-st.mean(m))/se:+.2f}"
              if se > 0 else "")

    if args.out:
        with open(args.out, "w") as fh:
            for r in allrows:
                fh.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(allrows)} per-token rows -> {args.out}")


if __name__ == "__main__":
    main()
