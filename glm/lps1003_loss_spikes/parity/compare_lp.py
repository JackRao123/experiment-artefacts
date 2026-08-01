#!/usr/bin/env python3
"""LPS-1003 parity analysis over probe_lp.py dumps.

Modes:
  wobble   within-arm rep-to-rep per-token delta distribution (noise floor)
  cross    prod-vs-devbox per-token delta distribution vs each arm's floor
  event    anatomy of destroyed datums: where along the doc the NLL excess
           starts, top offending tokens, vs the same datum in a healed rep

Dump format: {tag, rep, nlls, destroyed, logprobs: [[...] x 32]}.
Weights/prefix info comes from the payload JSON.
"""
import argparse
import glob
import gzip
import json
import math
import os


def load(fp):
    with gzip.open(fp, "rt") as fh:
        return json.load(fh)


def pairs_delta(a, b):
    """Per-token |delta| stats across all datums of two reps."""
    n = tot = 0
    gt = {0.01: 0, 0.1: 0, 1.0: 0}
    mx = 0.0
    sum2 = 0.0
    for la, lb in zip(a["logprobs"], b["logprobs"]):
        for x, y in zip(la, lb):
            d = abs(x - y)
            tot += 1
            sum2 += d * d
            if d > 0.01:
                gt[0.01] += 1
                if d > 0.1:
                    gt[0.1] += 1
                    if d > 1.0:
                        gt[1.0] += 1
            if d > mx:
                mx = d
    return {"tokens": tot, "rms": math.sqrt(sum2 / max(tot, 1)), "max": mx,
            "frac_gt_0.01": gt[0.01] / max(tot, 1),
            "frac_gt_0.1": gt[0.1] / max(tot, 1),
            "frac_gt_1.0": gt[1.0] / max(tot, 1)}


def supervised_slice(lp, w):
    """Return (positions, lp values) where weight > 0."""
    return [(i, v) for i, (v, ww) in enumerate(zip(lp, w)) if ww > 0]


def fmt(d):
    return (f"tokens={d['tokens']} rms={d['rms']:.4f} max={d['max']:.2f} "
            f">0.01={d['frac_gt_0.01']:.3%} >0.1={d['frac_gt_0.1']:.3%} "
            f">1.0={d['frac_gt_1.0']:.4%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["wobble", "cross", "event"])
    ap.add_argument("--a", nargs="+", help="dump files arm A (or the event rep for event mode)")
    ap.add_argument("--b", nargs="+", default=[], help="dump files arm B (or healed reps for event mode)")
    ap.add_argument("--payload", default=None, help="payload JSON (for weights; event mode)")
    args = ap.parse_args()

    A = [load(f) for f in sorted(sum([glob.glob(x) for x in args.a], []))]
    B = [load(f) for f in sorted(sum([glob.glob(x) for x in args.b], []))] if args.b else []

    if args.mode == "wobble":
        print(f"# within-arm wobble over {len(A)} reps")
        for i in range(len(A) - 1):
            d = pairs_delta(A[i], A[i + 1])
            print(f"rep{A[i]['rep']} vs rep{A[i+1]['rep']}: {fmt(d)}")
    elif args.mode == "cross":
        print(f"# cross-arm: {len(A)} x {len(B)} rep pairs")
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                d = pairs_delta(a, b)
                print(f"A[{a['tag']}/r{a['rep']}] vs B[{b['tag']}/r{b['rep']}]: {fmt(d)}")
    else:  # event
        w = None
        if args.payload:
            pl = json.load(open(args.payload))
            w = [d["loss_fn_inputs"]["weights"]["data"] for d in pl["data"]]
        ev = A[0]
        healed = B[0] if B else None
        print(f"# event rep {ev['tag']}/r{ev['rep']} destroyed={ev['destroyed']}")
        for idx_s, nll in ev["destroyed"].items():
            idx = int(idx_s)
            lp_e = ev["logprobs"][idx]
            lp_h = healed["logprobs"][idx] if healed else None
            ww = w[idx] if w else [1.0] * len(lp_e)
            sup = supervised_slice(lp_e, ww)
            n_sup = len(sup)
            # position-decile mean NLL (event vs healed)
            print(f"\n## datum {idx}: nll={nll} n_sup={n_sup} "
                  f"(healed nll={healed['nlls'][idx]:.3f})" if healed else
                  f"\n## datum {idx}: nll={nll} n_sup={n_sup}")
            dec = max(1, n_sup // 10)
            for d10 in range(10):
                seg = sup[d10 * dec:(d10 + 1) * dec]
                if not seg:
                    continue
                m_e = -sum(v for _, v in seg) / len(seg)
                if lp_h is not None:
                    m_h = -sum(lp_h[i] for i, _ in seg) / len(seg)
                    print(f"  decile {d10}: event {m_e:8.3f}  healed {m_h:8.3f}  excess {m_e-m_h:+8.3f}")
                else:
                    print(f"  decile {d10}: event {m_e:8.3f}")
            # first supervised position where excess becomes persistent
            if lp_h is not None:
                run = 0
                onset = None
                for i, v in sup:
                    if (lp_h[i] - v) > 1.0:
                        run += 1
                        if run >= 5 and onset is None:
                            onset = i
                    else:
                        run = 0
                print(f"  onset (5-run of >1 nat excess): target-pos {onset}")
                worst = sorted(((lp_h[i] - v, i) for i, v in sup), reverse=True)[:8]
                print("  worst excess tokens (excess, target-pos):",
                      [(round(e, 2), i) for e, i in worst])


if __name__ == "__main__":
    main()
