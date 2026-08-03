#!/usr/bin/env python3
"""Spike detector for the LPS-1003 fix validation (100-step replay of Mudith's recipe).

Applies the criterion documented in repro/README.md: an isolated loss spike is
~2-4x the LOCAL median and recovers within one step (grad_norm spikes 3-45x
alongside where available). Baseline for GLM-5.2-FP8 on B300 is 26 spikes in
147 steps (~17.7%/step); three BF16 models on identical data gave 0/745.

Usage:
  analyze_replay_spikes.py --replay /path/lps1003_replay100.jsonl [--tag fix100]
  analyze_replay_spikes.py --baseline            # self-check against the W&B history
"""
from __future__ import annotations
import argparse, json, math, statistics as st

# CALIBRATION (2026-08-02): repro/README.md documents 26 spikes / 147 steps
# (17.7%) for GLM-5.2-FP8 on B300. Sweeping this detector over the W&B history
# in loss_histories.json, WIN=5 + RATIO=1.6 yields 25/147 (17.0%) — i.e. it
# reproduces the documented baseline sensitivity. RATIO=2.0 under-counts (17).
# We judge the fixed run at the CALIBRATED threshold so a clean result cannot
# be an artifact of a lenient detector; STRICT is reported alongside.
WIN = 5          # half-width of the local-median window
RATIO = 1.6      # calibrated to reproduce the documented 26/147 baseline
STRICT = 2.0     # stricter threshold, reported for reference
RECOVER = 1.5    # "recovers in one step": next nll < RECOVER * local median


def local_median(ys, i):
    lo, hi = max(0, i - WIN), min(len(ys), i + WIN + 1)
    neigh = [ys[j] for j in range(lo, hi) if j != i]
    return st.median(neigh) if neigh else float("nan")


def find_spikes(steps, ys, gns=None, ratio=RATIO):
    out = []
    for i, y in enumerate(ys):
        m = local_median(ys, i)
        if not (m > 0) or math.isnan(m):
            continue
        r = y / m
        if r <= ratio:
            continue
        nxt = ys[i + 1] / m if i + 1 < len(ys) else None
        recovered = nxt is None or nxt < RECOVER
        g = None
        if gns and i < len(gns) and gns[i] is not None:
            gm = local_median([x for x in gns if x is not None], i)
            if gm and gm > 0:
                g = gns[i] / gm
        out.append({"step": steps[i], "nll": y, "local_median": m,
                    "ratio": r, "recovered_next_step": recovered,
                    "grad_norm_ratio": g})
    return out


def report(name, steps, ys, gns=None):
    sp = find_spikes(steps, ys, gns)
    n = len(ys)
    print(f"\n=== {name}")
    print(f"steps={n}  first={ys[0]:.4f}  last={ys[-1]:.4f}  "
          f"min={min(ys):.4f}  max={max(ys):.4f}  median={st.median(ys):.4f}")
    rate = len(sp) / n * 100 if n else 0
    strict = find_spikes(steps, ys, gns, ratio=STRICT)
    print(f"SPIKES calibrated (>{RATIO}x local median): {len(sp)}  ({rate:.1f}% of steps)")
    print(f"SPIKES strict     (>{STRICT}x local median): {len(strict)}  "
          f"({len(strict)/n*100 if n else 0:.1f}% of steps)")
    for s in sp:
        g = f"  grad_norm={s['grad_norm_ratio']:.1f}x" if s["grad_norm_ratio"] else ""
        print(f"   step {s['step']:>4}: nll={s['nll']:.4f} = {s['ratio']:.2f}x "
              f"local median {s['local_median']:.4f}"
              f"{'  (recovered)' if s['recovered_next_step'] else '  (SUSTAINED)'}{g}")
    if not sp:
        print("   none — curve is clean")
    return sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", help="train_replay.py output JSONL")
    ap.add_argument("--tag", default=None, help="only rows with this tag")
    ap.add_argument("--baseline", action="store_true",
                    help="self-check the detector against loss_histories.json")
    a = ap.parse_args()

    if a.baseline:
        hist = json.load(open("loss_histories.json"))
        for k, v in hist.items():
            steps = [p[0] for p in v]
            ys = [p[1] for p in v]
            report(f"BASELINE {k}", steps, ys)
            # documented window: first 147 steps -> expect 26
            if len(ys) >= 147:
                report(f"BASELINE {k} [first 147 steps, expect ~26]",
                       steps[:147], ys[:147])
        return

    rows = []
    for line in open(a.replay):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("kind") != "train_step":      # skip the status header row
            continue
        if a.tag and r.get("tag") not in (a.tag, None):
            continue
        rows.append(r)
    rows.sort(key=lambda r: int(r.get("step", 0)))
    steps = [int(r["step"]) for r in rows]
    ys = [float(r["mean_nll"]) for r in rows]

    def _gn(r):
        om = r.get("optim_metrics")
        if isinstance(om, str):
            try:
                om = json.loads(om.replace("'", '"'))
            except Exception:
                return None
        return float(om["grad_norm"]) if isinstance(om, dict) and "grad_norm" in om else None

    gns = [_gn(r) for r in rows]
    ok = [(s, y, g) for s, y, g in zip(steps, ys, gns) if y is not None]
    steps, ys, gns = [x[0] for x in ok], [x[1] for x in ok], [x[2] for x in ok]
    sp = report(f"REPLAY {a.replay}", steps, ys, gns)

    print("\n--- verdict vs documented baseline ---")
    print("baseline GLM-5.2-FP8 on B300: 26 spikes / 147 steps = 17.7%/step")
    exp = 0.177 * len(ys)
    print(f"expected if bug still live over {len(ys)} steps: ~{exp:.1f} spikes")
    if not sp:
        # Poisson probability of observing zero at the baseline rate
        print(f"observed: 0  ->  P(0 | baseline rate) = e^-{exp:.1f} "
              f"= {math.exp(-exp):.2e}")
        print("VERDICT: clean — consistent with the fix eliminating the spikes")
    else:
        print(f"observed: {len(sp)}  -> inspect above; "
              "compare rate to 17.7%/step before concluding")


if __name__ == "__main__":
    main()
