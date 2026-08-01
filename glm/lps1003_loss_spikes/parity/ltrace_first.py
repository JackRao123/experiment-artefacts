#!/usr/bin/env python3
"""First-divergence-in-forward-order analysis for ltrace dumps.

For each EVENT op (matched to HEALED ops by call signature), walk hooks in
forward order (embedding -> layer0.indexer -> layer0.core_attention ->
layer0.self_attention -> layer0.mlp -> layer0(out) -> layer1... -> final_ln ->
output_layer) and report the first hook whose max-bin z exceeds --zmin, plus a
per-layer max-z profile and the bin pattern at the first divergent hook.
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

from ltrace_analyze import load_rank, sig_of, unpack

STATS = ["rms", "absmax", "mean", "nnf"]


def fwd_key(path):
    if path == "embedding":
        return (-2, 0)
    m = re.match(r"decoder\.layers\.(\d+)(?:\.(.*))?$", path)
    if m:
        lay = int(m.group(1))
        sub = m.group(2) or "zzz_out"
        order = {"self_attention.indexer": 0, "self_attention.core_attention": 1,
                 "self_attention": 2, "mlp": 3, "zzz_out": 4}.get(sub, 5)
        return (lay, order)
    if path == "decoder.final_layernorm":
        return (9000, 0)
    if path == "output_layer":
        return (9001, 0)
    return (8000, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--event-ts", nargs=2, type=float, required=True)
    ap.add_argument("--healed-ts", nargs=2, type=float, required=True)
    ap.add_argument("--zmin", type=float, default=8.0)
    ap.add_argument("--relmin", type=float, default=0.02,
                    help="min relative deviation |x-m|/(|m|+1e-6)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "rank*_pid*.jsonl.gz")))
    for fp in files:
        rank = re.search(r"rank(\w+)_pid", os.path.basename(fp)).group(1)
        mm_fp = fp.replace(".jsonl.gz", "_modmap.json")
        modmap = json.load(open(mm_fp)) if os.path.exists(mm_fp) else {}
        ops = load_rank(fp)
        ev = [o for o in ops if args.event_ts[0] <= o["ts"] <= args.event_ts[1]]
        he = [o for o in ops if args.healed_ts[0] <= o["ts"] <= args.healed_ts[1]]
        by_sig = defaultdict(list)
        for o in he:
            by_sig[sig_of(o)].append(o)
        for o in ev:
            peers = by_sig.get(sig_of(o))
            if not peers or len(peers) < 2:
                continue
            pv = [unpack(p) for p in peers]
            x = unpack(o)
            ncu = len(o.get("cu") or [])
            # per hook: max z across bins/stats
            hook_z = {}
            hook_binz = {}
            for key, v in x.items():
                hid, b, s = key
                if s == 3:   # nonfinite counter: handle separately below
                    continue
                ref = [p.get(key) for p in pv if p.get(key) is not None]
                if len(ref) < 2:
                    continue
                m = sum(ref) / len(ref)
                sd = math.sqrt(sum((r - m) ** 2 for r in ref) / (len(ref) - 1))
                z = abs(v - m) / (sd + 1e-9)
                rel = abs(v - m) / (abs(m) + 1e-6)
                if rel < args.relmin:
                    continue
                if z > hook_z.get(hid, 0):
                    hook_z[hid] = z
                hook_binz.setdefault(hid, {})
                if z >= args.zmin:
                    hook_binz[hid][b] = max(hook_binz[hid].get(b, 0), z)
            if not hook_z:
                continue
            ordered = sorted(hook_z.items(),
                             key=lambda kv: fwd_key(modmap.get(str(kv[0]), {}).get("path", "")))
            first = next(((h, z) for h, z in ordered if z >= args.zmin), None)
            n_anom = sum(1 for _, z in hook_z.items() if z >= args.zmin)
            if first is None:
                continue
            fpth = modmap.get(str(first[0]), {}).get("path", "?")
            bins = hook_binz.get(first[0], {})
            nb = next((e[1] for e in o["entries"] if e[0] == first[0]), "?")
            print(f"rank{rank} op{o['op']} ncu={ncu} T_bins={nb} anomalous_hooks={n_anom}")
            print(f"    FIRST divergent: {fpth}  z={first[1]:.0f}  "
                  f"bins(z>={args.zmin:.0f}): {sorted(bins.items())}")
            # compact layer profile: max z per layer index
            prof = {}
            for h, z in hook_z.items():
                pth = modmap.get(str(h), {}).get("path", "")
                lay = fwd_key(pth)[0]
                prof[lay] = max(prof.get(lay, 0), z)
            keys = sorted(k for k in prof if 0 <= k < 9000)
            comp = " ".join(f"L{k}:{prof[k]:.0f}" for k in keys[:12])
            print(f"    layer max-z profile (first 12): emb:{prof.get(-2,0):.0f} {comp} "
                  f"finalLN:{prof.get(9000,0):.0f} out:{prof.get(9001,0):.0f}")


if __name__ == "__main__":
    main()
