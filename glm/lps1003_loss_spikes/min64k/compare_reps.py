#!/usr/bin/env python3
"""Compare 5 /forward reps of the min64k synthetic probe.

Prints per-datum NLL for every rep, plus rep0-vs-rest deltas and the max
per-token logprob divergence (the bug fingerprint = rep0 differs, heals by
rep1; destroyed = NLL delta > 2 nats).
"""
import glob
import gzip
import json
import sys

run_dir = sys.argv[1]
reps = {}
for f in sorted(glob.glob(f"{run_dir}/min64k_window_rep*.json.gz")):
    r = json.load(gzip.open(f, "rt"))
    reps[r["rep"]] = r

if not reps:
    sys.exit(f"no rep dumps in {run_dir}")

ks = sorted(reps)
n_datums = len(reps[ks[0]]["nlls"])
labels = ["palindrome", "random-tile"][:n_datums] + [
    f"d{i}" for i in range(2, n_datums)
]

print(f"reps={ks} datums={n_datums}")
print(f"{'datum':<12}" + "".join(f"rep{k:<9}" for k in ks) + "max|dNLL| vs rep1+")
verdict_fire = False
for i in range(n_datums):
    row = [reps[k]["nlls"][i] for k in ks]
    rest = row[1:]
    dmax = max(abs(row[0] - v) for v in rest) if rest else 0.0
    if dmax > 2.0:
        verdict_fire = True
    print(f"{labels[i]:<12}" + "".join(f"{v:<12.6f}" for v in row) + f"{dmax:.6f}")

# per-position divergence rep0 vs each later rep, as contiguous bands
def bands(deltas, thr):
    """Contiguous position ranges where |dlp| > thr (gap<=64 merges)."""
    out, start, last = [], None, None
    for p, d in enumerate(deltas):
        if d > thr:
            if start is None:
                start = p
            elif p - last > 64:
                out.append((start, last))
                start = p
            last = p
    if start is not None:
        out.append((start, last))
    return out

for i in range(n_datums):
    a = reps[ks[0]]["logprobs"][i]
    for k in ks[1:]:
        b = reps[k]["logprobs"][i]
        deltas = [abs(x - y) for x, y in zip(a, b)]
        worst = max(range(len(deltas)), key=deltas.__getitem__)
        n_big = sum(1 for d in deltas if d > 1.0)
        bl = bands(deltas, 1.0)
        band_str = " ".join(f"[{s}-{e}]" for s, e in bl[:12])
        if len(bl) > 12:
            band_str += f" ...(+{len(bl)-12} more)"
        print(
            f"datum {i} ({labels[i]}) rep0 vs rep{k}: max|dlp|={deltas[worst]:.4f}"
            f" @ pos {worst}; n(|dlp|>1)={n_big}; bands>1nat: {band_str or 'none'}"
        )

# dump full per-position delta arrays for offline plotting
import gzip as _gz

dump = {
    "positions_per_datum": [len(reps[ks[0]]["logprobs"][i]) for i in range(n_datums)],
    "labels": labels,
    "rep0_vs": {
        str(k): [
            [round(abs(x - y), 4) for x, y in
             zip(reps[ks[0]]["logprobs"][i], reps[k]["logprobs"][i])]
            for i in range(n_datums)
        ]
        for k in ks[1:]
    },
}
out_path = f"{run_dir}/positional_deltas.json.gz"
with _gz.open(out_path, "wt") as fh:
    json.dump(dump, fh)
print(f"full per-position |dlp| arrays -> {out_path}")

print("VERDICT:", "FIRED (rep0 destroyed vs later reps)" if verdict_fire
      else "clean — no rep0-vs-rest destruction above 2 nats")
