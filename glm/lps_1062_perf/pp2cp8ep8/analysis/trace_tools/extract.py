#!/usr/bin/env python3
"""Stream a kineto pt.trace.json (one event object per '  {' ... '  },' block)
and emit compact TSVs for analysis."""
import json, sys, os
from collections import defaultdict

src = sys.argv[1]
outdir = sys.argv[2]
os.makedirs(outdir, exist_ok=True)

KEEP_GPU = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}

catstats = defaultdict(lambda: [0, 0.0])  # (ph,cat) -> [count, total_dur]
namestats = defaultdict(lambda: [0, 0.0])  # (cat,name) -> [count, total_dur] for gpu cats

f_ann = open(os.path.join(outdir, "ann_cpu.tsv"), "w")
f_gann = open(os.path.join(outdir, "ann_gpu.tsv"), "w")
f_ker = open(os.path.join(outdir, "kernels.tsv"), "w")
f_rt = open(os.path.join(outdir, "runtime_big.tsv"), "w")

def emit(ev):
    ph = ev.get("ph")
    cat = ev.get("cat", "?")
    catstats[(ph, cat)][0] += 1
    dur = ev.get("dur")
    if dur is not None:
        catstats[(ph, cat)][1] += dur
    if ph != "X":
        return
    name = ev.get("name", "")
    ts = ev.get("ts", 0)
    pid = ev.get("pid", "")
    tid = ev.get("tid", "")
    if cat == "user_annotation":
        f_ann.write(f"{ts}\t{dur}\t{tid}\t{name[:200]}\n")
    elif cat in KEEP_GPU:
        key = (cat, name[:150])
        namestats[key][0] += 1
        namestats[key][1] += dur
        if cat == "gpu_user_annotation":
            f_gann.write(f"{ts}\t{dur}\t{pid}\t{tid}\t{name[:200]}\n")
        else:
            f_ker.write(f"{ts}\t{dur}\t{pid}\t{tid}\t{cat}\t{name[:150]}\n")
    elif cat == "cuda_runtime" or cat == "cuda_driver":
        if dur and dur > 500.0:
            f_rt.write(f"{ts}\t{dur}\t{tid}\t{name[:120]}\n")

nev = 0
buf = None
with open(src, "r", buffering=1 << 22) as f:
    for line in f:
        if line.startswith("  {"):
            buf = ["{"]
        elif buf is not None:
            if line.startswith("  },") or line.startswith("  }"):
                buf.append("}")
                try:
                    emit(json.loads("".join(buf)))
                except Exception:
                    pass
                nev += 1
                buf = None
            else:
                buf.append(line)

for fh in (f_ann, f_gann, f_ker, f_rt):
    fh.close()

with open(os.path.join(outdir, "catstats.tsv"), "w") as f:
    for (ph, cat), (c, d) in sorted(catstats.items(), key=lambda x: -x[1][1]):
        f.write(f"{ph}\t{cat}\t{c}\t{d:.0f}\n")
with open(os.path.join(outdir, "gpunames.tsv"), "w") as f:
    for (cat, name), (c, d) in sorted(namestats.items(), key=lambda x: -x[1][1]):
        f.write(f"{cat}\t{c}\t{d:.0f}\t{name}\n")
print(f"events={nev}", file=sys.stderr)
