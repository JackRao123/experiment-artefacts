#!/usr/bin/env python3
"""L3 d16 CPU-blocked deep dive v3 (gauss, LPS-1062 §3b).

v3 additions: fixed kernel-window hit test; GPU gap decomposition (size
histogram); gap -> preceding-host-block attribution (did the GPU run dry
while/just-after the host was blocked?); deviceSync cluster timing.
"""
import json
import sys
import time
from collections import Counter

import numpy as np
from perfetto.trace_processor import TraceProcessor

path = sys.argv[1]
out_json = sys.argv[2] if len(sys.argv) > 2 else None
t0 = time.time()
tp = TraceProcessor(trace=path)


def q(sql):
    return tp.query(sql).as_pandas_dataframe()


step = q("SELECT ts, dur, name FROM slice WHERE name LIKE 'ProfilerStep%' ORDER BY dur DESC")
if len(step):
    S = float(step.ts.iloc[0]); E = S + float(step.dur.iloc[0]); step_name = step.name.iloc[0]
else:
    ext = q("SELECT MIN(ts) AS lo, MAX(ts+dur) AS hi FROM slice WHERE dur>0")
    S, E = float(ext.lo.iloc[0]), float(ext.hi.iloc[0]); step_name = "whole-extent"
print(f"[{time.time()-t0:.0f}s] step {step_name}: {(E-S)/1e9:.2f}s", flush=True)

gpu = q("SELECT ts, dur FROM slice WHERE category IN ('kernel','gpu_memcpy','gpu_memset') AND dur>0")
ga = gpu.ts.to_numpy(dtype=np.int64); gb = ga + gpu.dur.to_numpy(dtype=np.int64)
order = np.argsort(ga, kind="stable"); ga, gb = ga[order], gb[order]
us, ue = [], []
cs, ce = int(ga[0]), int(gb[0])
for s_, e_ in zip(ga[1:], gb[1:]):
    if s_ <= ce:
        if e_ > ce: ce = int(e_)
    else:
        us.append(cs); ue.append(ce); cs, ce = int(s_), int(e_)
us.append(cs); ue.append(ce)
us = np.array(us, dtype=np.int64); ue = np.array(ue, dtype=np.int64)
busy = int((ue - us).sum())
gpu_idle_s = ((E - S) - busy) / 1e9
kernel_count = int(q("SELECT COUNT(*) AS n FROM slice WHERE category='kernel'").n.iloc[0])
print(f"[{time.time()-t0:.0f}s] gpu busy {busy/1e9:.2f}s idle {gpu_idle_s:.2f}s kernels {kernel_count:,}", flush=True)

# gap table (within step window)
gap_s = us[1:]          # gap starts = next busy start
gap_e = ue[:-1]         # gap ends = previous busy end... (invert: gap = [ue[i], us[i+1]])
gap_s, gap_e = ue[:-1], us[1:]
m = (gap_s >= S) & (gap_e <= E) & (gap_e > gap_s)
gap_s, gap_e = gap_s[m], gap_e[m]
gap_d = gap_e - gap_s
buckets = [(0, 100_000), (100_000, 1_000_000), (1_000_000, 10_000_000), (10_000_000, 1 << 62)]
gap_hist = []
for lo, hi in buckets:
    mm = (gap_d >= lo) & (gap_d < hi)
    gap_hist.append({"range_us": [lo / 1e3, hi / 1e3 if hi < 1 << 62 else -1],
                     "count": int(mm.sum()), "total_s": float(gap_d[mm].sum() / 1e9)})

hostapi = q("""
    SELECT id, parent_id, ts, dur, name FROM slice
    WHERE category='cuda_runtime' AND name IN (
        'cudaStreamSynchronize','cudaMemcpyAsync','cudaEventSynchronize','cudaDeviceSynchronize')
""")
ops = q("SELECT id, parent_id, ts, dur, name FROM slice WHERE category IN ('cpu_op','user_annotation')")
print(f"[{time.time()-t0:.0f}s] hostapi {len(hostapi):,}; ops {len(ops):,}; gaps {len(gap_d):,}", flush=True)

op_name = dict(zip(ops.id, ops.name))
op_parent = dict(zip(ops.id, ops.parent_id))
op_ts = dict(zip(ops.id, ops.ts))
op_dur = dict(zip(ops.id, ops.dur))
del ops

CKPT = {"CheckpointFunction", "CheckpointFunctionBackward",
        "LookaheadCheckpointFunction", "LookaheadCheckpointFunctionBackward"}
CKPT_BWD = {"CheckpointFunctionBackward", "LookaheadCheckpointFunctionBackward"}
CKPT_FWD = {"CheckpointFunction", "LookaheadCheckpointFunction"}

classes = {k: [0, 0] for k in
           ["FIXB_LAYOUT", "FIXA_DSA_BWD", "DISPATCH_REPLAY", "DISPATCH_FWD",
            "ROPE_HOST_F", "OTHER_ITEM", "OTHER"]}
class_windows = {k: [] for k in classes}
nz_index_windows = []
other_sig_t = Counter(); other_sig_c = Counter()
devsync_ts = []

for pid, ts, dur, name in zip(hostapi.parent_id, hostapi.ts, hostapi.dur, hostapi.name):
    if name == "cudaDeviceSynchronize":
        devsync_ts.append((int(ts), int(dur)))
    anc = []
    cur = pid
    d = 0
    while cur and cur in op_name and d < 40:
        anc.append(cur)
        cur = op_parent.get(cur)
        d += 1
    names = [op_name[a] for a in anc]
    aset = set(names)
    cls = "OTHER"
    if "FusedSparseAttentionFuncBackward" in aset:
        cls = "FIXA_DSA_BWD"
    elif name == "cudaEventSynchronize" and aset & CKPT_BWD:
        cls = "DISPATCH_REPLAY"
    elif name == "cudaEventSynchronize" and aset & CKPT_FWD:
        cls = "DISPATCH_FWD"
    elif len(names) >= 2 and names[0] == "aten::nonzero" and names[1] == "aten::index":
        cls = "FIXB_LAYOUT"
        nz_index_windows.append((int(op_ts[anc[1]]), int(op_dur[anc[1]])))
    elif names and names[0] in ("aten::to", "aten::item", "aten::_local_scalar_dense",
                                "aten::copy_", "aten::_to_copy", "aten::tolist"):
        rope = False
        for j, n in enumerate(names):
            if n in ("aten::to", "aten::_to_copy") and j + 1 < len(names) and names[j + 1] in CKPT:
                rope = True
                break
        if not rope and names[0] in ("aten::item", "aten::_local_scalar_dense") and aset & CKPT:
            rope = True
        cls = "ROPE_HOST_F" if rope else "OTHER_ITEM"
        if not rope:
            sg = names[0] + " <- " + (names[1] if len(names) > 1 else "?")
            other_sig_t[sg] += int(dur); other_sig_c[sg] += 1
    else:
        sg = name + " <- " + (names[0] if names else "?") + (" <- " + names[1] if len(names) > 1 else "")
        other_sig_t[sg] += int(dur); other_sig_c[sg] += 1
    classes[cls][0] += 1
    classes[cls][1] += int(dur)
    class_windows[cls].append((int(ts), int(dur)))

print(f"[{time.time()-t0:.0f}s] classified", flush=True)

# gap -> host block attribution, two rules:
#  STRICT: gap starts INSIDE a block window (block_s <= gap_s < block_e) —
#          the host was blocked at the moment the GPU ran dry.
#  PROXIMITY: a block ended within 2ms before gap start (launch-delay tail).
block_iv = {}
for k, wins in class_windows.items():
    if wins:
        ws = np.array(sorted(w[0] for w in wins), dtype=np.int64)
        we = np.array(sorted(w[0] + w[1] for w in wins), dtype=np.int64)
        block_iv[k] = (ws, we)
gap_attr = {k: [0, 0] for k in block_iv}
gap_attr_strict = {k: [0, 0] for k in block_iv}
gap_attr["UNATTRIBUTED"] = [0, 0]
gap_attr_strict["UNATTRIBUTED"] = [0, 0]
HORIZON = 2_000_000
for gs, gd in zip(gap_s, gap_d):
    if gd < 100_000:
        continue
    hit = None
    hit_strict = None
    for k, (ws, we) in block_iv.items():
        i = int(np.searchsorted(we, gs, side="left"))   # first block ending > gs
        if i < len(ws) and ws[i] <= gs < we[i]:
            hit_strict = k
        # proximity: nearest end within horizon before/after gs
        if i < len(we) and 0 <= we[i] - gs <= HORIZON:
            hit = hit or k
        if i > 0 and 0 <= gs - we[i - 1] <= HORIZON:
            hit = hit or k
    if hit_strict:
        gap_attr_strict[hit_strict][0] += 1; gap_attr_strict[hit_strict][1] += int(gd)
    else:
        gap_attr_strict["UNATTRIBUTED"][0] += 1; gap_attr_strict["UNATTRIBUTED"][1] += int(gd)
    if hit:
        gap_attr[hit][0] += 1; gap_attr[hit][1] += int(gd)
    else:
        gap_attr["UNATTRIBUTED"][0] += 1; gap_attr["UNATTRIBUTED"][1] += int(gd)

# launch storm: kernels strictly inside FIX-B aten::index windows
nz_index_windows.sort()
merged = []
for s_, d_ in nz_index_windows:
    e_ = s_ + d_
    if merged and s_ <= merged[-1][1]:
        if e_ > merged[-1][1]: merged[-1][1] = e_
    else:
        merged.append([s_, e_])
k_in_layout = 0
if merged:
    kk = q("SELECT ts FROM slice WHERE category='kernel' AND dur>0")
    ka = kk.ts.to_numpy(dtype=np.int64); ka.sort()
    ms = np.array([mm[0] for mm in merged]); me = np.array([mm[1] for mm in merged])
    idx = np.searchsorted(me, ka, side="left")
    ok = idx < len(ms)
    idxc = np.minimum(idx, len(ms) - 1)
    hit = ok & (ka >= ms[idxc]) & (ka < me[idxc])
    k_in_layout = int(hit.sum())

nz = q("""SELECT s.dur, p.name AS parent FROM slice s
          LEFT JOIN slice p ON s.parent_id = p.id WHERE s.name='aten::nonzero'""")

result = {
    "trace": path,
    "step_window_s": (E - S) / 1e9,
    "gpu_idle_s": gpu_idle_s,
    "kernel_count": kernel_count,
    "gap_histogram": gap_hist,
    "gap_attribution_gt0.1ms": {k: {"gaps": v[0], "idle_s": v[1] / 1e9} for k, v in gap_attr.items()},
    "gap_attribution_strict": {k: {"gaps": v[0], "idle_s": v[1] / 1e9} for k, v in gap_attr_strict.items()},
    "aten_nonzero": {"calls": int(len(nz)), "cpu_s": float(nz.dur.sum() / 1e9),
                     "layout_calls": int((nz.parent == "aten::index").sum()),
                     "gt5ms": int((nz.dur > 5e6).sum())},
    "kernels_in_layout_index_windows": k_in_layout,
    "devsync_count": len(devsync_ts),
    "devsync_cpu_s": sum(d for _, d in devsync_ts) / 1e9,
    "devsync_first_last_rel_s": ([(devsync_ts[0][0] - S) / 1e9, (devsync_ts[-1][0] - S) / 1e9]
                                 if devsync_ts else None),
    "classes": {k: {"calls": v[0], "cpu_s": v[1] / 1e9} for k, v in classes.items()},
    "other_top": [{"sig": s, "calls": other_sig_c[s], "cpu_s": t / 1e9}
                  for s, t in other_sig_t.most_common(12)],
    "elapsed_s": time.time() - t0,
}
txt = json.dumps(result, indent=1)
print(txt, flush=True)
if out_json:
    with open(out_json, "w") as f:
        f.write(txt)
