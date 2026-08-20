#!/usr/bin/env python3
"""A2A exposure decomposition at d16 (gauss, LPS-1062, borel assignment).

Per SendRecv (a2a) kernel: pass (FWD/REPLAY/BWD via correlation->parent chain),
call type (tokens BF16 / probs Float; dispatch vs combine by order-in-layer),
exposure (no concurrent compute), transfer floor (payload / wire rate), wait.
"""
import json
import sys
import time
from collections import defaultdict

import numpy as np
from perfetto.trace_processor import TraceProcessor

path = sys.argv[1]
out_json = sys.argv[2] if len(sys.argv) > 2 else None
t0 = time.time()
tp = TraceProcessor(trace=path)


def q(sql):
    return tp.query(sql).as_pandas_dataframe()


# --- step window ---
step = q("SELECT ts, dur, name FROM slice WHERE name LIKE 'ProfilerStep%' ORDER BY dur DESC")
S = float(step.ts.iloc[0]); E = S + float(step.dur.iloc[0])
print(f"[{time.time()-t0:.0f}s] step {(E-S)/1e9:.2f}s", flush=True)

# --- compute union (non-comm kernels) ---
comp = q("""SELECT ts, dur FROM slice WHERE category='kernel' AND dur>0
            AND name NOT LIKE '%SendRecv%' AND name NOT LIKE '%AllGather%'
            AND name NOT LIKE '%ReduceScatter%' AND name NOT LIKE '%AllReduce%'""")
ca = comp.ts.to_numpy(dtype=np.int64); cb = ca + comp.dur.to_numpy(dtype=np.int64)
o = np.argsort(ca, kind="stable"); ca, cb = ca[o], cb[o]
cus, cue = [], []
cs_, ce_ = int(ca[0]), int(cb[0])
for s_, e_ in zip(ca[1:], cb[1:]):
    if s_ <= ce_:
        if e_ > ce_: ce_ = int(e_)
    else:
        cus.append(cs_); cue.append(ce_); cs_, ce_ = int(s_), int(e_)
cus.append(cs_); cue.append(ce_)
cus = np.array(cus); cue = np.array(cue)
print(f"[{time.time()-t0:.0f}s] compute union built ({len(cus)} intervals)", flush=True)


def compute_overlap(s_, e_):
    i = int(np.searchsorted(cue, s_, side="right"))
    tot = 0
    while i < len(cus) and cus[i] < e_:
        tot += min(e_, cue[i]) - max(s_, cus[i])
        i += 1
    return tot


# --- a2a kernels with args ---
k = q("""
SELECT s.id, s.ts, s.dur,
       dt.display_value AS dtype,
       CAST(ine.display_value AS DOUBLE) AS nelems,
       corr.display_value AS corr
FROM slice s
JOIN args dt ON dt.arg_set_id=s.arg_set_id AND dt.key='args.dtype'
JOIN args ine ON ine.arg_set_id=s.arg_set_id AND ine.key='args.In msg nelems'
JOIN args corr ON corr.arg_set_id=s.arg_set_id AND corr.key='args.correlation'
WHERE s.category='kernel' AND s.name LIKE '%SendRecv%'
""")
print(f"[{time.time()-t0:.0f}s] a2a kernels {len(k):,}", flush=True)

# --- runtime slices + their correlation -> parent chain seed ---
rt = q("SELECT id, parent_id, ts, dur, name FROM slice WHERE category IN ('cuda_runtime','cuda_driver')")
rt_corr = q("""
SELECT s.id AS rid, corr.display_value AS corr FROM slice s
JOIN args corr ON corr.arg_set_id=s.arg_set_id AND corr.key='args.correlation'
WHERE s.category IN ('cuda_runtime','cuda_driver')
""")
corr2rid = dict(zip(rt_corr["corr"].astype(str), rt_corr.rid))
rt_parent = dict(zip(rt.id, rt.parent_id))
del rt, rt_corr

ops = q("SELECT id, parent_id, name FROM slice WHERE category IN ('cpu_op','user_annotation')")
op_name = dict(zip(ops.id, ops.name))
op_parent = dict(zip(ops.id, ops.parent_id))
del ops
print(f"[{time.time()-t0:.0f}s] trees built", flush=True)

PASS_OF = {"CheckpointFunction": "FWD", "CheckpointFunctionBackward": "REPLAY",
           "_AllToAllBackward": "BWD"}


def classify(rid):
    cur = rt_parent.get(rid)
    d = 0
    pass_ = None
    ckpt = None
    while cur and d < 80:
        n = op_name.get(cur)
        if n == "_AllToAllBackward":
            pass_ = "BWD"
        elif n == "CheckpointFunctionBackward":
            if pass_ is None:
                pass_ = "REPLAY"
            ckpt = cur
            break
        elif n == "CheckpointFunction":
            if pass_ is None:
                pass_ = "FWD"
            ckpt = cur
            break
        cur = op_parent.get(cur)
        d += 1
    return (pass_ or "UNKNOWN"), ckpt


# classify each kernel; also record enclosing checkpoint slice id for grouping
rows = []
for ts, dur, dtype, nelems, corr in zip(k.ts, k.dur, k.dtype, k.nelems, k["corr"]):
    rid = corr2rid.get(str(corr))
    pass_, ckpt = classify(rid) if rid else ("UNKNOWN", None)
    rows.append((int(ts), int(dur), dtype, float(nelems), pass_, ckpt))
print(f"[{time.time()-t0:.0f}s] classified passes", flush=True)

# pass stats before role assignment
cnt = defaultdict(int)
for r in rows:
    cnt[(r[4], r[2])] += 1
print("pass x dtype counts:", dict(cnt), flush=True)

# role assignment: group by (pass, checkpoint slice), sort by ts.
# FWD/REPLAY per layer: [BF16 dispatch, Float probs, BF16 combine]
# BWD per layer: [BF16 combine-grad, BF16 dispatch-grad, Float probs-grad]
role_rows = []
groups = defaultdict(list)
for r in rows:
    groups[(r[4], r[5])].append(r)
for (pass_, ck), g in groups.items():
    g.sort(key=lambda r: r[0])
    if pass_ in ("FWD", "REPLAY"):
        bf = [r for r in g if r[2] == "BFloat16"]
        if len(bf) >= 1:
            role_rows.append(bf[0] + ("dispatch",))
        if len(bf) >= 2:
            role_rows.append(bf[-1] + ("combine",))
        for r in g:
            if r[2] == "Float":
                role_rows.append(r + ("probs",))
    elif pass_ == "BWD":
        bf = [r for r in g if r[2] == "BFloat16"]
        if len(bf) >= 1:
            role_rows.append(bf[0] + ("combine_grad",))
        if len(bf) >= 2:
            role_rows.append(bf[-1] + ("dispatch_grad",))
        for r in g:
            if r[2] == "Float":
                role_rows.append(r + ("probs_grad",))
    else:
        for r in g:
            role_rows.append(r + ("unknown",))

# layer index: order checkpoint groups within each pass by group start, mod 40
ckpt_starts = defaultdict(list)
for (pass_, ck), g in groups.items():
    if ck is not None:
        ckpt_starts[pass_].append((min(r[0] for r in g), ck))
layer_of = {}
nlay = {}
for pass_, lst in ckpt_starts.items():
    lst.sort()
    n = max(1, round(len(lst) / 16))
    nlay[pass_] = n
    for i, (_, ck) in enumerate(lst):
        layer_of[(pass_, ck)] = i % n

# wire rate calibration: fastest decile achieved BW on big BF16 calls
bw = np.array([r[3] * 2 / r[1] for r in role_rows if r[2] == "BFloat16" and r[1] > 0])  # bytes/ns = GB/s
bw = bw * 1e9 / 1e9  # bytes/ns == GB/s
rate_floor = float(np.percentile(bw, 95))
print(f"[{time.time()-t0:.0f}s] wire rate p95 = {rate_floor:.0f} GB/s (max {bw.max():.0f})", flush=True)

# per-call exposure + transfer/wait
out = []
for ts, dur, dtype, nelems, pass_, ck, role in role_rows:
    ov = compute_overlap(ts, ts + dur)
    exposed = dur - ov
    nbytes = nelems * (2 if dtype == "BFloat16" else 4)
    floor_ns = nbytes / rate_floor  # GB/s = bytes/ns
    out.append({
        "ts": ts, "dur": dur, "dtype": dtype, "pass": pass_, "role": role,
        "layer": layer_of.get((pass_, ck), -1),
        "exposed_ns": int(exposed), "floor_ns": int(floor_ns),
        "wait_ns": int(max(0, dur - floor_ns)),
        "mb": nelems * (2 if dtype == "BFloat16" else 4) / 1e6,
    })

# aggregates
def agg(keyfn):
    acc = defaultdict(lambda: [0, 0, 0, 0, 0])
    for r in out:
        kk = keyfn(r)
        acc[kk][0] += 1
        acc[kk][1] += r["dur"]
        acc[kk][2] += r["exposed_ns"]
        acc[kk][3] += r["floor_ns"]
        acc[kk][4] += r["wait_ns"]
    return {kk: {"calls": v[0], "dur_s": v[1] / 1e9, "exposed_s": v[2] / 1e9,
                 "transfer_floor_s": v[3] / 1e9, "wait_s": v[4] / 1e9}
            for kk, v in sorted(acc.items(), key=lambda x: -x[1][2])}

by_pass_role = agg(lambda r: (r["pass"], r["role"]))
by_pass = agg(lambda r: r["pass"])
by_role = agg(lambda r: r["role"])

# per-layer wait stickiness (dispatch calls, FWD+REPLAY)
layer_wait = defaultdict(list)
for r in out:
    if r["role"] == "dispatch" and r["layer"] >= 0:
        layer_wait[r["layer"]].append(r["wait_ns"])
layer_means = {l: float(np.mean(w)) / 1e6 for l, w in sorted(layer_wait.items())}

# size vs wait correlation (dispatch BF16)
dw = [(r["mb"], r["wait_ns"]) for r in out if r["role"] == "dispatch"]
if dw:
    sizes = np.array([x[0] for x in dw]); waits = np.array([x[1] for x in dw])
    corr_size_wait = float(np.corrcoef(sizes, waits)[0, 1])
else:
    corr_size_wait = None

import numpy as _np
role_dur = defaultdict(list)
for r in out:
    role_dur[(r["pass"], r["role"])].append(r["dur"])
role_pct = {}
for kk, v in role_dur.items():
    a = _np.array(sorted(v)) / 1e6
    role_pct[str(kk)] = {"p50_ms": float(_np.percentile(a, 50)),
                         "p90_ms": float(_np.percentile(a, 90)),
                         "p99_ms": float(_np.percentile(a, 99)),
                         "max_ms": float(a[-1])}
result = {
    "trace": path,
    "layers_per_pass": nlay,
    "step_window_s": (E - S) / 1e9,
    "wire_rate_p95_gbs": rate_floor,
    "wire_rate_max_gbs": float(bw.max()),
    "total_a2a_calls": len(out),
    "total_dur_s": sum(r["dur"] for r in out) / 1e9,
    "total_exposed_s": sum(r["exposed_ns"] for r in out) / 1e9,
    "total_transfer_floor_s": sum(r["floor_ns"] for r in out) / 1e9,
    "total_wait_s": sum(r["wait_ns"] for r in out) / 1e9,
    "by_pass": {str(kk): v for kk, v in by_pass.items()},
    "by_role": {str(kk): v for kk, v in by_role.items()},
    "by_pass_role": {str(kk): v for kk, v in by_pass_role.items()},
    "layer_mean_wait_ms": layer_means,
    "dispatch_size_wait_corr": corr_size_wait,
    "role_duration_percentiles": role_pct,
    "elapsed_s": time.time() - t0,
}
txt = json.dumps(result, indent=1)
print(txt, flush=True)
if out_json:
    with open(out_json, "w") as f:
        f.write(txt)
