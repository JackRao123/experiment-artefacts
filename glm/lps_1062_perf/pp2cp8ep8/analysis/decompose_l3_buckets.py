"""L3 / SS3b bucket decomposition of a kineto trace (doppler, LPS-1062).

Buckets (per E2_REVISED_SPEC SS3b), all reported absolute + % of window:
  fill/drain parks   = p2p SendRecv (batched isend/irecv, no collective metadata)
  a2a exposure       = EP all_to_allv comm time with NO concurrent compute
  cpu-blocked crit   = blocking host ops (cuda_runtime sync/blocking-memcpy calls)
                       intersected with GPU-empty time
  optim+tail         = optimizer kernels + trailing gap after last GPU event
  other              = remainder
GPU-busy union includes cat=kernel AND cat=gpu_memcpy (device-side copies).
Host blocking ops come from cat=cuda_runtime/cuda_driver (NOT cpu_op).
"""
import json, sys
from collections import defaultdict

path = sys.argv[1]
with open(path) as f:
    trace = json.load(f)
events = trace["traceEvents"] if isinstance(trace, dict) else trace

kernels = []       # (ts, te, name, kind)
host_blocks = []   # (ts, te, name)
host_counts = defaultdict(int)
host_durs = defaultdict(int)

def classify_comm(e, low):
    args = e.get("args") or {}
    coll = str(args.get("Collective name", "")).lower()
    if "all_to_all" in coll:
        return "a2a"
    if "allgather" in low or "all_gather" in coll or "allgather" in coll:
        return "allgather"
    if "allreduce" in low or "allreduce" in coll:
        return "allreduce"
    if "sendrecv" in low:
        return "p2p" if not coll else "othercomm"
    return "othercomm"

BLOCK_HOST = ("streamsynchronize", "devicesynchronize", "eventsynchronize")

for e in events:
    if e.get("ph") != "X":
        continue
    cat = e.get("cat", "")
    name = e.get("name", "")
    ts, dur = e["ts"], e["dur"]
    low = name.lower()
    if cat == "kernel":
        kind = classify_comm(e, low) if "nccl" in low else "comp"
        kernels.append((ts, ts + dur, name, kind))
    elif cat == "gpu_memcpy":
        kernels.append((ts, ts + dur, name, "comp"))
    elif cat in ("cuda_runtime", "cuda_driver"):
        if (any(b in low for b in BLOCK_HOST)
                or ("memcpy" in low and "async" not in low)):
            host_blocks.append((ts, ts + dur, name))
            host_counts[name] += 1
            host_durs[name] += dur

def union(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out

def total(iv):
    return sum(e - s for s, e in iv)

def intersect_total(a, b):
    i = j = 0
    t = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0]); e = min(a[i][1], b[j][1])
        if s < e:
            t += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return t

if not kernels:
    print("no kernel events"); sys.exit(1)

w0 = min(k[0] for k in kernels); w1 = max(k[1] for k in kernels)
window = w1 - w0

comp_iv = union([(s, e) for s, e, n, k in kernels if k == "comp"])
comm_kinds = ("a2a", "p2p", "allgather", "allreduce", "othercomm")
comm_iv = union([(s, e) for s, e, n, k in kernels if k in comm_kinds])
gpu_iv = union([(s, e) for s, e, n, k in kernels])
a2a_iv = union([(s, e) for s, e, n, k in kernels if k == "a2a"])
p2p_iv = union([(s, e) for s, e, n, k in kernels if k == "p2p"])
allg_iv = union([(s, e) for s, e, n, k in kernels if k == "allgather"])
allr_iv = union([(s, e) for s, e, n, k in kernels if k == "allreduce"])
optim_iv = union([(s, e) for s, e, n, k in kernels
                  if "adam" in n.lower() or "multi_tensor" in n.lower()])

overlap = intersect_total(comp_iv, comm_iv)
a2a_exposure = total(a2a_iv) - intersect_total(a2a_iv, comp_iv)

empty = []
cur = w0
for s, e in gpu_iv:
    if s > cur:
        empty.append([cur, s])
    cur = max(cur, e)
if cur < w1:
    empty.append([cur, w1])

hb_iv = union([(s, e) for s, e, n in host_blocks])
cpu_blocked_crit = intersect_total(hb_iv, empty)

p2p_raw = sorted(((e - s, n) for s, e, n, k in kernels if k == "p2p"), reverse=True)

print(f"window: {window/1e6:.2f}s  (gpu events: {len(kernels)})")
print(f"gpu-busy union: {total(gpu_iv)/1e6:.2f}s ({100*total(gpu_iv)/window:.1f}%)")
print(f"compute union:  {total(comp_iv)/1e6:.2f}s ({100*total(comp_iv)/window:.1f}%)")
print(f"comm union:     {total(comm_iv)/1e6:.2f}s ({100*total(comm_iv)/window:.1f}%)")
print(f"compute/comm overlap: {overlap/1e6:.2f}s ({100*overlap/window:.1f}%)")
print()
print("=== SS3b buckets (% of window) ===")
print(f"fill/drain parks (p2p union): {total(p2p_iv)/1e6:.2f}s ({100*total(p2p_iv)/window:.1f}%)  n={sum(1 for *_, k in kernels if k == 'p2p')}")
print(f"a2a exposure (no compute):    {a2a_exposure/1e6:.2f}s ({100*a2a_exposure/window:.1f}%)  a2a_total={total(a2a_iv)/1e6:.2f}s n={sum(1 for *_, k in kernels if k == 'a2a')}")
print(f"cpu-blocked crit (host&empty): {cpu_blocked_crit/1e6:.2f}s ({100*cpu_blocked_crit/window:.1f}%)")
print(f"optim kernels:                {total(optim_iv)/1e6:.2f}s ({100*total(optim_iv)/window:.1f}%)")
print(f"allgather union: {total(allg_iv)/1e6:.2f}s | allreduce union: {total(allr_iv)/1e6:.2f}s")
print(f"gpu-empty total: {total(empty)/1e6:.2f}s ({100*total(empty)/window:.1f}%)")
print()
print("=== host blocking ops (cuda_runtime/driver) ===")
for n, c in sorted(host_counts.items(), key=lambda kv: -host_durs[kv[0]]):
    print(f"  {n}: n={c} total={host_durs[n]/1e6:.2f}s")
print()
print("=== top p2p park durations (s) ===")
for d, n in p2p_raw[:8]:
    print(f"  {d/1e6:.2f}  {n[:80]}")
