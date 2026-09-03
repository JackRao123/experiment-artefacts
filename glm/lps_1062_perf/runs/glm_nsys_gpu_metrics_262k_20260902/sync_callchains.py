#!/usr/bin/env python3
"""Group blocking CUDA API calls (cudaStreamSynchronize etc.) by call-chain signature.

Each call carries its own callchain row set in CUDA_CALLCHAINS (captured with
--cudabacktrace=sync). The Python frames are unresolved addresses, so the
signature is built from the resolved native frames above cudart: the libtorch /
c10 / TE / megatron frames that say which op issued the sync. Frames from cupti,
libcuda, cudart, python and libc are dropped.

Also reports, per signature, the NVTX phase (forward/backward/...) and the inner
range (layer / attention / moe / ...) the call sat in, using time containment on
the same process, so a sync can be traced to "backward > moe_dispatch".

Writes <out>/sync_callchains.csv and prints the top rows.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
from collections import defaultdict

PHASES = ["forward", "backward", "optimizer"]
INNER = ["recompute", "layer", "self_attention", "mlp", "attention", "moe", "dsa_indexer", "dsa_core", "moe_route", "moe_dispatch", "moe_experts",
         "moe_combine", "hybridep_dispatch", "hybridep_combine", "hybridep_dispatch_bwd", "hybridep_combine_bwd",
         "lm_head", "lm_head_gemm"]
SKIP_MODULE = re.compile(r"libcupti|libcuda\.so|libcudart|python3|libc\.so|libpthread|ToolsInjection|libstdc")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", default=None, help="'a,b' ns; default = span of the longest 'forward_backward' NVTX range")
    ap.add_argument("--frames", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    con = sqlite3.connect(args.sqlite)
    cur = con.cursor()
    S = {i: v for i, v in cur.execute("SELECT id, value FROM StringIds")}

    # window
    if args.window:
        W0, W1 = (int(x) for x in args.window.split(","))
    else:
        rows = cur.execute("SELECT start, end FROM NVTX_EVENTS WHERE eventType IN (59,60) AND text='forward_backward'").fetchall()
        if not rows:
            raise SystemExit("no forward_backward range; pass --window")
        W0 = min(r[0] for r in rows)
        W1 = max(r[1] for r in rows)
    print(f"window {W0} {W1} ({(W1 - W0) / 1e9:.3f} s)")

    # nvtx ranges for phase/inner attribution (normalize autograd op ids away)
    op_suffix = re.compile(r",\s*(op_id|seq)\s*=\s*\d+")
    ranges: dict[str, dict[int, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    for s, e, gtid, text, text_id in cur.execute(
            "SELECT start, end, globalTid, text, textId FROM NVTX_EVENTS WHERE eventType IN (59,60) AND end IS NOT NULL AND start>=? AND end<=?",
            (W0 - 1_000_000_000, W1 + 1_000_000_000)):
        nm = op_suffix.sub("", text if text is not None else S.get(text_id, "?")).strip()
        if nm.startswith("layer:"):
            nm = "layer"
        if nm in PHASES or nm in INNER:
            ranges[nm][(gtid >> 24) << 24].append((s, e))
    for nm in ranges:
        for pid in ranges[nm]:
            ranges[nm][pid].sort()

    def enclosing(names: list[str], pid: int, t: int) -> str:
        hits = []
        for nm in names:
            inst = ranges.get(nm, {}).get(pid)
            if not inst:
                continue
            # binary search on start
            lo, hi = 0, len(inst)
            while lo < hi:
                mid = (lo + hi) // 2
                if inst[mid][0] <= t:
                    lo = mid + 1
                else:
                    hi = mid
            # walk back a little: ranges of the same name can nest (layer inside recompute is a different name, fine)
            for j in range(lo - 1, max(-1, lo - 4), -1):
                if inst[j][0] <= t <= inst[j][1]:
                    hits.append(nm)
                    break
        return "+".join(hits) if hits else "-"

    sync_ids = [i for i, v in S.items() if v.startswith(("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize"))
                or (v.startswith("cudaMemcpy") and "Async" not in v)]
    q = (f"SELECT start, end, globalTid, nameId, callchainId FROM CUPTI_ACTIVITY_KIND_RUNTIME "
         f"WHERE nameId IN ({','.join(map(str, sync_ids))}) AND start>=? AND end<=?")
    calls = cur.execute(q, (W0, W1)).fetchall()
    # nsys records each runtime call twice (versioned + unversioned name); only one carries the callchain
    with_cc = [c for c in calls if c[4] is not None]
    if with_cc:
        calls = with_cc
    print(f"{len(calls)} blocking calls in window")
    cc_ids = sorted({c[4] for c in calls if c[4] is not None})
    frames: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
    # chunked IN queries
    for i in range(0, len(cc_ids), 5000):
        chunk = cc_ids[i:i + 5000]
        for ccid, sym, mod, depth in cur.execute(
                f"SELECT id, symbol, module, stackDepth FROM CUDA_CALLCHAINS WHERE id IN ({','.join(map(str, chunk))})"):
            symv = S.get(sym, str(sym)) if isinstance(sym, int) else str(sym)
            modv = S.get(mod, str(mod)) if isinstance(mod, int) else str(mod)
            frames[ccid].append((depth, symv, modv))

    def signature(ccid: int) -> str:
        fr = sorted(frames.get(ccid, []))
        keep = []
        for depth, sym, mod in fr:
            if SKIP_MODULE.search(os.path.basename(mod or "")):
                continue
            if sym.startswith("0x"):
                continue
            # compress the dispatcher boilerplate
            if "wrap_kernel_functor_unboxed" in sym or "callWithDispatchKeySlowPath" in sym or "::redispatch(" in sym:
                continue
            short = re.sub(r"\(.*$", "", sym)  # drop argument lists
            short = short.replace("at::native::", "").replace("at::_ops::", "ops::").replace("torch::autograd::", "autograd::")
            if keep and keep[-1] == short:
                continue
            keep.append(short)
            if len(keep) >= args.frames:
                break
        return " < ".join(keep) if keep else "(unresolved)"

    agg: dict[tuple[str, str, str, str], list[float]] = defaultdict(lambda: [0, 0.0, 0.0])
    per_gpu: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    pids = sorted({(c[2] >> 24) << 24 for c in calls})
    pid_idx = {p: i for i, p in enumerate(pids)}
    for s, e, gtid, name_id, ccid in calls:
        pid = (gtid >> 24) << 24
        sig = signature(ccid) if ccid is not None else "(no callchain)"
        ph = enclosing(PHASES, pid, s)
        inner = enclosing(INNER, pid, s)
        key = (S[name_id], sig, ph, inner)
        a = agg[key]
        a[0] += 1
        a[1] += (e - s) / 1e9
        a[2] = max(a[2], (e - s) / 1e6)
        per_gpu[sig][pid_idx[pid]] += 1
    rows = sorted(([k[0], k[1], k[2], k[3], v[0], round(v[1], 3), round(v[2], 3)] for k, v in agg.items()), key=lambda r: -r[4])
    with open(os.path.join(args.out, "sync_callchains.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["api", "signature", "phase", "inner_range", "calls_all_pids", "total_s_all_pids", "max_ms"])
        w.writerows(rows)
    # by signature only
    by_sig: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    for k, v in agg.items():
        by_sig[k[1]][0] += v[0]
        by_sig[k[1]][1] += v[1]
    n = max(1, len(pids))
    print(f"\n## Blocking syncs by call-chain signature (per GPU = all / {n})\n")
    print("| calls/GPU | blocked s/GPU | signature |\n|---|---|---|")
    for sig, (c, t) in sorted(by_sig.items(), key=lambda kv: -kv[1][0])[:25]:
        print(f"| {c / n:.0f} | {t / n:.3f} | `{sig}` |")
    print("\n## Top (signature, phase, inner range) rows\n")
    print("| calls/GPU | blocked s/GPU | max ms | phase | inner | signature |\n|---|---|---|---|---|---|")
    for r in rows[:30]:
        print(f"| {r[4] / n:.0f} | {r[5] / n:.3f} | {r[6]} | {r[2]} | {r[3]} | `{r[1][:120]}` |")


if __name__ == "__main__":
    main()
