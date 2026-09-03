#!/usr/bin/env python3
"""Deep attribution of an all-rank nsys sqlite export of one trainer step.

Outputs (CSV + one markdown summary) into --out:
  step_window.csv          per-GPU step window used
  categories_by_gpu.csv    kernel time by category per GPU inside the window
  phases_by_gpu.csv        (if NVTX phase ranges exist) GPU kernel time per phase per GPU
  ranges_summary.csv       every NVTX range name: count, wall time (per pid), GPU kernel time attributed by launch time
  layers.csv               per-layer forward/recompute range durations (if layer ranges exist)
  sync_waits.csv           every HybridEP device_sync wait >= --min-wait-ms: who was the laggard and what it was doing
  sync_wait_summary.csv    the same, aggregated
  host_syncs.csv           cudaStreamSynchronize count/time per pid/thread, and top callchains when available
  gpu_idle.csv             per-GPU time with no kernel resident, inside the window
  SUMMARY.md               human-readable summary

Attribution rule for "kernel belongs to NVTX range": the kernel's launch API call
(CUPTI runtime row with the same correlationId) started inside a range instance
of that name on the same process (any thread). This is time-on-host containment,
which is right for phase ranges (forward/backward/optimizer) and for per-layer
ranges because those never overlap in time within one process.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
import sys
from collections import defaultdict

import numpy as np

CATEGORY_RULES = [
    ("hybridep_sync", re.compile(r"hybrid_ep::device_sync_kernel")),
    ("hybridep_dispatch", re.compile(r"hybrid_ep::dispatch_kernel")),
    ("hybridep_combine", re.compile(r"hybrid_ep::combine_kernel")),
    ("dsa_backward", re.compile(r"dsa_bwd|sparse_attention_backward")),
    ("dsa_forward", re.compile(r"sparse_attn_fwd|sparse_attention_forward")),
    ("dsa_indexer", re.compile(r"indexer_forward|indexer_fwd|indexer_top_k|indexer_topk|indexer_bwd|indexer_backward")),
    ("fp32_simt_head", re.compile(r"simt_sgemm")),
    ("gemm", re.compile(r"nvjet|cutlass.*gemm|gemm_|Gemm|sm100.*mma|grouped_gemm|GroupedGemm|cublas", re.I)),
    ("nccl", re.compile(r"ncclDevKernel|ncclKernel")),
    ("moe_permute", re.compile(r"permute_kernel|unpermute_kernel|moe_permute|moe_unpermute|sort_chunks|gather_along|scatter_along")),
    ("topk_router", re.compile(r"gatherTopK|radixSelect|sbtopk|softmax|sigmoid|router|aux_loss", re.I)),
    ("cat_copy", re.compile(r"CatArrayBatchedCopy|direct_copy_kernel|copy_kernel|index_copy|index_elementwise|indexFuncLargeIndex|vectorized_gather|indexSelect|index_select|gather_kernel|scatter", re.I)),
    ("norm", re.compile(r"rmsnorm|layernorm|LayerNorm|RMSNorm", re.I)),
    ("activation", re.compile(r"silu|swiglu|gelu|Swiglu", re.I)),
    ("elementwise", re.compile(r"elementwise_kernel|vectorized_elementwise|unrolled_elementwise|reduce_kernel|fill|Fill|arange|where|masked|cast|Cast|mul_|add_", re.I)),
    ("rope", re.compile(r"rope|rotary|Rotary", re.I)),
    ("fp8_quant", re.compile(r"quantize|dequantize|fp8|FP8|amax|Amax|cast_transpose|scale", re.I)),
    ("cross_entropy", re.compile(r"cross_entropy|CrossEntropy|log_softmax|LogSoftmax|nll", re.I)),
    ("memcpy_like", re.compile(r"memcpy|Memcpy|memset|Memset", re.I)),
    ("nonzero_cub", re.compile(r"DeviceSelectSweepKernel|DeviceCompactInitKernel|DeviceReduceSingleTileKernel")),
    ("cub_scan_sort", re.compile(r"DeviceScan|DeviceRadixSort|radixSortKVInPlace|bitonicSortKVInPlace|DeviceReduceKernel")),
    ("hybridep_meta", re.compile(r"ag_nvl_kernel|permute_preprocessing_kernel|update_expected_value_kernel")),
    ("optimizer", re.compile(r"multi_tensor_apply")),
]

PHASE_NAMES = ["request", "forward", "backward", "optimizer"]
INNER_NAMES = ["layer", "attention", "moe", "hybridep_dispatch", "hybridep_combine", "hybridep_sync",
               "recompute", "lm_head", "loss", "router", "experts", "dsa_indexer", "dsa_core", "mlp"]


def categorize(name: str) -> str:
    for cat, rx in CATEGORY_RULES:
        if rx.search(name):
            return cat
    return "other"


def write_csv(path: str, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-wait-ms", type=float, default=5.0)
    ap.add_argument("--window", default="auto",
                    help="'auto' (largest dense kernel segment), 'nvtx:<name>' (span of that range on each pid), or 'a,b' ns")
    ap.add_argument("--gpu-metrics", action="store_true", help="also average GPU hardware metrics per phase (slow, scans GPU_METRICS)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    con = sqlite3.connect(args.sqlite)
    con.execute("PRAGMA temp_store=MEMORY")
    cur = con.cursor()

    strings = {i: v for i, v in cur.execute("SELECT id, value FROM StringIds")}

    # ---- kernels -------------------------------------------------------------------------
    print("loading kernels", flush=True)
    rows = cur.execute(
        "SELECT start, end, deviceId, globalPid, shortName, correlationId, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchall()
    k_start = np.array([r[0] for r in rows], dtype=np.int64)
    k_end = np.array([r[1] for r in rows], dtype=np.int64)
    k_dev = np.array([r[2] for r in rows], dtype=np.int32)
    k_pid = np.array([r[3] for r in rows], dtype=np.int64)
    k_corr = np.array([r[5] for r in rows], dtype=np.int64)
    k_name_id = np.array([r[4] for r in rows], dtype=np.int64)
    k_dem_id = np.array([r[6] for r in rows], dtype=np.int64)
    name_cache: dict[int, str] = {}
    cat_cache: dict[int, str] = {}
    for nid in np.unique(k_name_id):
        name_cache[int(nid)] = strings.get(int(nid), "?")
    # categorize on the demangled name (keeps namespaces such as hybrid_ep:: and the functor
    # inside elementwise_kernel<...>), display the short one
    for did in np.unique(k_dem_id):
        cat_cache[int(did)] = categorize(strings.get(int(did), "?"))
    k_cat = np.array([cat_cache[int(n)] for n in k_dem_id])
    k_short = np.array([name_cache[int(n)][:90] for n in k_name_id])
    del rows

    # rank/pid <-> device mapping. Each trainer process owns exactly one GPU.
    pid_to_dev: dict[int, int] = {}
    for pid in np.unique(k_pid):
        devs, cnt = np.unique(k_dev[k_pid == pid], return_counts=True)
        pid_to_dev[int(pid)] = int(devs[np.argmax(cnt)])
    dev_to_pid = {d: p for p, d in pid_to_dev.items()}
    devices = sorted(dev_to_pid)
    print("devices", devices, "pids", {d: hex(dev_to_pid[d]) for d in devices}, flush=True)

    # ---- NVTX ranges (push/pop = 59, start/end = 60) ----------------------------------------
    print("loading nvtx", flush=True)
    nvtx = cur.execute(
        "SELECT start, end, globalTid, text, textId, eventType FROM NVTX_EVENTS WHERE eventType IN (59, 60) AND end IS NOT NULL"
    ).fetchall()
    ranges: dict[str, dict[int, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    op_suffix = re.compile(r",\s*(op_id|seq)\s*=\s*\d+")
    for s, e, gtid, text, text_id, _ in nvtx:
        nm = text if text is not None else strings.get(text_id, "?")
        nm = op_suffix.sub("", nm).strip()  # collapse --pytorch=autograd-nvtx per-op ids into per-op names
        pid = (gtid >> 24) << 24
        ranges[nm][pid].append((s, e))
    del nvtx
    autograd_like = sum(1 for nm in ranges if "," in nm or nm.startswith("aten::") or "Backward" in nm)
    have_phase = any(nm in ranges for nm in PHASE_NAMES)
    print(f"nvtx range names: {len(ranges)} (autograd-like {autograd_like}); phase ranges present: {have_phase}", flush=True)

    # ---- step window per device ---------------------------------------------------------------
    windows: dict[int, tuple[int, int]] = {}
    if args.window.startswith("nvtx:"):
        nm = args.window[5:]
        for d in devices:
            inst = ranges.get(nm, {}).get(dev_to_pid[d], [])
            if not inst:
                sys.exit(f"no nvtx range {nm!r} for device {d}")
            s, e = max(inst, key=lambda x: x[1] - x[0])
            windows[d] = (s, e)
    elif "," in args.window:
        a, b = (int(x) for x in args.window.split(","))
        windows = {d: (a, b) for d in devices}
    else:
        # auto: largest kernel-dense segment separated by >2 s gaps, then one common
        # window for all GPUs (the union) so a rank that idles inside the step
        # (e.g. waiting in a long collective) is not cut short.
        for d in devices:
            m = k_dev == d
            order = np.argsort(k_start[m])
            s = k_start[m][order]
            e = np.maximum.accumulate(k_end[m][order])
            gaps = np.where(s[1:] - e[:-1] > 2_000_000_000)[0]
            bounds = np.concatenate([[0], gaps + 1, [len(s)]])
            best = max(range(len(bounds) - 1), key=lambda i: e[bounds[i + 1] - 1] - s[bounds[i]])
            windows[d] = (int(s[bounds[best]]), int(e[bounds[best + 1] - 1]))
        lo = min(w[0] for w in windows.values())
        hi = max(w[1] for w in windows.values())
        windows = {d: (lo, hi) for d in devices}
    write_csv(os.path.join(args.out, "step_window.csv"), ["gpu", "start_ns", "end_ns", "duration_s"],
              [[d, windows[d][0], windows[d][1], (windows[d][1] - windows[d][0]) / 1e9] for d in devices])
    W0 = min(w[0] for w in windows.values())
    W1 = max(w[1] for w in windows.values())
    print("window", {d: round((windows[d][1] - windows[d][0]) / 1e9, 3) for d in devices}, flush=True)

    in_win = np.zeros(len(k_start), dtype=bool)
    for d in devices:
        a, b = windows[d]
        in_win |= (k_dev == d) & (k_start >= a) & (k_end <= b)
    kw_start, kw_end, kw_dev, kw_cat, kw_short, kw_corr, kw_pid = (
        k_start[in_win], k_end[in_win], k_dev[in_win], k_cat[in_win], k_short[in_win], k_corr[in_win], k_pid[in_win])
    kw_dur = kw_end - kw_start

    # ---- categories per GPU ---------------------------------------------------------------------
    cat_rows = []
    cat_tot: dict[str, float] = defaultdict(float)
    for d in devices:
        m = kw_dev == d
        for c in np.unique(kw_cat[m]):
            mm = m & (kw_cat == c)
            tot = float(kw_dur[mm].sum()) / 1e9
            cat_tot[c] += tot
            cat_rows.append([d, c, int(mm.sum()), round(tot, 3), round(float(kw_dur[mm].max()) / 1e6, 3),
                             round(float(np.median(kw_dur[mm])) / 1e3, 1)])
    cat_rows.sort(key=lambda r: (r[0], -r[3]))
    write_csv(os.path.join(args.out, "categories_by_gpu.csv"),
              ["gpu", "category", "kernels", "summed_gpu_s", "max_ms", "median_us"], cat_rows)

    # top kernels inside "other" so the bucket is itemized
    other_rows = []
    m = kw_cat == "other"
    for nm in np.unique(kw_short[m]):
        mm = m & (kw_short == nm)
        other_rows.append([nm, int(mm.sum()), round(float(kw_dur[mm].sum()) / 1e9 / len(devices), 3),
                           round(float(kw_dur[mm].max()) / 1e6, 3)])
    other_rows.sort(key=lambda r: -r[2])
    write_csv(os.path.join(args.out, "other_kernels_per_gpu.csv"), ["kernel", "instances_all_gpus", "s_per_gpu", "max_ms"], other_rows[:60])

    # top kernels inside every category (s per GPU), so each bucket is itemized
    kbc_rows = []
    for c in np.unique(kw_cat):
        m = kw_cat == c
        for nm in np.unique(kw_short[m]):
            mm = m & (kw_short == nm)
            kbc_rows.append([c, nm, int(mm.sum()), round(float(kw_dur[mm].sum()) / 1e9 / len(devices), 3),
                             round(float(np.median(kw_dur[mm])) / 1e3, 1), round(float(kw_dur[mm].max()) / 1e6, 3)])
    kbc_rows.sort(key=lambda r: (r[0], -r[3]))
    write_csv(os.path.join(args.out, "kernels_by_category.csv"),
              ["category", "kernel", "instances_all_gpus", "s_per_gpu", "median_us", "max_ms"], kbc_rows)

    # ---- GPU idle inside window -----------------------------------------------------------------
    idle_rows = []
    for d in devices:
        m = kw_dev == d
        order = np.argsort(kw_start[m])
        s = kw_start[m][order]
        e = np.maximum.accumulate(kw_end[m][order])
        gaps = s[1:] - e[:-1]
        gaps = gaps[gaps > 0]
        a, b = windows[d]
        idle_rows.append([d, round(float(gaps.sum()) / 1e9, 3), int((gaps > 1_000_000).sum()), round(float(gaps.max()) / 1e6, 3) if len(gaps) else 0,
                          round(float(gaps.sum()) / (b - a) * 100, 2)])
    write_csv(os.path.join(args.out, "gpu_idle.csv"), ["gpu", "idle_s", "gaps_over_1ms", "max_gap_ms", "idle_pct"], idle_rows)

    # ---- launch time per kernel (for NVTX attribution) -------------------------------------------
    launch_time: dict[tuple[int, int], int] = {}
    rt_rows = []
    have_callchains = cur.execute("SELECT count(*) FROM sqlite_master WHERE name='CUDA_CALLCHAINS'").fetchone()[0] > 0
    if ranges:
        print("loading runtime launches", flush=True)
        for s, gtid, corr, name_id in cur.execute(
                "SELECT start, globalTid, correlationId, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start >= ? AND start <= ?", (W0 - 2_000_000_000, W1)):
            nm = strings.get(name_id, "")
            if "Launch" in nm or "launch" in nm:
                launch_time[((gtid >> 24) << 24, corr)] = s
    kw_launch = np.array([launch_time.get((int(p), int(c)), -1) for p, c in zip(kw_pid, kw_corr)], dtype=np.int64)
    print(f"launch time resolved for {(kw_launch >= 0).mean() * 100:.1f}% of in-window kernels", flush=True)

    def attribute(range_name: str) -> dict[int, float]:
        """GPU kernel seconds per device whose launch fell inside a range instance of this name."""
        out: dict[int, float] = {}
        for d in devices:
            inst = ranges.get(range_name, {}).get(dev_to_pid[d], [])
            if not inst:
                continue
            arr = np.array(sorted(inst), dtype=np.int64)
            m = (kw_dev == d) & (kw_launch >= 0)
            lt = kw_launch[m]
            idx = np.searchsorted(arr[:, 0], lt, side="right") - 1
            ok = (idx >= 0) & (lt <= arr[np.clip(idx, 0, len(arr) - 1), 1])
            out[d] = float(kw_dur[m][ok].sum()) / 1e9
        return out

    rs_rows = []
    for nm, per_pid in ranges.items():
        n_inst = sum(len(v) for v in per_pid.values())
        wall = {pid: sum(e - s for s, e in v) / 1e9 for pid, v in per_pid.items()}
        in_window = sum(1 for pid, v in per_pid.items() for s, e in v if s >= W0 and e <= W1)
        if in_window == 0 and nm not in PHASE_NAMES:
            continue
        gpu_s = attribute(nm) if n_inst < 2_000_000 else {}
        rs_rows.append([nm[:100], n_inst, in_window, round(float(np.mean(list(wall.values()))), 3),
                        round(float(np.mean(list(gpu_s.values()))), 3) if gpu_s else ""])
    rs_rows.sort(key=lambda r: -(r[4] if r[4] != "" else 0))
    write_csv(os.path.join(args.out, "ranges_summary.csv"),
              ["range", "instances_all_pids", "instances_in_window", "wall_s_per_pid_mean", "gpu_kernel_s_per_gpu_mean"], rs_rows[:200])

    phase_rows = []
    if have_phase:
        for nm in PHASE_NAMES + INNER_NAMES:
            if nm not in ranges:
                continue
            g = attribute(nm)
            for d in devices:
                inst = ranges[nm].get(dev_to_pid[d], [])
                wall = sum(e - s for s, e in inst if s >= W0 - 1 and e <= W1 + 1) / 1e9
                phase_rows.append([nm, d, len(inst), round(wall, 3), round(g.get(d, 0.0), 3)])
        write_csv(os.path.join(args.out, "phases_by_gpu.csv"), ["range", "gpu", "instances", "wall_s", "gpu_kernel_s"], phase_rows)

    # forward vs recompute-forward GPU time: layer:<n> instances inside the forward vs backward phase
    if have_phase and "forward" in ranges and "backward" in ranges:
        split_rows = []
        for d in devices:
            pid = dev_to_pid[d]
            fwd = ranges["forward"].get(pid, [])
            bwd = ranges["backward"].get(pid, [])
            def inside(inst, spans):
                return any(a <= inst[0] and inst[1] <= b for a, b in spans)
            layer_inst = [(nm, s, e) for nm in ranges if re.match(r"^layer[:_ ]?\d+", nm) for s, e in ranges[nm].get(pid, [])]
            arr_f = np.array(sorted((s, e) for nm, s, e in layer_inst if inside((s, e), fwd)), dtype=np.int64).reshape(-1, 2)
            arr_b = np.array(sorted((s, e) for nm, s, e in layer_inst if inside((s, e), bwd)), dtype=np.int64).reshape(-1, 2)
            m = (kw_dev == d) & (kw_launch >= 0)
            lt = kw_launch[m]
            for label, arr in (("layer_forward", arr_f), ("layer_recompute_forward", arr_b)):
                if len(arr) == 0:
                    continue
                idx = np.searchsorted(arr[:, 0], lt, side="right") - 1
                ok = (idx >= 0) & (lt <= arr[np.clip(idx, 0, len(arr) - 1), 1])
                gpu_s = float(kw_dur[m][ok].sum()) / 1e9
                wall = float((arr[:, 1] - arr[:, 0]).sum()) / 1e9
                split_rows.append([label, d, len(arr), round(wall, 3), round(gpu_s, 3)])
            bw = np.array(sorted(bwd), dtype=np.int64).reshape(-1, 2)
            idx = np.searchsorted(bw[:, 0], lt, side="right") - 1
            in_bwd = (idx >= 0) & (lt <= bw[np.clip(idx, 0, len(bw) - 1), 1])
            if len(arr_b):
                idx2 = np.searchsorted(arr_b[:, 0], lt, side="right") - 1
                in_rc = (idx2 >= 0) & (lt <= arr_b[np.clip(idx2, 0, len(arr_b) - 1), 1])
            else:
                in_rc = np.zeros_like(in_bwd)
            split_rows.append(["backward_minus_recompute_forward", d, len(bw), "", round(float(kw_dur[m][in_bwd & ~in_rc].sum()) / 1e9, 3)])
        write_csv(os.path.join(args.out, "fwd_recompute_split.csv"), ["what", "gpu", "instances", "wall_s", "gpu_kernel_s"], split_rows)
        phase_rows += split_rows

    # per-layer ranges: names like "layer:12" or "layer 12"
    layer_rows = []
    layer_names = sorted([nm for nm in ranges if re.match(r"^layer[:_ ]?\d+", nm)],
                         key=lambda x: int(re.findall(r"\d+", x)[0]))
    for nm in layer_names:
        for d in devices:
            for s, e in ranges[nm].get(dev_to_pid[d], []):
                if s >= W0 and e <= W1:
                    layer_rows.append([nm, d, s, round((e - s) / 1e6, 3)])
    if layer_rows:
        write_csv(os.path.join(args.out, "layers.csv"), ["layer", "gpu", "start_ns", "wall_ms"], layer_rows)

    # ---- HybridEP sync-wait attribution -----------------------------------------------------------
    print("sync-wait attribution", flush=True)
    per_dev_sync: dict[int, np.ndarray] = {}
    per_dev_busy: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for d in devices:
        m = kw_dev == d
        order = np.argsort(kw_start[m])
        per_dev_busy[d] = (kw_start[m][order], kw_end[m][order], kw_cat[m][order], kw_short[m][order])
        ms = m & (kw_cat == "hybridep_sync")
        per_dev_sync[d] = np.stack([kw_start[ms], kw_end[ms]], axis=1) if ms.any() else np.zeros((0, 2), dtype=np.int64)

    def overlap_by_cat(d: int, a: int, b: int) -> dict[str, float]:
        s, e, c, _ = per_dev_busy[d]
        lo = np.searchsorted(e, a, side="right")
        hi = np.searchsorted(s, b, side="left")
        out: dict[str, float] = defaultdict(float)
        for i in range(lo, hi):
            ov = min(e[i], b) - max(s[i], a)
            if ov > 0:
                out[str(c[i])] += ov
        return out

    def range_at(d: int, t: int, names: list[str]) -> str:
        pid = dev_to_pid[d]
        hits = []
        for nm in names:
            for s, e in ranges.get(nm, {}).get(pid, []):
                if s <= t <= e:
                    hits.append(nm)
                    break
        return "|".join(hits)

    min_wait = int(args.min_wait_ms * 1e6)
    sw_rows = []
    agg_lag: dict[tuple[int, int], float] = defaultdict(float)
    agg_cat: dict[str, float] = defaultdict(float)
    agg_kind: dict[str, float] = defaultdict(float)
    for d in devices:
        for s, e in per_dev_sync[d]:
            if e - s < min_wait:
                continue
            dur = e - s
            # laggard = GPU spending the smallest fraction of this window inside its own sync kernel
            frac = {}
            for h in devices:
                if h == d:
                    continue
                ov = overlap_by_cat(h, s, e)
                frac[h] = ov.get("hybridep_sync", 0.0) / dur
            lag = min(frac, key=frac.get)
            lag_ov = overlap_by_cat(lag, s, e)
            busy = sum(v for k, v in lag_ov.items() if k != "hybridep_sync")
            lag_busy_frac = busy / dur
            top = sorted(((k, v) for k, v in lag_ov.items() if k != "hybridep_sync"), key=lambda kv: -kv[1])[:3]
            kind = "laggard_gpu_busy" if lag_busy_frac > 0.6 else ("laggard_gpu_idle(host)" if lag_busy_frac < 0.3 else "mixed")
            phase = range_at(lag, (s + e) // 2, PHASE_NAMES + INNER_NAMES) if have_phase else ""
            sw_rows.append([d, s, round(dur / 1e6, 3), lag, round(frac[lag], 3), round(lag_busy_frac, 3), kind,
                            ";".join(f"{k}={v / 1e6:.1f}ms" for k, v in top), phase])
            agg_lag[(d, lag)] += dur / 1e9
            agg_kind[kind] += dur / 1e9
            for k, v in top[:1]:
                agg_cat[k] += dur / 1e9
    write_csv(os.path.join(args.out, "sync_waits.csv"),
              ["waiting_gpu", "start_ns", "wait_ms", "laggard_gpu", "laggard_sync_frac", "laggard_busy_frac", "kind",
               "laggard_top_categories", "laggard_nvtx"], sw_rows)
    sws = [["by_laggard", f"gpu{l}", round(sum(v for (w, ll), v in agg_lag.items() if ll == l), 3)] for l in devices]
    sws += [["by_kind", k, round(v, 3)] for k, v in sorted(agg_kind.items(), key=lambda kv: -kv[1])]
    sws += [["by_laggard_top_category", k, round(v, 3)] for k, v in sorted(agg_cat.items(), key=lambda kv: -kv[1])]
    total_long = sum(r[2] for r in sw_rows) / 1e3
    total_sync = cat_tot.get("hybridep_sync", 0.0)
    sws.append(["total", f"long_waits>={args.min_wait_ms}ms_all_gpus_s", round(total_long, 3)])
    sws.append(["total", "all_sync_all_gpus_s", round(total_sync, 3)])
    write_csv(os.path.join(args.out, "sync_wait_summary.csv"), ["group", "key", "wait_s_all_gpus"], sws)

    # ---- host syncs --------------------------------------------------------------------------------
    print("host syncs", flush=True)
    sync_name_ids = [i for i, v in strings.items() if v.startswith("cudaStreamSynchronize") or v.startswith("cudaDeviceSynchronize")
                     or v.startswith("cudaEventSynchronize") or v.startswith("cuStreamSynchronize") or v.startswith("cudaMemcpy") and "Async" not in v]
    hs_rows = []
    cc_rows = []
    if sync_name_ids:
        q = ("SELECT globalTid, nameId, count(*), sum(end-start), max(end-start) FROM CUPTI_ACTIVITY_KIND_RUNTIME "
             f"WHERE nameId IN ({','.join(str(i) for i in sync_name_ids)}) AND start >= ? AND end <= ? GROUP BY globalTid, nameId")
        for gtid, nid, n, tot, mx in cur.execute(q, (W0, W1)):
            pid = (gtid >> 24) << 24
            if have_callchains and strings[nid].endswith("_v3020"):
                continue  # duplicate row of the same call without the callchain
            hs_rows.append([pid_to_dev.get(pid, -1), hex(gtid), strings[nid], n, round(tot / 1e9, 3), round(mx / 1e6, 3)])
        hs_rows.sort(key=lambda r: (r[0], -r[4]))
        if have_callchains:
            q = ("SELECT r.callchainId, count(*), sum(r.end-r.start) FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                 f"WHERE r.nameId IN ({','.join(str(i) for i in sync_name_ids)}) AND r.start >= ? AND r.end <= ? AND r.callchainId IS NOT NULL "
                 "GROUP BY r.callchainId ORDER BY count(*) DESC LIMIT 40")
            for ccid, n, tot in cur.execute(q, (W0, W1)):
                frames = cur.execute("SELECT symbol, module, stackDepth FROM CUDA_CALLCHAINS WHERE id=? ORDER BY stackDepth", (ccid,)).fetchall()
                sym = " < ".join((strings.get(f[0], str(f[0])) if isinstance(f[0], int) else str(f[0]))[:60] for f in frames[:12])
                cc_rows.append([ccid, n, round(tot / 1e9, 3), sym])
    write_csv(os.path.join(args.out, "host_syncs.csv"), ["gpu", "globalTid", "api", "calls", "total_s", "max_ms"], hs_rows)
    if cc_rows:
        write_csv(os.path.join(args.out, "host_sync_callchains.csv"), ["callchainId", "calls", "total_s", "frames"], cc_rows)

    # ---- GPU metrics per phase (optional, slow) --------------------------------------------------------
    gm_rows = []
    if args.gpu_metrics:
        print("gpu metrics", flush=True)
        meta = cur.execute("SELECT typeId, metricId, metricName FROM TARGET_INFO_GPU_METRICS").fetchall()
        type_ids = sorted({t for t, _, _ in meta})  # one typeId per GPU, in device order
        wanted = ("sm active", "sms active", "sm issue", "tensor active", "dram read", "dram write", "gr active", "nvlink")
        metric_name = {(t, m): n for t, m, n in meta if any(w in n.lower() for w in wanted)}
        phase_windows = {"step": (W0, W1)}
        if have_phase:
            for nm in PHASE_NAMES:
                inst = [max(v, key=lambda x: x[1] - x[0]) for pid, v in ranges.get(nm, {}).items() if pid in pid_to_dev]
                if inst:
                    phase_windows[nm] = (min(s for s, e in inst), max(e for s, e in inst))
        for ph, (a, b) in phase_windows.items():
            for t, m, v in cur.execute(
                    "SELECT typeId, metricId, avg(value) FROM GPU_METRICS WHERE timestamp BETWEEN ? AND ? GROUP BY typeId, metricId", (a, b)):
                if (t, m) in metric_name:
                    d = devices[type_ids.index(t)] if t in type_ids and type_ids.index(t) < len(devices) else t
                    gm_rows.append([ph, d, metric_name[(t, m)], round(v or 0.0, 2)])
        write_csv(os.path.join(args.out, "gpu_metrics_by_phase.csv"), ["phase", "gpu", "metric", "avg"], gm_rows)

    # ---- summary -------------------------------------------------------------------------------------
    n = len(devices)
    lines = ["# Attribution summary", "", f"sqlite: `{os.path.basename(args.sqlite)}`", "",
             "## Step window per GPU (s)", "",
             ", ".join(f"gpu{d}: {(windows[d][1] - windows[d][0]) / 1e9:.3f}" for d in devices), "",
             "## Kernel time by category (mean per GPU, s)", "", "| category | s/GPU | % of window |", "|---|---|---|"]
    mean_win = np.mean([(windows[d][1] - windows[d][0]) for d in devices]) / 1e9
    for c, tot in sorted(cat_tot.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {c} | {tot / n:.3f} | {tot / n / mean_win * 100:.1f} |")
    lines += ["", "## GPU idle (no kernel resident) inside window", "", "| gpu | idle s | idle % | gaps >1ms | max gap ms |", "|---|---|---|---|---|"]
    for r in idle_rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[4]} | {r[2]} | {r[3]} |")
    if phase_rows:
        lines += ["", "## NVTX phases (wall s on host thread / GPU kernel s attributed by launch time), mean over GPUs", "",
                  "| range | instances/GPU | wall s | gpu kernel s |", "|---|---|---|---|"]
        by = defaultdict(list)
        for nm, d, inst, wall, g in phase_rows:
            by[nm].append((inst, wall, g))
        for nm, v in by.items():
            walls = [x[1] for x in v if x[1] != ""]
            lines.append(f"| {nm} | {np.mean([x[0] for x in v]):.0f} | {np.mean(walls) if walls else float('nan'):.3f} | {np.mean([x[2] for x in v]):.3f} |")
    lines += ["", f"## HybridEP sync waits >= {args.min_wait_ms} ms: who is everyone waiting for?", "",
              f"long waits total (all GPUs): {total_long:.2f} s of {total_sync:.2f} s total sync time", "",
              "| group | key | wait s (all GPUs) |", "|---|---|---|"]
    for r in sws:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} |")
    lines += ["", "## Host-side blocking API calls inside window (per GPU)", "", "| gpu | api | calls | total s | max ms |", "|---|---|---|---|---|"]
    agg_hs = defaultdict(lambda: [0, 0.0, 0.0])
    for g, tid, api, calls, tot, mx in hs_rows:
        a = agg_hs[(g, api)]
        a[0] += calls
        a[1] += tot
        a[2] = max(a[2], mx)
    for (g, api), (calls, tot, mx) in sorted(agg_hs.items()):
        lines.append(f"| {g} | {api} | {calls} | {tot:.3f} | {mx:.3f} |")
    if cc_rows:
        lines += ["", "## Top call chains for blocking syncs", ""]
        for ccid, nn, tot, sym in cc_rows[:15]:
            lines.append(f"- {nn} calls, {tot:.3f} s: `{sym}`")
    else:
        lines += ["", "(no CUDA API callchains in this export; capture with --cudabacktrace=sync to get them)"]
    lines += ["", "## Top 'other' kernels (s per GPU)", "", "| kernel | instances (all GPUs) | s/GPU | max ms |", "|---|---|---|---|"]
    for r in other_rows[:25]:
        lines.append(f"| `{r[0][:70]}` | {r[1]} | {r[2]} | {r[3]} |")
    if gm_rows:
        lines += ["", "## GPU hardware metrics by phase (mean over GPUs)", "", "| phase | metric | avg |", "|---|---|---|"]
        by = defaultdict(list)
        for ph, d, mname, v in gm_rows:
            by[(ph, mname)].append(v)
        for (ph, mname), v in by.items():
            lines.append(f"| {ph} | {mname} | {np.mean(v):.2f} |")
    with open(os.path.join(args.out, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
