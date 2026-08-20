#!/usr/bin/env python3
"""copy_exposure.py — the pinned-memory COPY class for the trace classifier
(LPS-1062 activation-placement ladder; hilbert spec, godel owns).

The offload arms (2c/3a-3c) move activations GPU<->CPU over PCIe as pinned
memcpys on their own CUDA streams. This tool measures EXPOSED copy stall:
time the main compute stream sits idle while copy streams are busy, after the
higher-priority comm classes have claimed their share of the idle.

Pipeline (hilbert's spec):
  (a) extract.py has already streamed the kineto JSON into TSVs (kernels,
      ann_gpu = NCCL collectives on GPU streams, runtime_big = cuda_runtime
      calls > 500 us).
  (b) Main compute stream = the tid with the most kernels (auto-detected;
      it was tid 7 on the reference trace). Merge its intervals into a busy
      union; the complement over the window is the idle set.
  (c) Busy union per class: a2a = nccl:all_to_all annotations; p2p =
      nccl:coalesced annotations; CP = AllGather/ReduceScatter kernels;
      COPY = kernels.tsv rows with cat=gpu_memcpy and 'Pinned' in the name
      ('Memcpy DtoH (Device -> Pinned)' / 'Memcpy HtoD (Pinned -> Device)'),
      on ANY stream (they live on their own).
  (d) Each compute-idle interval is attributed HIERARCHICALLY by
      intersection, in priority order: a2a -> p2p -> CP -> copy -> pure idle.
      Exposed-copy = idle claimed by copy after the higher classes took
      theirs (no double-counting).
  (e) Phase segmentation via the attention anchors, exactly as
      recompute_cost.py does: F blocks of --fwd-anchors consecutive
      sparse_attn_fwd_kernel events (rank 0: 38, rank 8: 40) and B phases of
      --bwd-anchors dsa_bwd_sm100 events (rank 0: 152 = 38x4, rank 8: 160).
      Rank 8's phases include the loss window between forward and backward.

Report: per-phase and per-class exposure, PLUS the seam windows — forward
tail (last F-anchor end -> F-block wall end) and backward head (B-phase start
-> first B-anchor start) — where the last-stage timing hazard would show up.

CROSS-CHECK (hilbert): cudaStreamSynchronize + cudaEventSynchronize calls
>500 us in runtime_big.tsv should ROUGHLY match the exposed-copy total. If
they do not, distrust the classifier before distrusting the result. (On a
no-offload trace the copy class is legitimately ZERO — the sync total then
measures other host stalls and the check is vacuous for copy; the tool says
so rather than forcing a comparison.)

SELF-TEST (the pre-verification): against the reference trace
fe127_d16_rank0 (no offload), the copy class must find ZERO events — zero,
not a crash, not phantom events.

Usage: copy_exposure.py <extract_dir> [--fwd-anchors 38] [--bwd-anchors 152]
"""
from __future__ import annotations

import argparse
import sys
from bisect import bisect_left
from collections import Counter, defaultdict

FWD_SIG = "sparse_attn_fwd_kernel"
BWD_SIG = "dsa_bwd_sm100"
COPY_NEEDLE = "Pinned"  # 'Memcpy DtoH (Device -> Pinned)' / 'HtoD (Pinned -> Device)'


# ---------- interval helpers (merged, sorted, non-overlapping) ----------
def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def total(iv):
    return sum(e - s for s, e in iv)


def inter(a, b):
    """intersection interval list of two merged lists"""
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            out.append([s, e])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract(a, b):
    """a minus b (both merged, sorted) -> merged remainder. O(n+m) sweep —
    the naive per-interval carve is too slow at campaign trace sizes."""
    out = []
    j = 0
    nb = len(b)
    for s, e in a:
        cur = s
        while j < nb and b[j][1] <= cur:
            j += 1
        k = j
        while k < nb and b[k][0] < e:
            bs, be = b[k]
            if bs > cur:
                out.append([cur, min(bs, e)])
            cur = max(cur, be)
            if cur >= e:
                break
            k += 1
        if cur < e:
            out.append([cur, e])
    return merge(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("extract_dir")
    ap.add_argument("--fwd-anchors", type=int, default=38,
                    help="F-anchor count per F block (rank 0 = 38, rank 8 = 40)")
    ap.add_argument("--bwd-anchors", type=int, default=152,
                    help="B-anchor count per B phase (rank 0 = 152, rank 8 = 160)")
    args = ap.parse_args()
    d = args.extract_dir

    # ---- load kernels (ts, dur, pid, tid, cat, name) ----
    rows = []
    with open(f"{d}/kernels.tsv") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            rows.append((float(p[0]), float(p[1]), int(p[2]), int(p[3]), p[4], p[5]))
    if not rows:
        sys.exit("kernels.tsv is empty — bad extract")
    rows.sort(key=lambda r: r[0])
    t0 = rows[0][0]
    tend = max(r[0] + r[1] for r in rows)
    wall = tend - t0

    # ---- (b) main compute stream = tid with the most kernels ----
    by_tid = Counter(r[3] for r in rows if r[4] == "kernel")
    if not by_tid:
        sys.exit("no kernel rows in kernels.tsv — bad extract")
    main_tid, main_n = by_tid.most_common(1)[0]
    print(f"window {wall/1e6:.2f}s | main compute stream: tid {main_tid} "
          f"({main_n} kernels; auto-detected, reference trace had tid 7)")

    compute = merge([(r[0], r[0] + r[1]) for r in rows if r[3] == main_tid])
    idle = []
    prev = t0
    for s, e in compute:
        if s > prev:
            idle.append([prev, s])
        prev = max(prev, e)
    if tend > prev:
        idle.append([prev, tend])
    idle = merge(idle)
    print(f"compute busy {total(compute)/1e6:.2f}s | idle {total(idle)/1e6:.2f}s "
          f"({total(idle)/wall*100:.1f}% of window)")

    # ---- (c) class busy unions ----
    a2a, p2p = [], []
    with open(f"{d}/ann_gpu.tsv") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            nm = p[4]
            if nm == "nccl:all_to_all":
                a2a.append((float(p[0]), float(p[0]) + float(p[1])))
            elif nm == "nccl:coalesced":
                p2p.append((float(p[0]), float(p[0]) + float(p[1])))
    cp = [(r[0], r[0] + r[1]) for r in rows
          if r[4] == "kernel" and ("AllGather" in r[5] or "ReduceScatter" in r[5])]
    copy_ev = [(r[0], r[0] + r[1], r[3], r[5]) for r in rows
               if r[4] == "gpu_memcpy" and COPY_NEEDLE in r[5]]

    classes = {
        "a2a": merge(a2a),
        "p2p": merge(p2p),
        "cp": merge(cp),
        "copy": merge([(s, e) for s, e, _, _ in copy_ev]),
    }
    n_copy = len(copy_ev)
    copy_dirs = Counter(x[3].split(" ")[1] for x in copy_ev) if copy_ev else {}
    # Duration split (kernels.tsv has no size column): offload traffic is
    # GiB-scale = long copies; the baseline path emits a background of tiny
    # pinned DtoH copies (metrics/host-cache class — 66k on the reference
    # trace, fully hidden). Report both so the offload arms' copy numbers can
    # be read as a delta against background.
    print(f"\nclass events: a2a={len(a2a)} p2p={len(p2p)} cp={len(cp)} "
          f"copy={n_copy} {dict(copy_dirs) if copy_dirs else ''}")
    if n_copy == 0:
        print("COPY CLASS: 0 events — CORRECT for a no-offload trace "
              "(the reference-trace self-test). On an offload arm, zero here "
              "means the offload never emitted pinned copies = investigate, "
              "do not trust a 'free offload' read.")
    else:
        # kernels.tsv has no size column; duration split instead
        big = [x for x in copy_ev if (x[1] - x[0]) >= 100.0]  # >=100us copies
        print(f"copy duration: total {sum(x[1]-x[0] for x in copy_ev)/1e6:.2f}s "
              f"resident | >=100us copies: {len(big)} "
              f"({sum(x[1]-x[0] for x in big)/1e6:.2f}s) — the offload "
              f"traffic class; the rest is baseline background")

    # ---- (d) hierarchical idle attribution: a2a -> p2p -> CP -> copy -> pure ----
    claimed: dict[str, list] = {}
    rem = idle
    for name in ("a2a", "p2p", "cp", "copy"):
        claimed[name] = inter(rem, classes[name])
        rem = subtract(rem, classes[name])
    claimed["pure_idle"] = rem

    print("\n=== compute-idle attribution (hierarchical, no double-count) ===")
    for name in ("a2a", "p2p", "cp", "copy", "pure_idle"):
        print(f"  {name:10s} {total(claimed[name])/1e6:8.2f}s "
              f"({total(claimed[name])/max(total(idle),1e-9)*100:5.1f}% of idle)")

    # ---- (e) phase segmentation via attention anchors ----
    attn = [(r[0], r[0] + r[1], "F" if FWD_SIG in r[5] else "B")
            for r in rows if r[3] == main_tid and (FWD_SIG in r[5] or BWD_SIG in r[5])]
    attn.sort()
    phases = []  # (kind, start, end, nF, nB, last_F_end, first_B_start)
    cur = None
    for s, e, k in attn:
        if cur is None:
            cur = {"start": s, "end": e, "nF": 0, "nB": 0,
                   "lastF": None, "firstB": None}
        cur["end"] = max(cur["end"], e)
        cur["nF"] += k == "F"
        cur["nB"] += k == "B"
        if k == "F":
            cur["lastF"] = e
        elif cur["firstB"] is None:
            cur["firstB"] = s
        if cur["nB"] == 0 and cur["nF"] == args.fwd_anchors:
            phases.append(("F", cur)); cur = None
        elif cur["nB"] == args.bwd_anchors:
            phases.append(("B", cur)); cur = None
    leftover = ""
    if cur is not None:
        leftover = f" (leftover partial phase: nF={cur['nF']} nB={cur['nB']} — "\
                   f"expected multiples of {args.fwd_anchors}/{args.bwd_anchors}; "\
                   "check the anchor args for this rank)"
    print(f"\nphases: {''.join(k for k, _ in phases)}{leftover} "
          f"(anchors F={sum(1 for a in attn if a[2]=='F')} B={sum(1 for a in attn if a[2]=='B')})")

    def exposure_in(a, b):
        """per-class claimed exposure inside window [a,b]"""
        return {name: total(inter([[a, b]], iv)) for name, iv in claimed.items()}

    print("\n=== per-phase exposure (seconds; idle claimed by each class) ===")
    hdr = ("phase", "wall", "idle", "a2a", "p2p", "cp", "copy", "pure")
    print("{:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(*hdr))
    for i, (k, ph) in enumerate(phases):
        a, b = ph["start"], ph["end"]
        ex = exposure_in(a, b)
        win_idle = total(inter([[a, b]], idle))
        print(f"{k}{i:<5} {(b-a)/1e6:8.2f} {win_idle/1e6:8.2f} "
              + " ".join(f"{ex[n]/1e6:8.2f}" for n in ("a2a", "p2p", "cp", "copy", "pure_idle")))

    print("\n=== seam windows (forward tail / backward head) ===")
    print("{:>6} {:>6} {:>10} {:>10} {:>10} {:>10}".format(
        "phase", "seam", "win_s", "copy_s", "a2a_s", "p2p_s"))
    for i, (k, ph) in enumerate(phases):
        if k == "F" and ph["lastF"]:
            # forward tail = last F-anchor end -> NEXT phase's start (the
            # trailing MoE/loss region; the block's own wall end is the last
            # anchor's end by construction, which would be empty)
            nxt = phases[i + 1][1]["start"] if i + 1 < len(phases) else tend
            a, b = ph["lastF"], nxt
            ex = exposure_in(a, b)
            print(f"F{i:<5} {'tail':>6} {(b-a)/1e6:10.3f} {ex['copy']/1e6:10.3f} "
                  f"{ex['a2a']/1e6:10.3f} {ex['p2p']/1e6:10.3f}")
        if k == "B" and ph["firstB"]:
            a, b = ph["start"], ph["firstB"]
            ex = exposure_in(a, b)
            print(f"B{i:<5} {'head':>6} {(b-a)/1e6:10.3f} {ex['copy']/1e6:10.3f} "
                  f"{ex['a2a']/1e6:10.3f} {ex['p2p']/1e6:10.3f}")

    # ---- cross-check vs long host syncs (hilbert) ----
    try:
        sync_tot = 0.0
        sync_n = 0
        with open(f"{d}/runtime_big.tsv") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if "StreamSynchronize" in p[3] or "EventSynchronize" in p[3]:
                    sync_tot += float(p[1])
                    sync_n += 1
        exposed_copy = total(claimed["copy"])
        print(f"\n=== cross-check ===")
        print(f"long host syncs (>500us): n={sync_n} total {sync_tot/1e6:.2f}s | "
              f"exposed copy {exposed_copy/1e6:.2f}s")
        if exposed_copy < 0.1e6:  # 100 ms, in trace units (us)
            # Below 100 ms the exposure is rounding-level; the sync total on a
            # no-offload trace measures OTHER host stalls (the reference trace
            # carries ~45 s of them), so the ratio is meaningless there.
            print(f"exposed copy ≈ 0 ({exposed_copy/1e3:.2f} ms): no-offload "
                  "trace; the sync total measures OTHER host stalls and the "
                  "copy cross-check is vacuous here (by design). On an "
                  "offload arm, these two numbers should roughly match — if "
                  "they do not, distrust the classifier before distrusting "
                  "the result.")
        else:
            ratio = sync_tot / exposed_copy
            verdict = "ROUGHLY MATCHES" if 0.5 <= ratio <= 2.0 else "MISMATCH — investigate the classifier first"
            print(f"sync/copy ratio {ratio:.2f} -> {verdict}")
    except FileNotFoundError:
        print("\n!! runtime_big.tsv missing — cross-check skipped (bad extract?)")


if __name__ == "__main__":
    main()
