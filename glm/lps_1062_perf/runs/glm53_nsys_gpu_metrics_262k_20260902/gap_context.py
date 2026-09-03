"""Explain the largest GPU-idle gaps: what the host was doing while the GPU had nothing.

For each of the top-N idle gaps (no kernel resident) on a GPU inside the step
window, prints the last kernel before / first kernel after the gap, the NVTX
ranges open on the owning process at the gap start, and every CUDA runtime call
on that process that overlaps the gap (name, duration, callchain summary).

usage: python gap_context.py <sqlite> --gpu 1 [--top 3] [--window-from analysis_dir/step_window.csv]
"""

import argparse
import csv
import sqlite3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--window-from", required=True, help="step_window.csv from nsys_attrib.py")
    args = ap.parse_args()

    win = {int(r["gpu"]): (int(r["start_ns"]), int(r["end_ns"])) for r in csv.DictReader(open(args.window_from))}
    w0, w1 = win[args.gpu]
    db = sqlite3.connect(args.sqlite)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))

    kernels = db.execute(
        "SELECT start, end, globalPid, shortName, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE deviceId = ? AND start >= ? AND end <= ? ORDER BY start",
        (args.gpu, w0, w1),
    ).fetchall()
    print(f"gpu {args.gpu}: {len(kernels)} kernels in window [{w0}, {w1}]")
    pid = kernels[0][2]
    # idle gaps: consecutive kernels with a hole (kernels can overlap across streams, so
    # track the running max end).
    gaps = []
    run_end = kernels[0][1]
    for i in range(1, len(kernels)):
        s = kernels[i][0]
        if s > run_end:
            gaps.append((s - run_end, run_end, s, i))
        run_end = max(run_end, kernels[i][1])
    gaps.sort(reverse=True)

    nvtx = db.execute(
        "SELECT start, end, text, textId, globalTid FROM NVTX_EVENTS WHERE eventType IN (59, 60) "
        "AND end IS NOT NULL AND (globalTid >> 24) = (? >> 24) AND start <= ? AND end >= ?",
        (pid, w1, w0),
    ).fetchall()

    def nvtx_open(t):
        names = []
        for s, e, text, tid, _ in nvtx:
            if s <= t <= e:
                name = text if text else strings.get(tid, "?")
                if not name.startswith("aten::"):
                    names.append(name)
        return names

    for gap_ns, g0, g1, idx in gaps[: args.top]:
        print(f"\n=== gap {gap_ns / 1e6:.1f} ms  [{(g0 - w0) / 1e9:.3f} s .. {(g1 - w0) / 1e9:.3f} s into the step]")
        kb = kernels[idx - 1]
        ka = kernels[idx]
        print(f"  last kernel before : {strings.get(kb[3], kb[3])}  ({(kb[1] - kb[0]) / 1e3:.0f} us)")
        print(f"  first kernel after : {strings.get(ka[3], ka[3])}  ({(ka[1] - ka[0]) / 1e3:.0f} us)")
        print(f"  NVTX open at gap start: {nvtx_open(g0)}")
        print(f"  NVTX open at gap end  : {nvtx_open(g1)}")
        rt = db.execute(
            "SELECT start, end, globalTid, nameId, callchainId FROM CUPTI_ACTIVITY_KIND_RUNTIME "
            "WHERE (globalTid >> 24) = (? >> 24) AND end >= ? AND start <= ? AND (end - start) > 200000 "
            "ORDER BY start",
            (pid, g0 - 50_000_000, g1),
        ).fetchall()
        print(f"  runtime calls >0.2 ms overlapping [gap-50ms, gap end] on this process: {len(rt)}")
        for s, e, tid, nid, cc in rt[:12]:
            name = strings.get(nid, nid)
            frames = ""
            if cc is not None:
                rows = db.execute(
                    "SELECT symbol FROM CUDA_CALLCHAINS WHERE id = ? ORDER BY stackDepth LIMIT 12", (cc,)
                ).fetchall()
                syms = [strings.get(r[0], "?") for r in rows]
                syms = [x.split("(")[0][:60] for x in syms if not x.startswith("0x") and "cudaStreamSynchronize" not in x]
                frames = " < ".join(syms[:5])
            print(
                f"    tid {tid & 0xFFFFFF:>7} {name:28s} {(e - s) / 1e6:9.2f} ms  "
                f"[{(s - w0) / 1e9:.3f}..{(e - w0) / 1e9:.3f}]  {frames}"
            )
        # NVTX ranges that START inside the gap on the main thread (what the host did next)
        started = [
            (s, text if text else strings.get(tid, "?"))
            for s, e, text, tid, _ in nvtx
            if g0 <= s <= g1 and not (text or "").startswith("aten::")
        ]
        started.sort()
        print(f"  NVTX ranges starting inside the gap: {[n for _, n in started[:10]]}")
        aten = [
            (s, text if text else strings.get(tid, "?"))
            for s, e, text, tid, _ in nvtx
            if g0 <= s <= g1 and (text or "").startswith("aten::")
        ]
        print(f"  aten ops starting inside the gap: {len(aten)}; first few: {[n for _, n in sorted(aten)[:8]]}")


if __name__ == "__main__":
    main()
