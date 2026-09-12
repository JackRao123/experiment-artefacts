"""Rank-wise idle-gap inventory from nsys SQLite; read-only, no causal guessing."""
import argparse
import bisect
import collections
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nsys_workflow_20260911"))
from analyze import clipped, duration, process_id, subtract


def run(path):
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    nvtx = collections.defaultdict(list)
    windows = []
    for a, b, tid, text, text_id in db.execute(
        "SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL"
    ):
        if not tid:
            continue
        name = text or strings.get(text_id, "")
        nvtx[process_id(tid)].append((a, b, tid, name))
        if name.startswith("forward_backward/rank:"):
            windows.append((a, b, process_id(tid), int(name.split(":")[-1])))
    output = {"trace": str(path), "warning": "CPU scope coincidence and next-launch attribution are not causal proofs", "ranks": {}}
    for lo, hi, pid, rank in sorted(windows, key=lambda x: x[-1]):
        scopes = nvtx[pid]
        runtime = {c: (a, b, tid, strings[n]) for a, b, tid, c, n in db.execute(
            "SELECT start,end,globalTid,correlationId,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE (globalTid & -16777216)=? AND start>=? AND start<=?",
            (pid, lo, hi))}
        ops = [(a, b, c, strings[n]) for a, b, c, n in db.execute(
            "SELECT start,end,correlationId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=? AND start<=? AND end>=?", (pid, hi, lo))]
        for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
            ops.extend((a, b, c, table) for a, b, c in db.execute(
                f"SELECT start,end,correlationId FROM {table} WHERE globalPid=? AND start<=? AND end>=?", (pid, hi, lo)))
        ops.sort()
        starts = [o[0] for o in ops]
        gaps = subtract([(lo, hi)], [(o[0], o[1]) for o in ops])
        host_groups = collections.defaultdict(list)
        for a, b, tid, name in scopes:
            if name.startswith(("fsdp_fetch_bucket", "fsdp:", "python_gc:")):
                host_groups[name].extend(clipped([(a,b)], lo, hi))
        host_summary = {name: {"count": len(ranges), "host_union_ms": duration(ranges),
            "gpu_idle_coincident_ms": duration(gaps)-duration(subtract(gaps, ranges))}
            for name, ranges in host_groups.items()}
        buckets = collections.Counter()
        details = []
        for a, b in sorted(gaps, key=lambda x: x[1]-x[0], reverse=True):
            if b-a < 100_000 and len(details) >= 25:
                continue
            idx = bisect.bisect_left(starts, b)
            op = ops[idx] if idx < len(ops) else None
            launch = runtime.get(op[2]) if op else None
            near = sorted((x for x in scopes if launch and x[2] == launch[2] and x[0] <= launch[0] and x[1] >= launch[1]), key=lambda x: x[1]-x[0])
            # Exact nested scopes only for material gaps; aggregate by next launch scope.
            label = near[0][3] if near else "unscoped/end"
            if b-a >= 100_000:
                buckets[label] += (b-a)/1e6
            if len(details) < 25:
                mid = (a+b)//2
                coincident = sorted((x for x in scopes if x[0] <= mid <= x[1]), key=lambda x:x[1]-x[0])[:8]
                apis = sorted((x for x in runtime.values() if x[0] < b and x[1] > a), key=lambda x:min(x[1],b)-max(x[0],a), reverse=True)[:4]
                details.append({"start_ms": (a-lo)/1e6, "ms": (b-a)/1e6,
                    "next_kernel": op[3] if op else None, "next_api": launch[3] if launch else None,
                    "next_launch_relative_gap_start_ms": (launch[0]-a)/1e6 if launch else None,
                    "next_scopes": [x[3] for x in near[:8]], "coincident_scopes": [x[3] for x in coincident],
                    "coincident_apis": [{"name":x[3], "ms":(min(x[1],b)-max(x[0],a))/1e6} for x in apis]})
        output["ranks"][str(rank)] = {"idle_ms":duration(gaps), "gaps":len(gaps),
            "host_scopes": host_summary,
            "gaps_ge_100us_ms":sum(b-a for a,b in gaps if b-a>=100_000)/1e6,
            "next_scope_ge_100us_ms":dict(buckets.most_common(15)), "largest_gaps":details}
        print(rank, output["ranks"][str(rank)]["idle_ms"], buckets.most_common(5), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.trace), indent=2)+"\n")
