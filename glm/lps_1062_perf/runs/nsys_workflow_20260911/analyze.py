"""One-command nsys report: python analyze.py CASE/timing.nsys-rep --benchmark CASE/benchmark.json.

Standard library only. Export on a machine with nsys; an exported SQLite file
can be analyzed anywhere. Explicit NVTX rank/step markers are required.
Reports measurements, not invented causal speedups. No raw input is modified.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import json
import re
import sqlite3
import statistics
import subprocess
from pathlib import Path

VERSION = 2
NS = 1e6


def merge(intervals):
    out = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return out


def duration(intervals):
    return sum(b - a for a, b in merge(intervals)) / NS


def subtract(left, right):
    out, j = [], 0
    right = merge(right)
    for a, b in merge(left):
        while j < len(right) and right[j][1] <= a:
            j += 1
        k = j
        while k < len(right) and right[k][0] < b:
            c, d = right[k]
            if c > a:
                out.append((a, min(c, b)))
            a = max(a, d)
            k += 1
        if a < b:
            out.append((a, b))
    return out


def clipped(intervals, a, b):
    return [(max(x, a), min(y, b)) for x, y in intervals if x < b and y > a]


def stats(values):
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)} if values else None


def process_id(global_id):
    # Nsight GlobalId layout: process identifier occupies all but low 24 thread bits.
    return global_id & ~((1 << 24) - 1)


def classify(name, scopes):
    low = name.lower()
    if name.startswith("memcpy:") or name == "memset":
        return "memory_copy"
    if "nccl" in low:
        for scope in reversed(scopes):
            if scope.startswith("collective:") and "/group:" in scope:
                function, group = scope.split("/group:", 1)
                if group == "CONTEXT_PARALLEL_GROUP":
                    return "cp_communication"
                if "EXPERT_DATA_PARALLEL_GROUP" in group:
                    return "fsdp_gather:expert" if "gather" in function or "allgather" in low else "fsdp_gradient:expert"
                if group in ("DATA_PARALLEL_GROUP_WITH_CP", "INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP"):
                    return "fsdp_gather:nonexpert" if "gather" in function or "allgather" in low else "fsdp_gradient:nonexpert"
                if group == "EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP":
                    return "expert_dispatch_combine"
        for scope in reversed(scopes):
            if scope.startswith("fsdp_gather:"):
                return scope
        if any("dispatch" in s or "combine" in s for s in scopes):
            return "expert_dispatch_combine"
        if "block_attention" in scopes or "attention" in scopes or "cp_communication" in scopes:
            return "cp_communication"
        return "collective_unclassified"
    if any(s in low for s in ("hybrid_ep", "hybridep", "nvshmem", "deepep")):
        return "expert_dispatch_combine"
    if "expert_gemm_forward" in scopes or "expert_gemm_backward" in scopes:
        return "expert_gemm_path"
    if any("dispatch" in s or "combine" in s for s in scopes):
        return "expert_dispatch_combine"
    if "moe_route" in scopes:
        return "expert_routing"
    if "lm_head" in scopes or "lm_head_gemm" in scopes:
        return "lm_head"
    if any(x in low for x in ("flash", "fmha", "sparse_attn", "sparse_attention", "lightning_indexer")):
        return "attention"
    if any(s in scopes for s in ("attention", "block_attention", "dsa_core", "dsa_indexer")):
        return "attention"
    if any(x in low for x in ("gemm", "cutlass", "nvjet", "matmul")):
        return "other_gemm"
    if "moe_experts" in scopes:
        return "expert_other"
    if "block_mlp" in scopes:
        return "mlp_other"
    return "other_compute"


def is_comm(category):
    return category.startswith(("fsdp_gather:", "fsdp_gradient:")) or category in (
        "collective_unclassified", "cp_communication", "expert_dispatch_combine")


def dependencies(db, tables, pid, contexts, operations):
    """Resolve observed CUDA event chains; missing/ambiguous records stay unknown.

    Stream order is reconstructed from correlated host API submission order.
    Follow event records and waits (including relayed waits) to the last device
    operation. Never treat an event handle as a unique recording instance.
    """
    needed = {"CUPTI_ACTIVITY_KIND_CUDA_EVENT", "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"}
    if not needed <= tables:
        return {}, {"status": "event tables absent"}
    streams, records, counts = collections.defaultdict(list), {}, collections.Counter()
    for i, o in enumerate(operations):
        if o["launch"] is not None:
            streams[(o["context"], o["stream"])].append({"time": o["launch"], "kind": "op", "operation": o, "index": i})
    for r in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_CUDA_EVENT WHERE globalPid=?", (pid,)):
        context = contexts.get(r["correlationId"])
        if not context or r["eventSyncId"] in (None, 4294967295):
            continue
        key = (r["contextId"], r["eventSyncId"])
        node = {"time": context[0]["start"], "kind": "record", "key": key}
        counts[key] += 1
        records[key] = node
        streams[(r["contextId"], r["streamId"])].append(node)
    wait_count = 0
    for r in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION WHERE globalPid=? AND syncType=2", (pid,)):
        context = contexts.get(r["correlationId"])
        if not context:
            continue
        streams[(r["contextId"], r["streamId"])].append({"time": context[0]["start"], "kind": "wait",
                                                       "key": (r["contextId"], r["eventSyncId"])})
        wait_count += 1
    records = {k: v for k,v in records.items() if counts[k] == 1}
    predecessors = {}
    for nodes in streams.values():
        previous = None
        for node in sorted(nodes, key=lambda n: n["time"]):
            node["previous"] = previous
            if node["kind"] == "op":
                predecessors[node["index"]] = previous
            previous = node
    cache, visiting = {}, set()

    def resolve(node):
        if node is None:
            return None
        if len(visiting) > 400:
            return None  # Bound malformed or unusually deep relay chains.
        ident = id(node)
        if ident in cache:
            return cache[ident]
        if ident in visiting:
            return None
        visiting.add(ident)
        if node["kind"] == "op":
            answer = (node["operation"]["b"], node["operation"]["category"])
        elif node["kind"] == "record":
            answer = resolve(node["previous"])
        else:
            before, source = resolve(node["previous"]), resolve(records.get(node["key"]))
            answer = max(before,source, key=lambda x:x[0]) if before and source else None
        visiting.remove(ident)
        cache[ident] = answer
        return answer

    resolved = {i: resolve(node) for i,node in predecessors.items()}
    inconsistent = 0
    for i, producer in resolved.items():
        if producer and producer[0] > operations[i]["a"] + 1000:
            # A supposed prerequisite cannot finish after its consumer began.
            # Ambiguous submission ordering/graph launches must not become a claim.
            resolved[i] = None
            inconsistent += 1
    return resolved, {"status": "partial event-chain reconstruction; unresolved dependencies excluded",
                      "waits": wait_count, "unique_record_instances": len(records),
                      "ambiguous_record_instances": sum(n>1 for n in counts.values()),
                      "inconsistent_edges_excluded": inconsistent,
                      "resolved_operation_predecessors": sum(v is not None for v in resolved.values())}


def prefetch_readiness(ops):
    """Strict one-gather-per-MoE, full-recompute scheduling diagnostic.

    This reports arrival versus preceding-block completion, NOT network cost
    or a causal zero-communication speedup. Unsupported patterns fail closed.
    """
    layer_ids = sorted({int(o["layer"]) for o in ops if o["layer"] is not None
                        and str(o["layer"]).isdigit() and o["phase"] == "forward"
                        and o["category"] == "expert_gemm_path"})
    gathers = sorted((o for o in ops if o["category"] == "fsdp_gather:expert"), key=lambda o:o["a"])
    if not gathers:
        return {"status": "no attributed expert-weight gathers; check group annotation coverage", "gathers": 0}
    if not layer_ids or len(gathers) != 2*len(layer_ids):
        return {"status": "unsupported/incomplete gather pattern", "gathers": len(gathers), "expert_layers": len(layer_ids)}
    compute = collections.defaultdict(list)
    for o in ops:
        if o["layer"] is not None and not is_comm(o["category"]) and o["category"] != "memory_copy":
            compute[(str(o["layer"]), o["phase"])].append((o["a"], o["b"]))
    rows = []
    n = len(layer_ids)
    for i, gather in enumerate(gathers):
        forward = i < n
        target = layer_ids[i] if forward else list(reversed(layer_ids))[i-n]
        phase = "forward" if forward else "recompute"
        consumer = compute.get((str(target), phase))
        if not consumer or gather["b"] > min(a for a,b in consumer) + 1000:
            return {"status": "gather-to-layer order not verified; no readiness claim", "gathers": len(gathers)}
        previous = target-1 if forward else target+1
        predecessor = compute.get((str(previous), "forward" if forward else "backward"))
        previous_end = max(b for a,b in predecessor) if predecessor else None
        rows.append({"phase": "forward" if forward else "backward", "layer": target,
                     "gather_ms": (gather["b"]-gather["a"])/NS,
                     "lead_before_layer_gpu_start_ms": (min(a for a,b in consumer)-gather["b"])/NS,
                     "lead_before_previous_block_end_ms": (previous_end-gather["b"])/NS if previous_end is not None else None})
    comparable = [r for r in rows if r["lead_before_previous_block_end_ms"] is not None]
    return {"status": "source-scoped sequential full-recompute pattern verified", "gathers": len(gathers),
            "comparable_predecessors": len(comparable),
            "ready_before_previous_block_ends": sum(r["lead_before_previous_block_end_ms"] >= 0 for r in comparable),
            "rows": rows}


def analyze(path, benchmark=None):
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    required = {"NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME", "StringIds"}
    if not required <= tables:
        raise ValueError(f"Missing tables: {required - tables}")
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    nvtx = collections.defaultdict(list)
    windows = collections.defaultdict(list)
    for r in db.execute("SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL ORDER BY start,end DESC"):
        name = r[3] or strings.get(r[4], "")
        if not r[2]:
            continue
        nvtx[r[2]].append((r[0], r[1], name))
        match = re.fullmatch(r"forward_backward/rank:(\d+)", name)
        if match:
            windows[process_id(r[2])].append((r[0], r[1], int(match[1])))
    if not windows:
        raise ValueError("No forward_backward/rank:N NVTX anchors. Enable trainer record_nvtx_ranges with this instrumentation revision.")
    result = {"analysis_version": VERSION, "input": str(path), "benchmark": benchmark,
              "quality": {"rank_source": "explicit NVTX marker, not filename/device index", "source_callstack_coverage": None,
                          "dependency_resolution": "partial CUDA event-chain reconstruction; exposure is NOT blocking time",
                          "hardware_metrics_present": "GPU_METRICS" in tables,
                          "tables": sorted(tables)}, "ranks": {}}
    diagnostics = []
    if "DIAGNOSTIC_EVENT" in tables:
        diagnostics = [dict(r) for r in db.execute("SELECT * FROM DIAGNOSTIC_EVENT WHERE severity>1")]
    result["quality"]["diagnostics"] = diagnostics
    grouped_diagnostics = collections.Counter((d["severity"], d["text"], d.get("globalPid") in windows) for d in diagnostics)
    result["quality"]["diagnostic_summary"] = [{"severity": sev, "text": text, "on_captured_rank": on_rank, "count": n}
                                                for (sev,text,on_rank),n in grouped_diagnostics.items()]
    for pid, steps in windows.items():
        ranks = {w[2] for w in steps}
        assert len(ranks) == 1, ranks
        rank = ranks.pop()
        lo, hi = min(w[0] for w in steps), max(w[1] for w in steps)
        runtimes = [dict(r) for r in db.execute(
            "SELECT start,end,globalTid,correlationId,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME "
            "WHERE (globalTid & -16777216)=? AND start>=? AND start<=? ORDER BY start", (pid, lo, hi))]
        # Sweep each CPU thread independently: async launches inherit their CPU
        # scopes, not whichever CPU range happens to overlap GPU execution.
        contexts, scope_keys, cursors, active = {}, {}, collections.defaultdict(int), collections.defaultdict(list)
        for r in runtimes:
            tid, t = r["globalTid"], r["start"]
            ranges = nvtx[tid]
            queue = active[tid]
            while cursors[tid] < len(ranges) and ranges[cursors[tid]][0] <= t:
                item = ranges[cursors[tid]]
                if item[1] >= r["end"]:
                    queue.append(item)
                cursors[tid] += 1
            queue[:] = [x for x in queue if x[1] >= r["end"]]
            scopes = [x[2] for x in queue]
            key = r["correlationId"]
            if key in contexts:
                raise ValueError(f"Duplicate runtime correlation in process {pid}: {key}")
            contexts[key] = (r, scopes)
            scope_keys[key] = [(tid, *x) for x in queue if x[2].startswith(("expert_gemm_", "expert_shape:", "fsdp_gather:"))]
        operations = []
        kernels = db.execute("SELECT start,end,correlationId,streamId,demangledName,deviceId,contextId FROM CUPTI_ACTIVITY_KIND_KERNEL "
                             "WHERE globalPid=? AND end>=? AND start<=? ORDER BY start", (pid, lo, hi))
        linked = 0
        scope_gpu = collections.defaultdict(list)
        material = collections.defaultdict(lambda: {"calls": 0, "summed_ms": 0})
        for r in kernels:
            context, scopes = contexts.get(r[2], (None, []))
            linked += context is not None
            name = strings.get(r[4], "unknown")
            category = classify(name, scopes)
            for key in scope_keys.get(r[2], []):
                scope_gpu[key].append((r[0], r[1]))
            layer, phase = None, "backward" if "backward" in scopes else "forward"
            for s in scopes:
                if s.startswith("layer_backward:"):
                    layer, phase = s.split(":", 1)[1], "backward"
                if s.startswith("layer:"):
                    layer = s.split(":", 1)[1]
                if s == "phase:recompute":
                    phase = "recompute"
            operations.append({"a": r[0], "b": r[1], "category": category, "layer": layer, "phase": phase,
                               "launch": context["start"] if context else None, "stream": r[3], "device": r[5], "context": r[6]})
            material[(category, name)]["calls"] += 1
            material[(category, name)]["summed_ms"] += (r[1] - r[0]) / NS
        kernel_count = len(operations)
        for table, label in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy:"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
            if table not in tables:
                continue
            for r in db.execute(f"SELECT * FROM {table} WHERE globalPid=? AND end>=? AND start<=?", (pid, lo, hi)):
                context = contexts.get(r["correlationId"])
                operations.append({"a": r["start"], "b": r["end"], "category": "memory_copy", "layer": None,
                                   "phase": "unknown", "launch": context[0]["start"] if context else None,
                                   "stream": r["streamId"], "device": r["deviceId"], "context": r["contextId"]})
        operations.sort(key=lambda x: x["a"])
        resolved, dependency_quality = dependencies(db, tables, pid, contexts, operations)
        expert_host_ranges = [(a,b) for tid,ranges in nvtx.items() if process_id(tid)==pid
                              for a,b,name in ranges if name.startswith("expert_shape:") or name=="expert_gemm_backward"]
        for i,o in enumerate(operations):
            o["producer"] = resolved.get(i)
        summaries, layers = [], []
        for index, (a, b, _) in enumerate(steps):
            ops = [o for o in operations if o["a"] < b and o["b"] > a]
            intervals = lambda items: clipped([(o["a"], o["b"]) for o in items], a, b)
            busy = merge(intervals(ops))
            idle = subtract([(a, b)], busy)
            categories = collections.defaultdict(list)
            by_layer = collections.defaultdict(list)
            for o in ops:
                categories[o["category"]].append((max(a, o["a"]), min(b, o["b"])))
                if o["layer"] is not None:
                    by_layer[(o["layer"], o["phase"], o["category"])].append((max(a, o["a"]), min(b, o["b"])))
            budgets = {}
            for cat, intervals_cat in categories.items():
                others = [v for c, values in categories.items() if c != cat for v in values]
                budgets[cat] = {"union_ms": duration(intervals_cat), "exclusive_ms": duration(subtract(intervals_cat, others)),
                                "summed_ms": sum(y - x for x, y in intervals_cat) / NS}
            comm = [v for c, values in categories.items() if is_comm(c) for v in values]
            noncomm = [v for c, values in categories.items() if not is_comm(c) for v in values]
            compute = [v for c, values in categories.items() if not is_comm(c) and c != "memory_copy" for v in values]
            absent = subtract([(a, b)], compute)
            compute_ops = [o for o in ops if not is_comm(o["category"]) and o["category"] != "memory_copy"]
            starts = [o["a"] for o in compute_ops]
            unissued, issued_or_unknown, blocking_comm = [], [], []
            for x, y in absent:
                pos = bisect.bisect_left(starts, y)
                launch = compute_ops[pos]["launch"] if pos < len(compute_ops) else None
                split = min(y, max(x, launch)) if launch is not None else x
                unissued.append((x, split))
                producer = compute_ops[pos]["producer"] if pos < len(compute_ops) else None
                blocked_until = split
                if launch is not None and producer and is_comm(producer[1]):
                    blocked_until = min(y, max(split, producer[0]))
                    blocking_comm.append((split, blocked_until))
                issued_or_unknown.append((blocked_until, y))
            alloc = [(r["start"], r["end"]) for r in runtimes
                     if re.search(r"(Malloc|Free|MemMap|MemUnmap|MemCreate|MemRelease|MemPoolTrimTo)", strings.get(r["nameId"], ""))]
            alloc_idle = duration(idle) - duration(subtract(idle, alloc))
            row = {"step": index, "start_ns": a, "end_ns": b, "step_ms": (b-a)/NS,
                   "device_busy_ms": duration(busy), "device_idle_ms": duration(idle),
                   "compute_busy_ms": duration(compute), "compute_absent_ms": duration(absent),
                   "next_compute_not_yet_issued_ms": duration(unissued),
                   "issued_or_unresolved_ms": duration(issued_or_unknown),
                   "communication_dispatcher_union_ms": duration(comm),
                   "communication_dispatcher_exposed_upper_bound_ms": duration(subtract(comm, noncomm)),
                   "resolved_event_blocking_communication_ms": duration(blocking_comm) if "waits" in dependency_quality else None,
                   "allocation_api_coincident_idle_ms": alloc_idle, "categories": budgets}
            # Association with a host scope is not a claim that every cycle is
            # CPU computation; it can include blocking CUDA API calls.
            row["gpu_idle_during_expert_host_scope_ms"] = duration(idle)-duration(subtract(idle, expert_host_ranges))
            row["mixed_category_overlap_ms"] = row["device_busy_ms"] - sum(v["exclusive_ms"] for v in budgets.values())
            row["expert_prefetch_readiness"] = prefetch_readiness(ops)
            assert row["mixed_category_overlap_ms"] >= -1e-5
            assert abs(row["step_ms"]-row["device_busy_ms"]-row["device_idle_ms"]) < 1e-5
            assert abs(row["compute_absent_ms"]-row["next_compute_not_yet_issued_ms"]-row["issued_or_unresolved_ms"]-(row["resolved_event_blocking_communication_ms"] or 0)) < 1e-5
            assert row["compute_busy_ms"] <= row["device_busy_ms"] + 1e-5
            summaries.append(row)
            for (layer, phase, cat), spans in by_layer.items():
                layers.append({"step": index, "layer": layer, "phase": phase, "category": cat,
                               "gpu_union_ms": duration(spans), "gpu_span_ms": (max(y for x,y in spans)-min(x for x,y in spans))/NS})
        mean_categories = {}
        for cat in {cat for s in summaries for cat in s["categories"]}:
            mean_categories[cat] = {key: stats([s["categories"].get(cat, {}).get(key, 0) for s in summaries])
                                    for key in ("union_ms", "exclusive_ms", "summed_ms")}
        hardware = {}
        physical_device = (benchmark or {}).get("physical_gpu_by_rank", {}).get(str(rank))
        if "GPU_METRICS" in tables and physical_device is not None:
            # NVIDIA nsys_recipe/recipes/nvlink_sum uses typeId & 0xFF for physical GPU id.
            # Require the capture's explicit CUDA-visible-to-physical mapping.
            roi = merge([(a,b) for a,b,_ in steps])
            by_category = collections.defaultdict(list)
            for o in operations:
                by_category[o["category"]].extend(clipped(roi, o["a"], o["b"]))
            metric_rows = list(db.execute(
                "SELECT typeId,metricId,metricName FROM TARGET_INFO_GPU_METRICS WHERE (typeId & 255)=?",
                (physical_device,)))
            wanted = ("SMs Active", "SM Issue", "Tensor Active", "DRAM Read", "DRAM Write", "NVLink", "GPC Clock")
            metric_rows = [r for r in metric_rows if r[2].startswith(wanted)]
            wanted_keys = {(r[0],r[1]) for r in metric_rows}
            samples_by_metric = collections.defaultdict(list)
            # One scan per GPU, not a full SQLite scan per individual metric.
            for timestamp,type_id,metric_id,value in db.execute(
                "SELECT timestamp,typeId,metricId,value FROM GPU_METRICS WHERE (typeId & 255)=? AND timestamp>=? AND timestamp<=?",
                (physical_device,lo,hi)):
                if (type_id,metric_id) in wanted_keys:
                    samples_by_metric[(type_id,metric_id)].append((timestamp,value))
            by_category = {cat: merge(spans) for cat,spans in by_category.items()}
            exclusive_spans = {cat: subtract(spans,[v for c,vv in by_category.items() if c!=cat for v in vv])
                               for cat,spans in by_category.items()}
            for type_id, metric_id, name in metric_rows:
                samples = sorted(samples_by_metric[(type_id,metric_id)])
                timestamps = [s[0] for s in samples]
                def sample_values(spans):
                    return [samples[i][1] for x,y in spans
                            for i in range(bisect.bisect_left(timestamps,x), bisect.bisect_left(timestamps,y))]
                per_category = {}
                for cat, spans in by_category.items():
                    values = sample_values(spans)
                    exclusive = sample_values(exclusive_spans[cat])
                    per_category[cat] = {"all_samples": stats(values), "exclusive_samples": stats(exclusive),
                                         "exclusive_fraction": len(exclusive)/len(values) if values else None}
                hardware[name] = {"whole_step": stats(sample_values(roi)), "categories": per_category}
        ranked = sorted(mean_categories.items(), key=lambda kv: kv[1]["exclusive_ms"]["mean"], reverse=True)[:5]
        candidates = [{"category": c, "exclusive_ms": v["exclusive_ms"]["mean"], "union_ms": v["union_ms"]["mean"],
                       "claim": "observed cost, not recoverable time or proven bottleneck"} for c,v in ranked]
        idle_mean = statistics.mean(s["device_idle_ms"] for s in summaries)
        candidates.append({"category": "gpu_idle", "exclusive_ms": idle_mean, "union_ms": idle_mean,
                           "claim": "no recorded GPU activity; inspect submission/sync/allocator evidence before assigning a cause"})
        result["ranks"][str(rank)] = {"global_pid": pid, "devices": sorted({o["device"] for o in operations}),
            "graph_launch_api_calls": sum("GraphLaunch" in strings.get(r["nameId"], "") for r in runtimes),
            "kernel_count": kernel_count, "runtime_correlation_fraction": linked/kernel_count if kernel_count else None,
            "step_ms": stats([s["step_ms"] for s in summaries]), "steps": summaries, "layers": layers,
            "categories": mean_categories, "top_five_observed_costs": sorted(candidates, key=lambda c:c["exclusive_ms"], reverse=True)[:5],
            "hardware_metrics": hardware,
            "dependency_quality": dependency_quality,
            "expert_scope_instances": [{"name": name, "cpu_start_ns": a, "cpu_end_ns": b,
                "cpu_duration_ms": (b-a)/NS, "gpu_start_ns": min(x for x,y in spans), "gpu_end_ns": max(y for x,y in spans),
                "gpu_union_ms": duration(spans), "gpu_span_ms": (max(y for x,y in spans)-min(x for x,y in spans))/NS,
                "cpu_entry_to_first_kernel_ms": (min(x for x,y in spans)-a)/NS,
                "warning": "Entry delay may include earlier GPU work; not isolated CPU overhead."}
                for (tid,a,b,name), spans in scope_gpu.items()],
            "kernel_inventory": [{"category": c, "name": n, **v} for (c, n), v in
                                 sorted(material.items(), key=lambda kv: kv[1]["summed_ms"], reverse=True)]}
    if benchmark and benchmark.get("control"):
        result["quality"]["control_warnings"] = []
        if benchmark["control"]["n"] < 3:
            result["quality"]["control_warnings"].append("Fewer than three controls; steady-state variability is not established.")
        phases = [w["fb_elapsed_s"] for w in benchmark["windows"] if w["phase"] == path.stem]
        if phases:
            result["quality"]["same_run_capture_slowdown_pct"] = 100*(statistics.median(phases)/benchmark["control"]["median_s"]-1)
            if not benchmark["control"]["min_s"] <= statistics.median(phases) <= benchmark["control"]["max_s"]:
                result["quality"]["control_warnings"].append("Profile median is outside the observed control range: instrumentation, remaining warmup, or workload drift may affect attribution.")
    result["quality"]["rank_coverage"] = {"captured": len(result["ranks"]), "world_size": (benchmark or {}).get("num_gpus")}
    result["quality"]["graph_launch_api_calls"] = sum(r["graph_launch_api_calls"] for r in result["ranks"].values())
    db.close()
    return result


def markdown(result):
    q = result["quality"]
    lines = ["# Nsight runtime report", "", "## Verdict", "",
             "Ranked observed costs below are candidates for investigation, not guaranteed speedups.", "",
             "## Capture quality", "", f"- Input: `{result['input']}`",
             f"- Ranks: {', '.join(result['ranks'])}; explicit NVTX rank/forward-backward anchors.",
             f"- Rank coverage: {q['rank_coverage']}; observed CUDA graph launch calls: {q['graph_launch_api_calls']}.",
             f"- Same-run capture slowdown: {q.get('same_run_capture_slowdown_pct', 'unavailable')}%.",
             f"- Hardware metrics present: {q['hardware_metrics_present']}; this timing report does not infer saturation.",
             f"- Warning/error records: {len(q['diagnostics'])}; see diagnostics below before trusting event completeness.",
             "- Source call stacks not captured. NVTX scope attribution is distinct from call-stack coverage.",
             "- Partial CUDA event dependency reconstruction; unresolved/ambiguous events are excluded, not guessed.", "",
             "## Per-step budget", "", "All times ms, means across captured forward/backward requests. Optimizer excluded.", "",
             "| Rank | n | Step | GPU busy | GPU idle | Compute absent | Next compute unissued | Comm/dispatcher | Exposed upper bound | Event-linked blocking |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rank, r in result["ranks"].items():
        fields = ("step_ms", "device_busy_ms", "device_idle_ms", "compute_absent_ms", "next_compute_not_yet_issued_ms",
                  "communication_dispatcher_union_ms", "communication_dispatcher_exposed_upper_bound_ms", "resolved_event_blocking_communication_ms")
        values = [statistics.mean(s[k] for s in r["steps"]) if all(s[k] is not None for s in r["steps"]) else None for k in fields]
        lines.append(f"| {rank} | {len(r['steps'])} | " + " | ".join(f"{x:.2f}" if x is not None else "unknown" for x in values) + " |")
    lines += ["", "### Capture diagnostics", ""]
    for warning in q.get("control_warnings", []):
        lines.append("- " + warning)
    for d in q["diagnostic_summary"]:
        who = "captured trainer ranks" if d["on_captured_rank"] else "other/helper processes"
        lines.append(f"- {d['count']}× on {who}: {d['text']}")
    if any(d["on_captured_rank"] for d in q["diagnostic_summary"]):
        lines += ["", "Capture completeness is qualified by these warnings. Complete step anchors and runtime links do not prove zero event loss."]
    lines += ["", "## Expert-weight prefetch arrivals", "",
              "Arrival relative to preceding-block compute completion; not a zero-communication speedup estimate.", "",
              "| Rank | Step | Attributed gathers | Ready by preceding block end | Status |",
              "|---|---:|---:|---|---|"]
    for rank, r in result["ranks"].items():
        for step in r["steps"]:
            p = step["expert_prefetch_readiness"]
            ready = f"{p['ready_before_previous_block_ends']}/{p['comparable_predecessors']}" if "comparable_predecessors" in p else "unknown"
            lines.append(f"| {rank} | {step['step']} | {p['gathers']} | {ready} | {p['status']} |")
    lines += ["", "## Ranked observed costs", "", "Exclusive time means no OTHER classified device category overlapped. It is not a causal gain estimate."]
    for rank, r in result["ranks"].items():
        lines += ["", f"### Rank {rank}", "", f"Runtime correlation: {r['runtime_correlation_fraction']:.2%}.", "",
                  "| Category | Exclusive ms | Union ms |", "|---|---:|---:|"]
        for c in r["top_five_observed_costs"]:
            lines.append(f"| {c['category']} | {c['exclusive_ms']:.2f} | {c['union_ms']:.2f} |")
    if any(r["hardware_metrics"] for r in result["ranks"].values()):
        lines += ["", "## Expert-GEMM hardware samples", "",
                  "Category-exclusive samples only. Excludes launch preparation and gaps. Device-wide counters are not per-kernel rooflines.", "",
                  "| Rank | Metric | Exclusive mean | Sample count | Exclusive fraction |", "|---|---|---:|---:|---:|"]
        for rank, r in result["ranks"].items():
            for name, metric in r["hardware_metrics"].items():
                item = metric["categories"].get("expert_gemm_path", {})
                samples = item.get("exclusive_samples")
                if samples:
                    lines.append(f"| {rank} | {name} | {samples['mean']:.2f} | {samples['n']} | {item['exclusive_fraction']:.1%} |")
    lines += ["", "## Claim status", "", "- Interval budgets: trace-measured; overlapping category unions do not add to step time.",
              "- Communication taxonomy: NVTX scope plus kernel names; unclassified collectives remain unclassified.",
              "- Dispatch/combine includes packing and synchronization, not just network transfer.",
              "- Next-compute-unissued intervals are observations, not proof the CPU could legally launch earlier.",
              "- Event-linked blocking is a partial trace-derived bound; missing edges and opaque dispatcher-internal dependencies remain unresolved.",
              "- GEMM inefficiency, bandwidth saturation and recoverable gains require counters or matched A/B evidence.",
              "- Rank durations alone do not prove straggling; compare aligned start/end timestamps in the JSON."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true", help="Recompute even if inputs and analyzer match the cached report")
    args = parser.parse_args()
    trace = args.trace
    if trace.suffix == ".nsys-rep":
        exported = trace.with_suffix(".sqlite")
        if exported.exists() and exported.stat().st_mtime_ns < trace.stat().st_mtime_ns:
            raise ValueError("SQLite export is older than the report; export to a fresh path instead of using stale data")
        if not exported.exists():
            subprocess.run(["nsys", "export", "--type=sqlite", f"--output={exported}", str(trace)], check=True)
        trace = exported
    benchmark = json.loads(args.benchmark.read_text()) if args.benchmark else None
    output = args.output or trace.with_suffix(".analysis.json")
    fingerprint = {"path": str(trace.resolve()), "size": trace.stat().st_size, "mtime_ns": trace.stat().st_mtime_ns,
                   "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "benchmark_sha256": hashlib.sha256(json.dumps(benchmark, sort_keys=True).encode()).hexdigest()}
    if output.exists() and not args.force and json.loads(output.read_text()).get("cache_key") == fingerprint:
        print(f"Cached: {output}")
        return
    result = analyze(trace, benchmark)
    result["cache_key"] = fingerprint
    output.write_text(json.dumps(result, indent=2))
    output.with_suffix(".md").write_text(markdown(result))
    print(output)
    print(output.with_suffix(".md"))


if __name__ == "__main__":
    main()
