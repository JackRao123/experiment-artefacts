"""Perfetto kernel attribution and interval unions, not sums treated as wall time."""
import argparse, csv, io, json, pickle, subprocess
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
QUERIES={
"kernels":"SELECT id,ts/1e3 ts_us,dur/1e3 dur_us,name,track_id,EXTRACT_ARG(arg_set_id,'args.External id') external_id FROM slice WHERE category='kernel' AND dur>0 ORDER BY ts",
"owners":"SELECT id,ts/1e3 ts_us,dur/1e3 dur_us,name,EXTRACT_ARG(arg_set_id,'args.External id') external_id FROM slice WHERE category='cpu_op' AND dur>0",
"blocks":"SELECT ts/1e3 ts_us,dur/1e3 dur_us,name FROM slice WHERE category='user_annotation' AND name GLOB 'block_*/*' AND dur>0",
}
def union(intervals):
    total=0
    end=None
    for a,b in sorted(intervals):
        if end is None or a>end:
            total+=b-a
        elif b>end:
            total+=b-end
        end=max(end,b) if end is not None else b
    return total
def group(name):
    n=name.lower()
    if "nccl" in n:
        if "allgather" in n: return "nccl_allgather"
        if "reducescatter" in n: return "nccl_reducescatter"
        if "allreduce" in n: return "nccl_allreduce"
        return "nccl_other"
    if any(x in n for x in ["dispatch","combine","hybrid_ep","hybridep"]): return "dispatch_combine"
    if any(x in n for x in ["topk","top_k","indexer","index_scores"]): return "indexing"
    if any(x in n for x in ["sdpa","fmha","flash_attn","sparse_attn","attention_backwarddsa"]): return "attention"
    if any(x in n for x in ["gemm","nvjet","cublas","grouped_mm"]): return "gemm"
    if any(x in n for x in ["topk","top_k","indexer","index_scores"]): return "indexing"
    return "other"
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("topology")
    args=ap.parse_args()
    folder=ROOT/args.topology
    out=folder/"analysis";out.mkdir(exist_ok=True)
    traces=list((folder/"runtime").glob("*.pt.trace.json"));assert len(traces)==1
    data={}
    for label,sql in QUERIES.items():
        (out/f"{label}.sql").write_text(sql+";\n")
        p=subprocess.run(["/Users/jackrao/bin/trace_processor","query",str(traces[0]),sql],capture_output=True,text=True,check=True)
        (out/f"{label}.csv").write_text(p.stdout)
        (out/f"{label}.log").write_text(p.stderr)
        data[label]=list(csv.DictReader(io.StringIO(p.stdout)))
    owners=defaultdict(list)
    for r in data["owners"]:
        if r["external_id"] != "[NULL]": owners[r["external_id"]].append(r)
    totals=defaultdict(lambda:{"calls":0,"sum_ms":0})
    scopes=defaultdict(lambda:defaultdict(lambda:{"calls":0,"sum_ms":0,"intervals":[]}))
    kernel_names=defaultdict(lambda:{"calls":0,"sum_ms":0})
    for k in data["kernels"]:
        t=float(k["ts_us"]);d=float(k["dur_us"]);g=group(k["name"])
        totals[g]["calls"]+=1;totals[g]["sum_ms"]+=d/1000
        kernel_names[k["name"]]["calls"]+=1;kernel_names[k["name"]]["sum_ms"]+=d/1000
        candidates=[]
        for o in owners[k["external_id"]]:
            ot=float(o["ts_us"]);oe=ot+float(o["dur_us"])
            for b in data["blocks"]:
                bt=float(b["ts_us"]);bd=float(b["dur_us"])
                if bt<=ot and oe<=bt+bd+0.01: candidates.append(b)
        scope=min(candidates,key=lambda b:float(b["dur_us"]))["name"] if candidates else "outside_blocks_or_unattributed"
        v=scopes[scope][g];v["calls"]+=1;v["sum_ms"]+=d/1000;v["intervals"].append((t,t+d))
    for groups in scopes.values():
        for value in groups.values(): value["union_ms"]=union(value.pop("intervals"))/1000
    intervals=[(float(k["ts_us"]),float(k["ts_us"])+float(k["dur_us"])) for k in data["kernels"]]
    span=(max(b for a,b in intervals)-min(a for a,b in intervals))/1000
    result={"kernel_count":len(intervals),"gpu_span_ms":span,"gpu_union_busy_ms":union(intervals)/1000,"groups":dict(totals),"block_attribution":dict(scopes),
            "kernels":dict(sorted(kernel_names.items(),key=lambda x:-x[1]["sum_ms"]))}
    memories=list((folder/"memory").glob("*.pickle"))
    result["memory_validation"]={}
    for p in memories:
        with p.open("rb") as f: snap=pickle.load(f)
        result["memory_validation"][p.name]={"segments":len(snap.get("segments",[])),"history_events":sum(map(len,snap.get("device_traces",[])))}
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ["kernels","block_attribution"]},indent=2))
if __name__=="__main__":main()
