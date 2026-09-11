"""Collect exact artifacts and CUDA-event timings for one topology."""
import argparse, hashlib, json, statistics, subprocess
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent
REMOTE = "/root/glm53-1d2m-262k-20260909"
def copy(src,dst):
    subprocess.run(["scp","-C",f"tj-32vj99q:{src}",str(dst)],check=True)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("topology",choices=["cp8ep8","cp8ep1","cp1ep1"])
    a=ap.parse_args()
    dst=ROOT/a.topology
    label=f"glm53-1d2m-{a.topology}-262k-c5"
    copy(f"{REMOTE}/results/{label}.json",dst/"benchmark.json")
    bench=json.loads((dst/"benchmark.json").read_text())
    assert Counter(w["phase"] for w in bench["windows"]) == dict(warmup=1,control=5,memory_profile=1,runtime_profile=1)
    gpus=1 if a.topology=="cp1ep1" else 8
    assert bench["initial_status"]["world_size"]==gpus
    assert bench["seq_len"]==262144 and bench["datums_per_window"]==1
    assert bench["final_status"]["step"]-bench["initial_status"]["step"]==8
    manifest={}
    for kind in ["runtime","memory"]:
        info=bench[f"{kind}_profile_stop"]
        folder=dst/kind
        folder.mkdir(exist_ok=True)
        total_bytes=0
        for name in info["files"]:
            assert Path(name).name==name
            p=folder/name
            copy(f"{info['local_path']}/{name}",p)
            with p.open("rb") as f: sha=hashlib.file_digest(f,"sha256").hexdigest()
            manifest[str(p.relative_to(dst))]={"bytes":p.stat().st_size,"sha256":sha}
            total_bytes+=p.stat().st_size
        assert total_bytes==info["size_bytes"], (kind,total_bytes,info["size_bytes"])
    copy(f"{REMOTE}/.devbox_up/trainer_srun.log",dst/"trainer_srun.log")
    copy(f"{REMOTE}/{a.topology}/driver.log",dst/"driver.log")
    copy(f"{REMOTE}/results/{label}.windows.jsonl",dst/"windows.jsonl")
    timing_dir=dst/"layer_timings"
    timing_dir.mkdir(exist_ok=True)
    records=[]
    for rank in range(gpus):
        p=timing_dir/f"rank{rank}.jsonl"
        copy(f"{REMOTE}/{a.topology}/layer_timings/rank{rank}.jsonl",p)
        records.extend(json.loads(s) for s in p.read_text().splitlines())
    controls=[w for w in bench["windows"] if w["phase"]=="control"]
    # The server also executes a 64-token startup fb before serving requests.
    # Align the final eight recorded fb calls to the explicit driver protocol.
    steps=sorted({r["step"] for r in records if r["rank"]==0})
    assert len(steps)>=8, steps
    protocol_steps=steps[-8:]
    step_map={s:i+1 for i,s in enumerate(protocol_steps)}
    rows=[{**r,"worker_step":r["step"],"step":step_map[r["step"]]} for r in records if r["step"] in protocol_steps[1:6]]
    assert len(rows)==5*3*gpus,(len(rows),gpus)
    assert len({(r["step"],r["layer"],r["rank"]) for r in rows})==len(rows)
    keys=["forward_ms","recompute_ms","backward_including_recompute_ms","backward_excluding_recompute_ms","backward_after_recompute_ms"]
    def stats(values):
        return {"mean":statistics.mean(values),"sd":statistics.stdev(values),"min":min(values),"max":max(values),"n":len(values)}
    grouped=defaultdict(list)
    for r in rows: grouped[(r["step"],r["layer"])].append(r)
    # Max rank per block is a conservative distributed critical-rank proxy.
    slow=[{"step":s,"layer":l,**{k:max(r[k] for r in rr) for k in keys}} for (s,l),rr in grouped.items()]
    modes={}
    for mode,rr in [("rank0",[r for r in rows if r["rank"]==0]),("rank_max",slow)]:
        modes[mode]={"by_layer":{str(l):{k:stats([r[k] for r in rr if r["layer"]==l]) for k in keys} for l in range(3)},
                     "moe_average":{k:stats([r[k] for r in rr if r["layer"]>=1]) for k in keys}}
        modes[mode]["full_model_blocks_seconds"]={k:sum(modes[mode]["by_layer"][str(l)][k]["mean"]*n for l,n in enumerate([3,57,18]))/1000 for k in keys}
        estimates=[]
        for step in range(2,7):
            by_layer={r["layer"]:r for r in rr if r["step"]==step}
            block_fb=lambda r: r["forward_ms"]+r["backward_including_recompute_ms"]
            measured_blocks=sum(block_fb(r) for r in by_layer.values())/1000
            weighted_blocks=sum(block_fb(by_layer[l])*n for l,n in enumerate([3,57,18]))/1000
            residual=controls[step-2]["fb_elapsed_s"]-measured_blocks
            estimates.append({"step":step,"proxy_blocks_s":measured_blocks,"proxy_nonblock_residual_s":residual,"full_blocks_s":weighted_blocks,"full_fb_estimate_s":weighted_blocks+residual})
        modes[mode]["per_control_extrapolation"]=estimates
        modes[mode]["full_blocks_fb_s"]=stats([e["full_blocks_s"] for e in estimates])
        modes[mode]["full_fb_with_proxy_residual_s"]=stats([e["full_fb_estimate_s"] for e in estimates])
    mean=statistics.mean(w["fb_elapsed_s"] for w in controls)
    summary={"topology":a.topology,"gpus":gpus,"seq_len":262144,"fb_s":stats([w["fb_elapsed_s"] for w in controls]),"tps":262144/mean,"tps_per_gpu":262144/mean/gpus,
    "peak_control_allocated_gib":max(w["peak_allocated_bytes"] for w in controls)/2**30,
    "peak_control_reserved_gib":max(w["peak_reserved_bytes"] for w in controls)/2**30,
    "peak_all_windows_allocated_gib":max(w["peak_allocated_bytes"] for w in bench["windows"])/2**30,
    "layer_timings":modes,"artifacts":manifest}
    (dst/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ["layer_timings","artifacts"]},indent=2))
    print(json.dumps(modes["rank0"]["moe_average"],indent=2))
if __name__=="__main__":main()
