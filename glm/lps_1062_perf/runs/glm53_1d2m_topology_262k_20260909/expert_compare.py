"""Compare routed forward regions and grouped-linear backward from saved Perfetto CSVs."""
import csv, json, statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):
    with p.open() as f: return list(csv.DictReader(f))
def interval(r): return float(r['ts_us']),float(r['ts_us'])+float(r['dur_us'])
def contains(a,b):
    x,y=interval(a);u,v=interval(b)
    return x<=u and v<=y+0.01
def gemm(k): return any(x in k['name'].lower() for x in ['gemm','nvjet','cublas','grouped_mm'])
def stats(ks):
    if not ks:return {}
    gs=[k for k in ks if gemm(k)]
    return dict(kernels=len(ks),gpu_span_ms=(max(interval(k)[1] for k in ks)-min(interval(k)[0] for k in ks))/1000,kernel_sum_ms=sum(float(k['dur_us']) for k in ks)/1000,gemm_calls=len(gs),gemm_ms=sum(float(k['dur_us']) for k in gs)/1000)
def main():
    out={}
    for topology in ['cp8ep8','cp8ep1']:
        d=ROOT/topology/'analysis';ks=read(d/'kernels.csv');ops=read(d/'owners.csv');blocks=read(d/'blocks.csv')
        owners={r['external_id']:r for r in ops if r['external_id']!='[NULL]'}
        def kernels_within(op):return [k for k in ks if k['external_id'] in owners and contains(op,owners[k['external_id']])]
        regions=[]
        for b in blocks:
            if not (b['name'].endswith('/forward') or b['name'].endswith('/recompute')):continue
            routes=[r for r in ops if r['name']=='RouterGatingLinearFunction' and contains(b,r)]
            if not routes:continue
            assert len(routes)==1
            route=routes[0];end=interval(route)[1]
            selected=[k for k in kernels_within(b) if interval(owners[k['external_id']])[0]>=end]
            lo=min(interval(k)[0] for k in selected);hi=max(interval(k)[1] for k in selected)
            # TE's no-grad grouped GEMMs may have no CPU-op external-id owner.
            # Complete the GPU interval using all streams, not just owned kernels.
            complete=[k for k in ks if lo<=interval(k)[0] and interval(k)[1]<=hi+0.01]
            regions.append({'scope':b['name'],'definition':'GPU interval after router gating through block end; shared expert before router excluded; includes postprocessing; boundaries inferred from owned kernels',**stats(complete)})
        grouped=[]
        for op in ops:
            if op['name']=='_GroupedLinearBackward':grouped.append({'cpu_ts':op['ts_us'],**stats(kernels_within(op))})
        reductions=[dict(name=k['name'],start_us=float(k['ts_us']),ms=float(k['dur_us'])/1000) for k in ks if 'ReduceScatter' in k['name']]
        controls=read_timings(ROOT/topology/'layer_timings')
        out[topology]={'routed_regions':regions,'grouped_linear_backward':grouped,'reduce_scatter':reductions,'control_max_by_rank':controls}
    (ROOT/'expert_comparison.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
def read_timings(folder):
    rows=[json.loads(line) for p in folder.glob('rank*.jsonl') for line in p.read_text().splitlines()]
    result=[]
    for step in range(3,8):
        rr=[r for r in rows if r['step']==step]
        worst=sorted(rr,key=lambda r:r['backward_including_recompute_ms'],reverse=True)[:3]
        result.append({'worker_step':step,'worst_block_backward':worst})
    return result
if __name__=='__main__':main()
