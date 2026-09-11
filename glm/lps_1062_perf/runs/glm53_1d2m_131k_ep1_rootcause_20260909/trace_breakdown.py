"""Perfetto GPU-scope accounting for the BF16/storage comparison."""
import argparse
import csv
import io
import json
import subprocess
from collections import defaultdict
from pathlib import Path

SQL="""WITH regions AS MATERIALIZED (
 SELECT name,MIN(ts) ts,MAX(ts+dur) en FROM slice
 WHERE category='gpu_user_annotation' AND dur>0
 AND (name GLOB 'layer_*/forward/*' OR name GLOB 'layer_*/recompute/*') GROUP BY name)
SELECT r.name scope,k.id,k.ts/1e3 ts_us,k.dur/1e3 dur_us,k.name,k.track_id
FROM regions r JOIN slice k ON k.ts>=r.ts AND k.ts+k.dur<=r.en
WHERE k.category='kernel' AND k.dur>0 ORDER BY r.name,k.ts"""
def union(intervals):
    total=0.;end=None
    for a,b in sorted(intervals):
        if end is None or a>end:total+=b-a
        elif b>end:total+=b-end
        end=max(end,b) if end is not None else b
    return total
def category(name):
    n=name.lower()
    if 'to_copy_mul_unsqueeze_view' in n:return 'weight_dequant_signature'
    if 'nccl' in n:return 'nccl'
    if 'dispatch' in n or 'combine' in n:return 'dispatch_combine'
    if 'gemm' in n or 'nvjet' in n or 'cublas' in n:return 'gemm'
    if any(x in n for x in ['permute','chunk_sort','sort_chunk','topk','top_k']):return 'permutation_topk'
    if 'indexer_fwd' in n:return 'indexer_score'
    if 'sparse_attn' in n or 'attention_backwarddsa' in n:return 'sparse_attention'
    return 'other'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);ap.add_argument('--rank',type=int,default=0);a=ap.parse_args()
    traces=list((a.folder/'runtime').glob(f'*rank{a.rank}*.json'))
    assert len(traces)==1,traces
    out=a.folder/'analysis';out.mkdir(exist_ok=True)
    (out/'gpu-scopes.sql').write_text(SQL+';\n')
    p=subprocess.run(['/Users/jackrao/bin/trace_processor','query',str(traces[0]),SQL],capture_output=True,text=True,check=True)
    prefix='gpu-scopes' if a.rank==0 else f'gpu-scopes-rank{a.rank}'
    (out/f'{prefix}.csv').write_text(p.stdout);(out/f'{prefix}.log').write_text(p.stderr)
    scopes=defaultdict(dict)
    for r in csv.DictReader(io.StringIO(p.stdout)):scopes[r['scope']][r['id']]=r
    results={}
    for scope,by_id in scopes.items():
        rows=list(by_id.values());groups=defaultdict(lambda:{'calls':0,'sum_ms':0.})
        names=defaultdict(lambda:{'calls':0,'sum_ms':0.})
        intervals=[]
        for r in rows:
            t=float(r['ts_us']);d=float(r['dur_us']);intervals.append((t,t+d))
            g=groups[category(r['name'])];g['calls']+=1;g['sum_ms']+=d/1000
            n=names[r['name']];n['calls']+=1;n['sum_ms']+=d/1000
        span=(max(b for _,b in intervals)-min(a for a,_ in intervals))/1000
        busy=union(intervals)/1000
        results[scope]=dict(kernel_span_ms=span,kernel_union_ms=busy,within_scope_kernel_gaps_ms=span-busy,groups=dict(groups),kernel_names=dict(names))
    (out/f'{prefix}.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({k:{n:x for n,x in v.items() if n!='kernel_names'} for k,v in results.items() if '/mlp' in k},indent=2))
if __name__=='__main__':main()
