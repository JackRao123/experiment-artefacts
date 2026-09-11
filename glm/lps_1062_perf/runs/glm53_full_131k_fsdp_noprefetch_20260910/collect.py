"""Retrieve and validate full-model protocol, traces, memory and layer timers."""
import argparse
import hashlib
import json
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REMOTE='/root/glm53-full-131k-fsdp-noprefetch-20260910'
ap=argparse.ArgumentParser();ap.add_argument('cp',type=int,choices=(8,4,2));a=ap.parse_args()
topology=f'cp{a.cp}ep1';label=f'glm53-full-fsdp-noprefetch-{topology}';dst=ROOT/topology/'result';dst.mkdir(exist_ok=True)
def copy(pair):
    remote,local=pair;local.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['scp','-C',f'tj-32vj99q:{remote}',str(local)],check=True)
copy((f'{REMOTE}/results/{label}.json',dst/'benchmark.json'))
b=json.loads((dst/'benchmark.json').read_text())
assert b['num_gpus']==8 and b['seq_len']==131072 and b['datums_per_window']==8//a.cp
assert b['initial_status']['data_parallel_size']==8//a.cp
assert b['final_status']['step']-b['initial_status']['step']==10
jobs=[]
for kind in ('runtime','memory'):
    meta=b[kind+'_profile_stop']
    for name in meta['files']:
        assert Path(name).name==name
        jobs.append((meta['local_path']+'/'+name,dst/kind/name))
for rank in range(8):
    for name in (f'rank{rank}.jsonl',f'gc.rank{rank}.jsonl',f'weights.rank{rank}.json',f'shapes.rank{rank}.jsonl',f'allocator.rank{rank}.jsonl'):
        jobs.append((f'{REMOTE}/{topology}/result_timings/{name}',dst/'timings'/name))
jobs += [(f'{REMOTE}/{topology}/result.log',dst/'driver.log'),(f'{REMOTE}/{topology}/result.trainer.log',dst/'trainer.log')]
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(copy,jobs))
artifacts={}
for kind in ('runtime','memory'):
    files=list((dst/kind).iterdir());assert sum(p.stat().st_size for p in files)==b[kind+'_profile_stop']['size_bytes']
    for p in files:
        with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        artifacts[str(p.relative_to(dst))]=dict(bytes=p.stat().st_size,sha256=sha)
assert len(list((dst/'runtime').glob('*.json')))==8
steps=list(range(b['initial_status']['step']+2,b['final_status']['step']+2))
control_steps=[s for s,w in zip(steps,b['windows'],strict=True) if w['phase']=='control']
rows=[json.loads(line) for p in (dst/'timings').glob('rank*.jsonl') for line in p.read_text().splitlines()]
rows=[r for r in rows if r['step'] in control_steps]
assert len(rows)==5*78*8,(len(rows),sorted(set(r['layer'] for r in rows)))
assert all(r['owns_gc_freeze'] and r['gc_enabled'] for r in rows)
def stats(v):return dict(mean=statistics.mean(v),sd=statistics.stdev(v),min=min(v),max=max(v),n=len(v))
parts=[k for k in rows[0] if k.endswith('_ms')]
layers={str(layer):{p:stats([r[p] for r in rows if r['rank']==0 and r['layer']==layer]) for p in parts} for layer in range(78)}
w=[w for w in b['windows'] if w['phase']=='control'];fb=stats([x['fb_elapsed_s'] for x in w])
summary=dict(topology=topology,datums=8//a.cp,fb_s=fb,tps_per_gpu=b['tokens_per_step']/8/fb['mean'],peak_allocated_gib=max(x['peak_allocated_bytes'] for x in w)/2**30,peak_reserved_gib=max(x['peak_reserved_bytes'] for x in w)/2**30,layers=layers,artifacts=artifacts)
(dst/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('layers','artifacts')},indent=2))
