"""Collect one 131k attempt, including every requested rank's trace and timers."""
import argparse
import hashlib
import json
import statistics
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REMOTE='/root/glm53-131k-fixes-20260909'
HOST='tj-32vj99q'
def copy(pair):
    remote,local=pair
    local.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['scp','-C',f'{HOST}:{remote}',str(local)],check=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('topology');ap.add_argument('attempt');ap.add_argument('label');args=ap.parse_args()
    dst=ROOT/args.topology/args.attempt;dst.mkdir(parents=True,exist_ok=True)
    copy((f'{REMOTE}/results/{args.label}.json',dst/'benchmark.json'))
    b=json.loads((dst/'benchmark.json').read_text());gpus=b['num_gpus']
    assert b['seq_len']==131072 and b['initial_status']['world_size']==gpus
    assert Counter(w['phase'] for w in b['windows'])==dict(warmup=b.get('warmup_repeats',1),control=5,memory_profile=1,runtime_profile=1)
    jobs=[]
    for kind in ['runtime','memory']:
        info=b[f'{kind}_profile_stop']
        for name in info['files']:
            assert Path(name).name==name
            jobs.append((f"{info['local_path']}/{name}",dst/kind/name))
    for rank in range(gpus):
        for name in [f'rank{rank}.jsonl',f'gc.rank{rank}.jsonl']:
            jobs.append((f'{REMOTE}/{args.topology}/{args.attempt}_timings/{name}',dst/'timings'/name))
    jobs.extend([(f'{REMOTE}/{args.topology}/{args.attempt}.trainer.log',dst/'trainer.log'),(f'{REMOTE}/{args.topology}/{args.attempt}.log',dst/'driver.log')])
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(copy,jobs))
    artifacts={}
    for kind in ['runtime','memory']:
        files=list((dst/kind).iterdir())
        assert sum(p.stat().st_size for p in files)==b[f'{kind}_profile_stop']['size_bytes']
        for p in files:
            with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
            artifacts[str(p.relative_to(dst))]=dict(bytes=p.stat().st_size,sha256=sha)
    assert len(b['runtime_profile_stop']['files'])==gpus
    rows=[json.loads(line) for p in (dst/'timings').glob('rank*.jsonl') for line in p.read_text().splitlines()]
    # One internal startup fb precedes HTTP steps; every benchmark fb has an
    # optimizer step. Use protocol counters so later stress calls cannot shift
    # the headline window while its artifacts are being retrieved.
    steps=list(range(b['initial_status']['step']+2,b['final_status']['step']+2))
    assert len(steps)==len(b['windows'])
    control_steps=[s for s,w in zip(steps,b['windows']) if w['phase']=='control']
    controls=[r for r in rows if r['step'] in control_steps]
    assert len(controls)==5*3*gpus
    if args.attempt=='result':
        assert all(r['owns_gc_freeze'] and r['gc_enabled'] for r in controls)
    metrics=[k for k in rows[0] if k.endswith('_ms')]
    def stats(v):return dict(mean=statistics.mean(v),sd=statistics.stdev(v),min=min(v),max=max(v),n=len(v))
    layers={str(l):{k:stats([r[k] for r in controls if r['layer']==l and r['rank']==0]) for k in metrics} for l in range(3)}
    windows=[w for w in b['windows'] if w['phase']=='control'];times=[w['fb_elapsed_s'] for w in windows]
    gcrows=[json.loads(line) for p in (dst/'timings').glob('gc.rank*.jsonl') for line in p.read_text().splitlines()]
    gccontrols=[r for r in gcrows if r['step'] in control_steps and r.get('in_fb',True)]
    summary=dict(topology=args.topology,attempt=args.attempt,fb_s=stats(times),tps_per_gpu=131072/statistics.mean(times)/gpus,tps=131072/statistics.mean(times),peak_allocated_gib=max(w['peak_allocated_bytes'] for w in windows)/2**30,peak_reserved_gib=max(w['peak_reserved_bytes'] for w in windows)/2**30,layers=layers,gc_slowest_controls=sorted(gccontrols,key=lambda r:-r['duration_ms'])[:20],artifacts=artifacts)
    summary['full_blocks_seconds']={k:sum(layers[str(l)][k]['mean']*n for l,n in enumerate([3,57,18]))/1000 for k in metrics}
    (dst/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['layers','artifacts']},indent=2))
if __name__=='__main__':main()
