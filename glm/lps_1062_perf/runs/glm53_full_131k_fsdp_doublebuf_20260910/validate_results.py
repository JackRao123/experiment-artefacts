"""Validate completed cases and summarize allocation behavior, without tests."""
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent
results={}
for topology in ('cp8ep1','cp4ep1','cp2ep1'):
    p=ROOT/topology/'result'
    if not (p/'summary.json').exists():continue
    b=json.loads((p/'benchmark.json').read_text());s=json.loads((p/'summary.json').read_text())
    assert b['model_layers']==78 and b['seq_len']==131072 and b['num_gpus']==8
    assert Counter(w['phase'] for w in b['windows'])==dict(warmup=3,control=5,memory_profile=1,runtime_profile=1)
    for rel,meta in s['artifacts'].items():
        f=p/rel;assert f.stat().st_size==meta['bytes']
        with f.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==meta['sha256']
    steps=range(b['initial_status']['step']+2,b['final_status']['step']+2)
    controls={step for step,w in zip(steps,b['windows'],strict=True) if w['phase']=='control'}
    a=[json.loads(line) for f in (p/'timings').glob('allocator.rank*.jsonl') for line in f.read_text().splitlines()]
    a=[x for x in a if x['step'] in controls];assert len(a)==40
    totals={key:sum(x['deltas'][key] for x in a) for key in a[0]['deltas']}
    with (p/'memory/memory.rank0.pickle').open('rb') as f:memory=pickle.load(f)
    events=[e for xs in memory['device_traces'] for e in xs];assert events
    mapping={action:dict(count=sum(e['action']==action for e in events),bytes=sum(e.get('size',0) for e in events if e['action']==action)) for action in ('segment_map','segment_unmap')}
    results[topology]=dict(artifact_integrity='passed',runtime_ranks=len(b['runtime_profile_stop']['files']),control_rank_windows=len(a),allocator_delta_totals=totals,memory_mapping=mapping,fb_s=s['fb_s'],tps_per_gpu=s['tps_per_gpu'],peak_allocated_gib=s['peak_allocated_gib'],peak_reserved_gib=s['peak_reserved_gib'])
(ROOT/'validation.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
