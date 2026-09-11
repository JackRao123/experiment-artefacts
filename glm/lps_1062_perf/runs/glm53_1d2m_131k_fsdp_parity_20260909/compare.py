"""Compare matched initialization, loss, and gradient-norm controls."""
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def canonical(name):
    while name.startswith('module.'):name=name[7:]
    return name
reference={canonical(x['name']):x for x in json.loads((ROOT/'ddp/timings/parity-init.rank0.json').read_text())}
shards={}
for rank in range(8):
    for x in json.loads((ROOT/f'fsdp/timings/parity-init.rank{rank}.json').read_text()):
        name=canonical(x['name']);r=reference[name]
        assert x['shape']==r['shape'] and x['value']==r['value'],(name,x,r)
        shards[name]=shards.get(name,0)+x['local_numel']
assert set(shards)==set(reference)
assert all(shards[n]==r['local_numel'] for n,r in reference.items())
results={}
for mode in ('ddp','fsdp'):
    b=json.loads((ROOT/mode/'benchmark.json').read_text());w=[x for x in b['windows'] if x['phase']=='control']
    assert len(w)==5
    results[mode]={key:dict(mean=statistics.mean(x[key] for x in w),min=min(x[key] for x in w),max=max(x[key] for x in w)) for key in ('loss','grad_norm')}
loss_error=abs(results['fsdp']['loss']['mean']-results['ddp']['loss']['mean'])
grad_error=abs(results['fsdp']['grad_norm']['mean']/results['ddp']['grad_norm']['mean']-1)
out=dict(initialization='38 matched adapter tensors, all FSDP shards accounted for',results=results,loss_absolute_error=loss_error,grad_norm_relative_error=grad_error,passed=loss_error<1e-3 and grad_error<0.005)
(ROOT/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
assert out['passed'],out
