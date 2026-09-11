"""CPU-only safetensors-reader microbenchmark on the immutable GLM snapshot."""
import contextlib
import json
import statistics
import time
from pathlib import Path
import torch
from safetensors import safe_open

root=Path('/root/.cache/team_artifacts/huggingface/hub/models--zai-org--GLM-5.3/snapshots/187fb9fff6319062325ff825627ef6db084d9bc6')
index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
key=next(k for k in index if '.experts.0.gate_proj.weight' in k and not k.endswith('_scale_inv'))
keys=[key,key+'_scale_inv']
def uncached():
    result={}
    for key in keys:
        path=root/index[key]
        assert path.exists()
        with safe_open(path,framework='pt',device='cpu') as f:
            assert key in set(f.keys())
            result[key]=f.get_tensor(key)
    return result
def samples(fn):
    fn();times=[]
    for _ in range(30):
        t=time.perf_counter();result=fn();times.append(time.perf_counter()-t)
    return result,dict(mean_s=statistics.mean(times),median_s=statistics.median(times),samples=times)
a,ta=samples(uncached)
with contextlib.ExitStack() as stack:
    handles={fn:stack.enter_context(safe_open(root/fn,framework='pt',device='cpu')) for fn in set(index[k] for k in keys)}
    def cached():return {k:handles[index[k]].get_tensor(k) for k in keys}
    b,tb=samples(cached)
    equal=all(torch.equal(a[k].view(torch.uint8),b[k].view(torch.uint8)) for k in keys)
assert equal
print(json.dumps(dict(keys=keys,uncached=ta,cached=tb,median_speedup=ta['median_s']/tb['median_s'],bitwise_equal=equal,cuda_used=False),indent=2))
