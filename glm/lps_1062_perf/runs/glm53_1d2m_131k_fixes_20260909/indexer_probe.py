"""Numerical/timing experiment for the causal-offset route; not a test-suite addition."""
import json
import time
import torch
from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels as kernels

torch.manual_seed(1355)
torch.backends.cuda.matmul.allow_tf32=False
kernels._INDEXER_SCORE_CHUNK_MAX_BYTES=32*1024*1024
results=[]
for s,k in [(2048,128),(4096,2048),(8192,2048)]:
    q=torch.randn(1,s,32,128,device="cuda",dtype=torch.bfloat16)
    keys=torch.randn(1,s,128,device="cuda",dtype=torch.bfloat16)
    weights=torch.randn(1,s,32,device="cuda",dtype=torch.bfloat16)/32**0.5
    starts=torch.zeros(s,device="cuda",dtype=torch.int64)
    ends=torch.arange(1,s+1,device="cuda",dtype=torch.int64)
    outputs=[];times=[]
    for fast in [False,True]:
        torch.cuda.synchronize();t=time.perf_counter()
        outputs.append(kernels._indexer_topk_bshd(q,keys,weights,k,varlen_starts=starts,varlen_ends=ends,return_scores=False,varlen_is_plain_causal=fast))
        torch.cuda.synchronize();times.append(time.perf_counter()-t)
    a,b=outputs[0][0],outputs[1][0]
    def mask(ids):
        m=torch.zeros((1,s,s),device="cuda",dtype=torch.int32)
        m.scatter_add_(2,ids.clamp_min(0).long(),(ids>=0).int())
        return m>0
    ma,mb=mask(a),mask(b)
    overlap=((ma&mb).sum()/ma.sum()).item()
    valid_counts_equal=bool(torch.equal((a>=0).sum(-1),(b>=0).sum(-1)))
    causal=bool(((b<ends.view(1,s,1))|(b<0)).all().item())
    results.append(dict(sequence=s,topk=k,reference_seconds=times[0],fast_seconds=times[1],selected_key_overlap=overlap,valid_counts_equal=valid_counts_equal,causal=causal))
    print(json.dumps(results[-1]),flush=True)
    if not valid_counts_equal or not causal or overlap<0.999:
        raise RuntimeError("Causal-offset numerical probe needs investigation")
    del q,keys,weights,a,b,ma,mb,outputs
print("PROBE_COMPLETE",flush=True)
