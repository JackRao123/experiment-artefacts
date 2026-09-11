"""Isolated launch-geometry experiment; does not modify TE or trainer code."""
import json
import torch
import triton
from transformer_engine.common.triton.permutation import _sort_chunks_by_map_kernel

torch.manual_seed(1355)
kernel=getattr(_sort_chunks_by_map_kernel,'fn',_sort_chunks_by_map_kernel)
n,h=131072,6144
x=torch.randn(n,h,device='cuda',dtype=torch.bfloat16)
chunk_order=torch.randperm(256,device='cuda')
mapping=(chunk_order[:,None]*512+torch.arange(512,device='cuda')[None,:]).reshape(-1).int()
probs=torch.rand(n,device='cuda')
reference=None
results=[]
for block_size in (64,128,256,512,1024,2048,4096):
    y=torch.empty_like(x);yp=torch.empty_like(probs)
    def launch():
        kernel[(n,triton.cdiv(h,block_size))](x,mapping,probs,y,h,1,h,1,1,1,y,yp,hidden_size=h,PERMUTE_PROBS=True,BLOCK_SIZE=block_size,FORWARD=True,num_warps=4)
    launch();torch.cuda.synchronize()
    if reference is None:reference=(y.clone(),yp.clone())
    equal=bool(torch.equal(reference[0],y) and torch.equal(reference[1],yp))
    if not equal:raise RuntimeError('Launch configurations differ numerically')
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):launch()
    end.record();end.synchronize()
    result=dict(block_size=block_size,grid=[n,triton.cdiv(h,block_size)],mean_ms=start.elapsed_time(end)/20,exact_equal=equal)
    results.append(result);print(json.dumps(result),flush=True)
print('COMPLETE',flush=True)
