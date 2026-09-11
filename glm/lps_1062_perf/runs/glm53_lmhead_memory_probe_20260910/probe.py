"""Numerical and peak-memory experiment for the FP32 LM-head changes."""
import gc
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['BT_MEMORY_EFFICIENT_LM_HEAD']='1'
import torch

torch.use_deterministic_algorithms(True)
torch.manual_seed(4321)
ROOT=Path(__file__).resolve().parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
old=load('old_head',ROOT/'baseline.py')
new=load('new_head',ROOT/'candidate.py')
results=[]
for dtype,m,k,n in ((torch.bfloat16,17,64,97),(torch.float16,65,256,1024),(torch.bfloat16,4096,6144,154880)):
    weight=torch.randn(n,k,device='cuda',dtype=dtype)*0.01
    hidden=torch.randn(m,k,device='cuda',dtype=dtype)
    a=torch.randn(32,k,device='cuda',dtype=dtype)*0.01
    b=torch.randn(n,32,device='cuda',dtype=dtype)*0.01
    grad=torch.randn(m,n,device='cuda',dtype=torch.float32)*0.01
    # Explicitly check the rounding-sensitive high/residual split.
    high=grad.to(dtype);residual=(grad-high.float()).to(dtype)
    h,r=new._split_logits_gradient(grad,dtype)
    assert torch.equal(high,h) and torch.equal(residual,r)
    del high,residual,h,r
    outputs=[];peaks=[]
    for impl in (old,new):
        x=hidden.clone().requires_grad_();aa=a.clone().requires_grad_();bb=b.clone().requires_grad_()
        base=SimpleNamespace(weight=weight,sequence_parallel=False,tp_group=SimpleNamespace(size=lambda:1))
        lora=SimpleNamespace(to_wrap=base,_adapter_enabled=True,adapter=None)
        lora.adapter_forward=lambda adapter,h,**kwargs: (h@aa.t())@bb.t()
        head=impl.FP32LMHead(lora,None)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=torch.cuda.memory_allocated()
        y=head(x)
        grads=torch.autograd.grad(y,(x,aa,bb),grad_outputs=grad)
        torch.cuda.synchronize();peaks.append(torch.cuda.max_memory_allocated()-start)
        outputs.append((y.cpu(),tuple(g.cpu() for g in grads)))
        del y,grads,head,lora,base,x,aa,bb
        gc.collect();torch.cuda.empty_cache()
    assert torch.equal(outputs[0][0],outputs[1][0])
    assert all(torch.equal(x,y) for x,y in zip(outputs[0][1],outputs[1][1],strict=True))
    result=dict(dtype=str(dtype),M=m,K=k,N=n,outputs_bitwise_equal=True,gradients_bitwise_equal=True,
                old_incremental_peak_gib=peaks[0]/2**30,new_incremental_peak_gib=peaks[1]/2**30)
    results.append(result);print(json.dumps(result),flush=True)
    del outputs,weight,hidden,a,b,grad
    gc.collect();torch.cuda.empty_cache()
print('COMPLETE',json.dumps(results),flush=True)
