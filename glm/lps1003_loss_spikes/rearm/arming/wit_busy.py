import json, torch
import torch.profiler as P
from cudnn.deepseek_sparse_attention.indexer_forward import _interface as iface
assert "_get_kernel_stream" not in open(iface.__file__).read()
device = "cuda"
torch.manual_seed(1234)
TOTAL_Q, SEG_K, N_SEGS = 8192, 73728, 2
SEG_Q = TOTAL_Q // N_SEGS
case = dict(
    q=torch.randn(TOTAL_Q,32,128,dtype=torch.bfloat16,device=device),
    k=torch.randn(SEG_K*N_SEGS,1,128,dtype=torch.bfloat16,device=device),
    w=torch.randn(TOTAL_Q,32,dtype=torch.bfloat16,device=device),
    cu_q=torch.arange(0,TOTAL_Q+1,SEG_Q,dtype=torch.int32,device=device),
    cu_k=torch.arange(0,SEG_K*N_SEGS+1,SEG_K,dtype=torch.int32,device=device),
    offs=torch.tensor([SEG_K-SEG_Q]*N_SEGS,dtype=torch.int32,device=device))
def call():
    return iface.indexer_fwd(case["q"],case["k"],case["w"],ratio=1,
        cu_seqlens_q=case["cu_q"],cu_seqlens_k=case["cu_k"],
        max_seqlen_q=SEG_Q,max_seqlen_k=SEG_K,q_causal_offsets=case["offs"])
a=torch.randn(8192,8192,dtype=torch.bfloat16,device=device); c=torch.empty_like(a)
o=call(); torch.cuda.synchronize(); del o   # consume first-call latency
t_local = torch.arange(TOTAL_Q, device=device) % SEG_Q
exp_neg = (SEG_K - torch.clamp(SEG_K-SEG_Q + t_local + 1, max=SEG_K)).to(torch.int64)
with P.profile(activities=[P.ProfilerActivity.CPU,P.ProfilerActivity.CUDA]) as prof:
    for _ in range(40): torch.matmul(a,a,out=c)
    out = call()
    torch.cuda.synchronize()
prof.export_chrome_trace("/root/arming/results/trace_busy_exec1.json")
neg = torch.isneginf(out); cnt = neg.sum(1)
er = int(((cnt==SEG_K)&(exp_neg<SEG_K)).sum()); pa = int(((cnt!=exp_neg)&(cnt!=SEG_K)).sum())
print(json.dumps({"busy_exec1_erased":er,"partial":pa}))
evs=json.load(open("/root/arming/results/trace_busy_exec1.json"))["traceEvents"]
ks=sorted([e for e in evs if e.get("cat")=="kernel"],key=lambda x:x["ts"])
t0=ks[0]["ts"]
for k in ks:
    n=k["name"]
    if "FillFunctor" in n or "indexer" in n or n.startswith("kernel_cutlass"):
        print(f"{k[chr(116)+chr(115)]-t0:>10.1f}us dur={k[chr(100)+chr(117)+chr(114)]:>9.1f} stream={k["args"]["stream"]} {n[:60]}")
print("last junk matmul end:", max(k["ts"]+k["dur"]-t0 for k in ks if "FillFunctor" not in k["name"] and not k["name"].startswith("kernel_cutlass")))
