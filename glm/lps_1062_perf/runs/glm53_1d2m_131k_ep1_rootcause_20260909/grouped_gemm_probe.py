"""Balanced-shape TE BF16 expert GEMM experiment; no routing/comm/dequant/LoRA."""
import gc
import argparse
import json
import statistics
import torch
from transformer_engine.pytorch import GroupedLinear

torch.manual_seed(328)
ap = argparse.ArgumentParser()
ap.add_argument('--single-grouped-weight', action='store_true')
ap.add_argument('--torch-grouped-mm', action='store_true')
args = ap.parse_args()
rows = []
for name, experts, tokens_per_expert in (
    ('cp8ep8_shape', 32, 4096),
    ('cp8ep1_shape', 256, 512),
    ('cp1ep1_shape', 256, 4096),
):
    for projection, kin, nout in (('gate_up', 6144, 4096), ('down', 2048, 6144)):
        if args.torch_grouped_mm:
            weight = torch.randn(experts,nout,kin,device='cuda',dtype=torch.bfloat16)*0.01
            offs = torch.arange(1,experts+1,device='cuda',dtype=torch.int32)*tokens_per_expert
            def layer(inp, splits):
                return torch.nn.functional.grouped_mm(inp,weight.transpose(1,2),offs=offs)
        else:
            layer = GroupedLinear(experts, kin, nout, bias=False, params_dtype=torch.bfloat16,
                                  single_grouped_weight=args.single_grouped_weight)
            layer.requires_grad_(False)
        x = torch.randn(experts*tokens_per_expert, kin, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        grad = torch.randn(experts*tokens_per_expert, nout, device='cuda', dtype=torch.bfloat16)
        splits = [tokens_per_expert]*experts
        fwd, bwd = [], []
        for iteration in range(13):
            x.grad = None
            a, b, c = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            a.record(); y = layer(x, splits); b.record()
            y.backward(grad); c.record(); c.synchronize()
            if iteration >= 3:
                fwd.append(a.elapsed_time(b)); bwd.append(b.elapsed_time(c))
        flops = 2*experts*tokens_per_expert*kin*nout
        row = dict(name=name, projection=projection, experts=experts, tokens_per_expert=tokens_per_expert,
                   single_grouped_weight=args.single_grouped_weight,
                   torch_grouped_mm=args.torch_grouped_mm,
                   M_total=experts*tokens_per_expert, K=kin, N=nout, flops_each_fwd_or_dgrad=flops,
                   resident_weight_gib=experts*kin*nout*2/2**30,
                   fwd_ms=dict(mean=statistics.mean(fwd),sd=statistics.stdev(fwd),samples=fwd),
                   dgrad_ms=dict(mean=statistics.mean(bwd),sd=statistics.stdev(bwd),samples=bwd),
                   fwd_tflops_s=flops/(statistics.mean(fwd)*1e9),
                   dgrad_tflops_s=flops/(statistics.mean(bwd)*1e9))
        rows.append(row); print(json.dumps(row), flush=True)
        del layer, x, grad, y
        if args.torch_grouped_mm: del weight, offs
        gc.collect(); torch.cuda.empty_cache()
print('COMPLETE', json.dumps(rows), flush=True)
