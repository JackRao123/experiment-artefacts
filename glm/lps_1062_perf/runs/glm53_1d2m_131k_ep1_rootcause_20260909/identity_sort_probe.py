"""GPU numerical experiment: singleton chunk sorts are identity, including grads."""
import json
import torch
from megatron.core.transformer.moe.moe_utils import sort_chunks_by_idxs

torch.manual_seed(4701)
results = []
for experts, tokens, hidden in ((4, 37, 13), (256, 8192, 6144)):
    counts = torch.bincount(torch.randint(experts, (tokens,), device='cuda'), minlength=experts)
    # Both dispatcher transpose maps reduce to arange when ETP*EP == 1.
    chunks = torch.arange(experts, device='cuda')
    forward_map = chunks.reshape(-1, experts).T.ravel()
    reverse_map = chunks.reshape(experts, -1).T.ravel()
    for fused in (False, True):
        for with_probs in (False, True):
            x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device='cuda', requires_grad=True)
            p = torch.rand(tokens, device='cuda', requires_grad=True) if with_probs else None
            y, q = sort_chunks_by_idxs(x, counts, forward_map, probs=p, fused=fused)
            z, _ = sort_chunks_by_idxs(y, counts, reverse_map, fused=fused)
            assert torch.equal(z, x)
            g = torch.randn_like(z)
            outputs, grads = [z], [g]
            if p is not None:
                assert torch.equal(q, p)
                gp = torch.randn_like(q)
                outputs.append(q); grads.append(gp)
            torch.autograd.backward(outputs, grads)
            assert torch.equal(x.grad, g)
            if p is not None:
                assert torch.equal(p.grad, gp)
            results.append(dict(experts=experts, tokens=tokens, hidden=hidden, fused=fused, with_probs=with_probs, values_bitwise_equal=True, gradients_bitwise_equal=True))
print(json.dumps(results, indent=2))
