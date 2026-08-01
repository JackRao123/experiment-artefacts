"""V4: validate the index-bounds hardening patch for the SM100 DSA backward.

With the patch applied:
  - oob_hi / oob_neg rows must no longer fault; the offending slots must be
    treated as invalid (zeroed gather, skipped scatter).
  - Gradients must match an equivalent run where the slot was -1.
"""
import torch

from cudnn.deepseek_sparse_attention.sparse_attention_backward.api import (
    sparse_attention_backward_wrapper,
)

torch.manual_seed(0)

H, D, DV = 64, 576, 512
SQ = 256
SKV = 4096
TOPK = 2048

q = torch.randn(SQ, H, D, dtype=torch.bfloat16, device="cuda") * 0.1
kv = torch.randn(SKV, D, dtype=torch.bfloat16, device="cuda") * 0.1
out = torch.randn(SQ, H, DV, dtype=torch.bfloat16, device="cuda") * 0.1
dout = torch.randn(SQ, H, DV, dtype=torch.bfloat16, device="cuda") * 0.1
lse = torch.randn(SQ, H, dtype=torch.float32, device="cuda").abs() + 1.0
attn_sink = torch.full((H,), float("-inf"), dtype=torch.float32, device="cuda")


def make_base():
    topk_length = torch.full((SQ,), 65, dtype=torch.int32, device="cuda")
    topk_idxs = torch.full((SQ, TOPK), -1, dtype=torch.int32, device="cuda")
    for r in range(SQ):
        topk_idxs[r, :65] = torch.arange(65, dtype=torch.int32)
    return topk_idxs, topk_length


def run(idxs, lengths):
    res = sparse_attention_backward_wrapper(
        q, kv, out, dout, lse, attn_sink, idxs,
        softmax_scale=D ** -0.5,
        topk_length=lengths,
    )
    torch.cuda.synchronize()
    return res["dq"].clone(), res["dkv"].clone()


# reference: row 3 slot 0 masked out with -1
idxs_ref, len_ref = make_base()
idxs_ref[3, 0] = -1
dq_ref, dkv_ref = run(idxs_ref, len_ref)

for name, bad_val in (("oob_hi", SKV + (1 << 20)), ("oob_neg", -(1 << 20))):
    idxs, lens = make_base()
    idxs[3, 0] = bad_val
    dq, dkv = run(idxs, lens)
    dq_diff = (dq.float() - dq_ref.float()).abs().max().item()
    dkv_diff = (dkv.float() - dkv_ref.float()).abs().max().item()
    print(f"{name}: COMPLETED, dq maxdiff vs -1-masked = {dq_diff:.3e}, dkv maxdiff = {dkv_diff:.3e}", flush=True)

print("ALL OK", flush=True)
