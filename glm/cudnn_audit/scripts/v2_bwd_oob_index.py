"""V2: sparse_attention_backward dereferences out-of-range top-k index values.

Root cause (cudnn-frontend v1.26.0, sparse_attention_backward/dsa_bwd_sm100.py):
  - KV gather `_copy_kv_row` (line ~1175): mKV[topk_idx, ...] with only a
    `topk_idx >= 0` predicate in non-compact mode, and NO value predicate at
    all in compact mode (topk_length given) on the load side (lines ~1253-1260).
  - dKV fp32 atomic scatter (`reduce_dKV*` / `store_dKV*`, lines ~2261-2431):
    same >= 0-only guard.
  - `max_seqlen_kv` is plumbed into reduce_dKV (call site ~1046, param ~2084)
    and never used -- the clamp was intended but never implemented.

Contrast with FlashMLA sparse fwd, whose documented contract tolerates -1 AND
values >= s_kv. The cudnn backward tolerates neither reliably.

Expected: control (in-range) completes; oob_hi (index = SKV + 2**20) faults
with an illegal memory access in the gather or the scatter; oob_neg
(index = -2**20 in a compact row's first slot) faults on the gather.

Usage: python v2_bwd_oob_index.py [control|oob_hi|oob_neg]
"""
import sys
import torch

from cudnn.deepseek_sparse_attention.sparse_attention_backward.api import (
    sparse_attention_backward_wrapper,
)

torch.manual_seed(0)

H, D, DV = 64, 576, 512
SQ = 256
SKV = 4096
TOPK = 2048

mode = sys.argv[1] if len(sys.argv) > 1 else "control"

q = torch.randn(SQ, H, D, dtype=torch.bfloat16, device="cuda") * 0.1
kv = torch.randn(SKV, D, dtype=torch.bfloat16, device="cuda") * 0.1
out = torch.randn(SQ, H, DV, dtype=torch.bfloat16, device="cuda") * 0.1
dout = torch.randn(SQ, H, DV, dtype=torch.bfloat16, device="cuda") * 0.1
lse = torch.randn(SQ, H, dtype=torch.float32, device="cuda").abs() + 1.0
attn_sink = torch.full((H,), float("-inf"), dtype=torch.float32, device="cuda")

topk_length = torch.full((SQ,), 65, dtype=torch.int32, device="cuda")
topk_idxs = torch.full((SQ, TOPK), -1, dtype=torch.int32, device="cuda")
for r in range(SQ):
    topk_idxs[r, :65] = torch.arange(65, dtype=torch.int32)

if mode == "oob_hi":
    # Well past the KV extent: gather reads kv_base + idx*1152 (~1.2 GiB past).
    topk_idxs[3, 0] = SKV + (1 << 20)
elif mode == "oob_neg":
    # Compact mode never checks >= 0 on the load side: kv_base - 1.2 GiB.
    topk_idxs[3, 0] = -(1 << 20)

print(f"mode={mode} launching backward ...", flush=True)
res = sparse_attention_backward_wrapper(
    q, kv, out, dout, lse, attn_sink, topk_idxs,
    softmax_scale=D ** -0.5,
    topk_length=topk_length,
)
torch.cuda.synchronize()
print("COMPLETED OK", flush=True)
print("dkv norm:", res["dkv"].float().norm().item(), flush=True)
