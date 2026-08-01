"""V1: sparse_attention_backward with topk_length[row] == 0 (empty row).

Root cause (cudnn-frontend v1.26.0, sparse_attention_backward/dsa_bwd_sm100.py):
  topk = mTopkLength[token_idx]; tile_count = ceil_div(topk, 64) == 0
  load_KV prologue runs unconditionally with tile_index = -1:
    idx = -64 + row  (negative); guard `idx < max_topk` is TRUE for negatives
    -> OOB read of mTopkIdxs before the row
  full_tiles = (0 % 64) == 0 -> _load_kv_rows(is_first=False)
    -> compact mode: UNCONDITIONAL _copy_kv_row with the garbage index
  Upstream #439 also reports a zero-tile pipeline deadlock (hang).

Expected: control run (all lengths >= 1) completes; experimental run
(one zero-length row, placed at token 0) hangs (timeout) or faults.

Usage: python v1_bwd_empty_row.py [control|zero]
"""
import sys
import torch

from cudnn.deepseek_sparse_attention.sparse_attention_backward.api import (
    sparse_attention_backward_wrapper,
)

torch.manual_seed(0)

H, D, DV = 64, 576, 512
SQ = 256          # total query rows
SKV = 4096        # total kv rows
TOPK = 2048       # production topk width

mode = sys.argv[1] if len(sys.argv) > 1 else "control"

q = torch.randn(SQ, H, D, dtype=torch.bfloat16, device="cuda") * 0.1
kv = torch.randn(SKV, D, dtype=torch.bfloat16, device="cuda") * 0.1
out = torch.randn(SQ, H, DV, dtype=torch.bfloat16, device="cuda") * 0.1
dout = torch.randn(SQ, H, DV, dtype=torch.bfloat16, device="cuda") * 0.1
lse = torch.randn(SQ, H, dtype=torch.float32, device="cuda").abs() + 1.0
attn_sink = torch.full((H,), float("-inf"), dtype=torch.float32, device="cuda")

# Every row attends to its causal prefix (small, in-bounds); compact layout.
topk_length = torch.full((SQ,), 65, dtype=torch.int32, device="cuda")
topk_idxs = torch.full((SQ, TOPK), -1, dtype=torch.int32, device="cuda")
for r in range(SQ):
    n = 65
    topk_idxs[r, :n] = torch.arange(n, dtype=torch.int32) % (SKV // 2)

if mode == "zero":
    topk_length[0] = 0  # empty row at token 0 -> tile_count = 0 -> prologue OOB

print(f"mode={mode} launching backward ...", flush=True)
res = sparse_attention_backward_wrapper(
    q, kv, out, dout, lse, attn_sink, topk_idxs,
    softmax_scale=D ** -0.5,
    topk_length=topk_length,
)
torch.cuda.synchronize()
print("COMPLETED OK", flush=True)
print("dq norm:", res["dq"].float().norm().item(), flush=True)
