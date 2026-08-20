"""1-GPU smoke: compile + run the cuDNN DSA indexer fwd (sm100) — the exact
path that crashed boot warmup on the torn cutlass overlay (elect_sync skew).
GLM-5.2 indexer shapes: 32 q heads, 8 kv heads (ratio 4), head_dim 128.
"""
import torch
from cudnn.deepseek_sparse_attention.indexer_forward.api import indexer_forward_wrapper

torch.manual_seed(0)
q = torch.randn(1, 256, 32, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 256, 8, 128, device="cuda", dtype=torch.bfloat16)
w = torch.rand(1, 256, 32, device="cuda", dtype=torch.bfloat16)
scores = indexer_forward_wrapper(q, k, w, ratio=4, sm_scale=1.0)["scores"]
torch.cuda.synchronize()
s = float(scores.abs().sum())
assert scores.shape == (1, 256, 256) and scores.dtype == torch.float32
assert s == s and s > 0  # finite, nonzero
print(f"INDEXER COMPILE+RUN OK  scores{tuple(scores.shape)} sum={s:.1f}")
