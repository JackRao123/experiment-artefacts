"""V3 (fixed): isolate the non-contiguous scores slice effect on indexer top-k.

Compares cudnn indexer_top_k on:
  (a) the non-contiguous slice returned by indexer_forward_wrapper (sk % 4 != 0)
  (b) a contiguous clone of the same values
against per-row torch.topk references computed with correct seq_lens
(causal limit per row, so no -inf inside the read region).
"""
import torch

from cudnn import DSA

torch.manual_seed(0)

B, SQ, HQ, HKV, D = 1, 64, 64, 1, 128
SK = 1029  # deliberately not a multiple of 4
TOPK = 64

q = torch.randn(B, SQ, HQ, D, dtype=torch.bfloat16, device="cuda") * 0.1
k = torch.randn(B, SK, HKV, D, dtype=torch.bfloat16, device="cuda") * 0.1
w = torch.rand(B, SQ, HQ, dtype=torch.bfloat16, device="cuda")

scores = DSA.indexer_forward_wrapper(q, k, w, ratio=1, sm_scale=1.0)["scores"]
print("scores shape:", tuple(scores.shape), "strides:", scores.stride(),
      "contiguous:", scores.is_contiguous(), flush=True)

scores2d = scores[0]  # (SQ, SK) view with row stride 1032
print("scores2d strides:", scores2d.stride(), "contiguous:", scores2d.is_contiguous(), flush=True)

# ratio=1 causal: row r may attend to columns [0, r]; valid length = r + 1.
seq_lens = torch.arange(1, SQ + 1, dtype=torch.int32, device="cuda")

def mismatches_vs_ref(got):
    bad = 0
    for r in range(SQ):
        n = min(TOPK, r + 1)
        ref = set(torch.topk(scores2d[r, : r + 1].float(), n).indices.tolist())
        gots = set(got[r][got[r] >= 0].tolist())
        if gots != ref:
            bad += 1
    return bad

ref_ok = True
try:
    tk_a = DSA.indexer_top_k_wrapper(scores2d, seq_lens, top_k=TOPK, next_n=1, return_val=False)
    got_a = tk_a["indices"]
    print(f"(a) non-contig slice:  mismatches vs per-row torch.topk = {mismatches_vs_ref(got_a)}/{SQ}", flush=True)
except Exception as e:
    print(f"(a) non-contig slice RAISED: {type(e).__name__}: {e}", flush=True)
    got_a = None

tk_b = DSA.indexer_top_k_wrapper(scores2d.contiguous(), seq_lens, top_k=TOPK, next_n=1, return_val=False)
got_b = tk_b["indices"]
print(f"(b) contiguous clone:  mismatches vs per-row torch.topk = {mismatches_vs_ref(got_b)}/{SQ}", flush=True)

if got_a is not None:
    same = int((got_a == got_b).all(dim=1).sum().item())
    print(f"(a) vs (b): identical rows = {same}/{SQ}", flush=True)
