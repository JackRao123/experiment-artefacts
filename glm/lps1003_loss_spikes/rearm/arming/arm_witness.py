#!/usr/bin/env python
"""LPS-1003 witness: per-exec kineto traces of F1/F2/S order.

Same geometry as arm_lab.py. Sequence (one process):
  boot     exec1  profiled   (first-ever call: compile + module load)
  warm1    exec2  profiled
  ec1      empty_cache -> exec3 profiled
  warm2    exec4  UNprofiled (suppression control)
  ec2      empty_cache -> exec5 UNprofiled (suppression control)
  warm3    exec6  profiled
Each exec is scanned (erased/partial rows) and the result is emitted next
to the trace path, so every trace is labeled corrupt/clean.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import torch

from cudnn.deepseek_sparse_attention.indexer_forward import _interface as iface

RESULTS = "/root/arming/results"
TAG = sys.argv[1] if len(sys.argv) > 1 else "wit"

src = open(iface.__file__).read()
assert "_get_kernel_stream" not in src, "FIXED wheel resolved — shadow failed"
assert os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") is None

device = "cuda"
torch.manual_seed(1234)
N_HEADS, HEAD_DIM = 32, 128
TOTAL_Q, SEG_K, N_SEGS = 8192, 73728, 2
SEG_Q = TOTAL_Q // N_SEGS
TOTAL_K = SEG_K * N_SEGS
OFFSET = SEG_K - SEG_Q

case = {
    "q": torch.randn(TOTAL_Q, N_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device),
    "k": torch.randn(TOTAL_K, 1, HEAD_DIM, dtype=torch.bfloat16, device=device),
    "w": torch.randn(TOTAL_Q, N_HEADS, dtype=torch.bfloat16, device=device),
    "cu_q": torch.arange(0, TOTAL_Q + 1, SEG_Q, dtype=torch.int32, device=device),
    "cu_k": torch.arange(0, TOTAL_K + 1, SEG_K, dtype=torch.int32, device=device),
    "offs": torch.tensor([OFFSET] * N_SEGS, dtype=torch.int32, device=device),
}

t_local = torch.arange(TOTAL_Q, device=device) % SEG_Q
expected_neg = (SEG_K - torch.clamp(OFFSET + t_local + 1, max=SEG_K)).to(torch.int64)
negbuf = torch.empty(TOTAL_Q, SEG_K, dtype=torch.bool, device=device)
rowcnt = torch.empty(TOTAL_Q, dtype=torch.int64, device=device)


def call():
    return iface.indexer_fwd(
        case["q"], case["k"], case["w"], ratio=1,
        cu_seqlens_q=case["cu_q"], cu_seqlens_k=case["cu_k"],
        max_seqlen_q=SEG_Q, max_seqlen_k=SEG_K,
        q_causal_offsets=case["offs"],
    )


def scan(out):
    torch.isneginf(out, out=negbuf)
    torch.sum(negbuf, dim=1, out=rowcnt)
    erased = (rowcnt == SEG_K) & (expected_neg < SEG_K)
    partial = (rowcnt != expected_neg) & ~erased
    n_er, n_pa = int(erased.sum()), int(partial.sum())
    lo = int(erased.nonzero().flatten()[0]) if n_er else -1
    return {"erased": n_er, "partial": n_pa, "lo": lo}


def emit(rec):
    print(json.dumps(rec), flush=True)


def one(tag: str, profiled: bool, lever_ec: bool):
    if lever_ec:
        torch.cuda.empty_cache()
    if profiled:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, with_stack=False,
        ) as prof:
            out = call()
            torch.cuda.synchronize()
        path = f"{RESULTS}/trace_{TAG}_{tag}.json"
        prof.export_chrome_trace(path)
    else:
        out = call()
        torch.cuda.synchronize()
        path = None
    s = scan(out)
    del out
    emit({"tag": tag, "profiled": profiled, "ec": lever_ec, **s, "trace": path})


one("boot", True, False)
one("warm1", True, False)
one("ec1", True, True)
one("warm2", False, False)
one("ec2", False, True)
one("warm3", True, False)
print("WITNESS DONE", flush=True)
