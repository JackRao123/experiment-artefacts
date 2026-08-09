#!/usr/bin/env python3
"""Standalone repro attempt: TE cuDNN FusedAttention forward nondeterminism
under context-parallel p2p, THD layout. No megatron, no model weights.

Geometry mirrors the LPS-1063 Nemotron-3-Ultra probe forward (from the
trainer's get_attention_backend log + probe shapes):
  qkv_format=thd, per-rank q heads=8, kv heads=1 (MQA slice), head_dim=128,
  attn_mask_type=padding_causal, window=(-1,0), bf16, cp_size=4 (p2p ring),
  one packed sequence of 698 tokens padded to 704 (=2*cp*88), softmax scale
  default, no dropout, forward only (inference_mode).

Each CP rank holds load-balanced chunks [r, 2*cp-1-r] of 88 tokens each.
Runs N forwards on identical inputs and compares outputs bitwise across
iterations; reports distinct-count and differing rows per rank.

Launch (on one 8-GPU node, uses 4):
  torchrun --standalone --nproc-per-node=4 standalone_te_cp4_repro.py
"""
from __future__ import annotations

import hashlib
import os
import sys

import torch
import torch.distributed as dist

import transformer_engine.pytorch as te


SEQ_REAL = 698
T_TOTAL = 704       # padded; must be divisible by 2*CP
HEADS_Q = 8
HEADS_KV = 1
HEAD_DIM = 128
N_ITERS = 20
SEEDS = [16, 17, 18]


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()[:16]


def main() -> int:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    CP = world
    CHUNK = T_TOTAL // (2 * CP)
    T_LOCAL = 2 * CHUNK
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    if rank == 0:
        import transformer_engine
        print(f"CP={CP} chunk={CHUNK} TE={transformer_engine.__version__} "
              f"torch={torch.__version__} cudnn={torch.backends.cudnn.version()} "
              f"NVTE_ALLOW_NONDETERMINISTIC_ALGO={os.environ.get('NVTE_ALLOW_NONDETERMINISTIC_ALGO')}",
              flush=True)

    cp_group = dist.new_group(ranks=list(range(CP)), backend="nccl")
    cp_stream = torch.cuda.Stream(device=dev)

    attn = te.DotProductAttention(
        num_attention_heads=HEADS_Q,
        kv_channels=HEAD_DIM,
        num_gqa_groups=HEADS_KV,
        attention_dropout=0.0,
        attn_mask_type="padding_causal",
        qkv_format="thd",
        softmax_scale=None,
    ).to(dev)
    attn.set_context_parallel_group(
        cp_group, list(range(CP)), cp_stream, cp_comm_type="p2p"
    )

    t_total = 2 * CP * CHUNK  # 704
    cu_q = torch.tensor([0, SEQ_REAL], dtype=torch.int32, device=dev)
    cu_q_padded = torch.tensor([0, t_total], dtype=torch.int32, device=dev)

    overall_rc = 0
    for seed in SEEDS:
        g = torch.Generator(device="cpu").manual_seed(seed)
        qf = torch.randn(t_total, HEADS_Q, HEAD_DIM, generator=g).bfloat16()
        kf = torch.randn(t_total, HEADS_KV, HEAD_DIM, generator=g).bfloat16()
        vf = torch.randn(t_total, HEADS_KV, HEAD_DIM, generator=g).bfloat16()

        # load-balanced CP shard: chunks [rank, 2*CP-1-rank]
        idx = torch.cat(
            [
                torch.arange(rank * CHUNK, (rank + 1) * CHUNK),
                torch.arange((2 * CP - 1 - rank) * CHUNK, (2 * CP - rank) * CHUNK),
            ]
        )
        q = qf[idx].to(dev).requires_grad_(False)
        k = kf[idx].to(dev)
        v = vf[idx].to(dev)

        outs = []
        with torch.inference_mode():
            for _ in range(N_ITERS):
                o = attn(
                    q, k, v,
                    cu_seqlens_q=cu_q,
                    cu_seqlens_kv=cu_q,
                    cu_seqlens_q_padded=cu_q_padded,
                    cu_seqlens_kv_padded=cu_q_padded,
                    max_seqlen_q=t_total,
                    max_seqlen_kv=t_total,
                )
                outs.append(o.clone())

        hashes = [sha(o) for o in outs]
        distinct = len(set(hashes))
        rows = set()
        if distinct > 1:
            base = outs[0].view(T_LOCAL, -1).float()
            for o in outs[1:]:
                d = (o.view(T_LOCAL, -1).float() - base).abs().amax(dim=1)
                rows.update(torch.nonzero(d > 0).flatten().tolist())
        print(
            f"[seed {seed}] rank {rank}: {distinct} distinct outputs over "
            f"{N_ITERS} iters; wobble rows: {sorted(rows)[:10]}",
            flush=True,
        )
        flag = torch.tensor([1 if distinct > 1 else 0], device=dev)
        dist.all_reduce(flag)
        if rank == 0:
            verdict = "NONDETERMINISTIC" if flag.item() > 0 else "deterministic"
            print(f"[seed {seed}] GLOBAL: {verdict} ({flag.item()}/{CP} ranks)",
                  flush=True)
        if flag.item() > 0:
            overall_rc = 1

    dist.destroy_process_group()
    return overall_rc


if __name__ == "__main__":
    sys.exit(main())
