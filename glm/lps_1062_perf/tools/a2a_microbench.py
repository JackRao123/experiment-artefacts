"""Standalone intra-node EP-style all-to-all microbenchmark (maxwell's
platform-vs-trainer discriminator). 8 ranks on ONE node, all_to_all_single
at payload sizes bracketing the traced trainer a2a (~25MB/peer, ~200MB/rank).

If latency here is ~ms or less -> trainer-side serialization/peer-wait.
If ~30ms at these sizes -> platform transport slow despite P2P/CUMEM.

Run: torchrun --nnodes=1 --nproc_per_node=8 a2a_microbench.py
"""
import os
import time

import torch
import torch.distributed as dist

HIDDEN = 6144
DT = torch.bfloat16


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    world = dist.get_world_size()
    assert world == 8, world

    # tokens-per-rank values bracketing the trainer's per-layer dispatch:
    # 16384 tokens x 6144 x 2B = 201MB send buffer; 128 = 1.6MB small-payload
    # control (chunking/protocol-cliff discriminator per maxwell).
    for tokens in (128, 2048, 8192, 16384):
        send = torch.randn(tokens * world // world * world, HIDDEN, device="cuda", dtype=DT)
        # equal split: each rank sends `tokens` rows total, tokens//world per peer
        send = torch.randn(tokens, HIDDEN, device="cuda", dtype=DT)
        recv = torch.empty_like(send)
        mb = tokens * HIDDEN * 2 / 1e6
        # warmup
        for _ in range(3):
            dist.all_to_all_single(recv, send)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        N = 10
        for _ in range(N):
            dist.all_to_all_single(recv, send)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / N
        if rank == 0:
            print(f"tokens/rank={tokens:6d}  send={mb:7.1f}MB  -> {dt*1e3:7.2f} ms/a2a  "
                  f"(~{mb/1e3/dt:6.1f} GB/s per-rank send rate)", flush=True)
        del send, recv
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
