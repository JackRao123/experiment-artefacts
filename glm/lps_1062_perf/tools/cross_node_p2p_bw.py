"""Measure direct cross-node NCCL send/recv bandwidth on matching GPU/NIC rails.

Run with two nodes and eight processes per node. By default, local GPU i is
pinned to mlx5_bond_i before NCCL initializes.
"""

import argparse
import os
import socket
import time

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pair", "all-pairs"), required=True)
    parser.add_argument("--direction", choices=("one-way", "bidirectional"), required=True)
    parser.add_argument("--size-mib", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--no-pin-local-hca", action="store_true")
    return parser.parse_args()


def transfer(
    send: torch.Tensor,
    recv: torch.Tensor,
    peer: int,
    bidirectional: bool,
    group: dist.ProcessGroup,
) -> None:
    ops = []
    if send.numel():
        ops.append(dist.P2POp(dist.isend, send, peer, group))
    if recv.numel():
        ops.append(dist.P2POp(dist.irecv, recv, peer, group))
    if bidirectional:
        assert len(ops) == 2
    for request in dist.batch_isend_irecv(ops):
        request.wait()


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if world_size != 16 or local_world_size != 8:
        raise RuntimeError(f"expected 2x8 ranks, got world={world_size}, local={local_world_size}")

    torch.cuda.set_device(local_rank)
    if not args.no_pin_local_hca:
        os.environ["NCCL_IB_HCA"] = f"mlx5_bond_{local_rank}"
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))

    pair_groups = [
        dist.new_group((i, i + local_world_size), backend="nccl")
        for i in range(local_world_size)
    ]
    pair_group = pair_groups[local_rank]

    node_rank = rank // local_world_size
    peer = (1 - node_rank) * local_world_size + local_rank
    active = args.mode == "all-pairs" or local_rank == 0
    element_count = args.size_mib * 1024 * 1024
    send = (
        torch.full(
            (element_count,), rank % 251, dtype=torch.uint8, device="cuda"
        )
        if active
        else torch.empty(0, dtype=torch.uint8, device="cuda")
    )
    receive = (
        torch.empty_like(send)
        if active
        else torch.empty(0, dtype=torch.uint8, device="cuda")
    )
    bidirectional = args.direction == "bidirectional"

    if not bidirectional and node_rank == 1:
        send = torch.empty(0, dtype=torch.uint8, device="cuda")
    if not bidirectional and node_rank == 0:
        receive = torch.empty(0, dtype=torch.uint8, device="cuda")

    dist.barrier()
    for _ in range(args.warmup):
        if active:
            transfer(send, receive, peer, bidirectional, pair_group)
    torch.cuda.synchronize()
    dist.barrier()

    started = time.perf_counter()
    for _ in range(args.iterations):
        if active:
            transfer(send, receive, peer, bidirectional, pair_group)
    torch.cuda.synchronize()
    elapsed = torch.tensor(time.perf_counter() - started, device="cuda", dtype=torch.float64)
    rank_elapsed = [torch.empty_like(elapsed) for _ in range(world_size)]
    dist.all_gather(rank_elapsed, elapsed)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)

    if active and receive.numel():
        expected = peer % 251
        sample = receive[torch.tensor([0, receive.numel() - 1], device="cuda")]
        if not bool(torch.all(sample == expected)):
            raise RuntimeError(f"rank {rank} received corrupt data from rank {peer}")

    if rank == 0:
        pairs = 1 if args.mode == "pair" else 8
        directions = 2 if bidirectional else 1
        payload_bytes = args.size_mib * 1024 * 1024 * args.iterations
        seconds = elapsed.item()
        per_direction_gbs = payload_bytes / seconds / 1e9
        aggregate_gbs = payload_bytes * pairs * directions / seconds / 1e9
        pair_gbs = []
        for local_gpu in range(pairs):
            pair_seconds = max(
                rank_elapsed[local_gpu].item(),
                rank_elapsed[local_gpu + local_world_size].item(),
            )
            pair_gbs.append(payload_bytes / pair_seconds / 1e9)
        print(
            "RESULT "
            f"mode={args.mode} direction={args.direction} size_mib={args.size_mib} "
            f"iterations={args.iterations} seconds={seconds:.6f} "
            f"per_pair_per_direction_GBps={per_direction_gbs:.3f} "
            f"aggregate_GBps={aggregate_gbs:.3f} pairs={pairs} directions={directions} "
            f"qps={os.environ.get('NCCL_IB_QPS_PER_CONNECTION', 'default')} "
            f"channels_per_peer={os.environ.get('NCCL_NCHANNELS_PER_NET_PEER', 'default')} "
            f"pin_local_hca={not args.no_pin_local_hca} "
            f"pair_GBps={','.join(f'{rate:.3f}' for rate in pair_gbs)} "
            f"host={socket.gethostname()}",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
