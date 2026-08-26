"""Benchmark a balanced top-k=8 token all-to-all across 2x8 GPUs."""

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist


BASE_TOKENS = 16_384
TOP_K = 8
HIDDEN_SIZE = 6_144


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--no-pin-local-hca", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if world_size != 16 or local_world_size != 8:
        raise RuntimeError(
            f"expected 2 nodes x 8 GPUs, got world={world_size}, "
            f"local_world={local_world_size}"
        )

    torch.cuda.set_device(local_rank)
    if not args.no_pin_local_hca:
        os.environ["NCCL_IB_HCA"] = f"mlx5_bond_{local_rank}"
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))

    routed_tokens = BASE_TOKENS * TOP_K
    if routed_tokens % world_size:
        raise RuntimeError("routed token count must divide evenly across ranks")
    chunk_tokens = routed_tokens // world_size

    send = torch.full(
        (routed_tokens, HIDDEN_SIZE),
        rank,
        dtype=torch.bfloat16,
        device="cuda",
    )
    receive = torch.empty_like(send)

    for _ in range(args.warmup):
        dist.all_to_all_single(receive, send)
    torch.cuda.synchronize()

    expected = torch.arange(world_size, dtype=torch.bfloat16, device="cuda")
    observed = receive[::chunk_tokens, 0]
    if not bool(torch.equal(observed, expected)):
        raise RuntimeError(
            f"rank {rank} received corrupt chunks: {observed.tolist()}"
        )

    dist.barrier()
    durations_sec = []
    for _ in range(args.iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        dist.all_to_all_single(receive, send)
        torch.cuda.synchronize()
        durations_sec.append(time.perf_counter() - started)

    local_durations = torch.tensor(
        durations_sec, dtype=torch.float64, device="cuda"
    )
    all_durations = [torch.empty_like(local_durations) for _ in range(world_size)]
    dist.all_gather(all_durations, local_durations)

    if rank == 0:
        duration_matrix = torch.stack(all_durations).cpu()
        collective_ms = duration_matrix.max(dim=0).values * 1_000
        rank_spread_ms = (
            duration_matrix.max(dim=0).values
            - duration_matrix.min(dim=0).values
        ) * 1_000

        tensor_bytes = send.numel() * send.element_size()
        chunk_bytes = tensor_bytes // world_size
        cross_node_bytes_per_rank = chunk_bytes * 8
        cross_node_bytes_per_node = cross_node_bytes_per_rank * 8
        cross_node_bytes_combined = cross_node_bytes_per_node * 2
        mean_sec = collective_ms.mean().item() / 1_000
        result = {
            "world_size": world_size,
            "tensor_shape": list(send.shape),
            "dtype": str(send.dtype),
            "tensor_mib_per_gpu": tensor_bytes / 2**20,
            "chunk_shape": [chunk_tokens, HIDDEN_SIZE],
            "chunk_mib": chunk_bytes / 2**20,
            "cross_node_gib_per_rank": cross_node_bytes_per_rank / 2**30,
            "cross_node_gib_combined": cross_node_bytes_combined / 2**30,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "latency_ms": {
                "min": collective_ms.min().item(),
                "median": torch.quantile(collective_ms, 0.5).item(),
                "mean": collective_ms.mean().item(),
                "p95": torch.quantile(collective_ms, 0.95).item(),
                "max": collective_ms.max().item(),
            },
            "mean_rank_spread_ms": statistics.mean(rank_spread_ms.tolist()),
            "per_node_one_way_GBps": cross_node_bytes_per_node
            / mean_sec
            / 1e9,
            "combined_cross_node_GBps": cross_node_bytes_combined
            / mean_sec
            / 1e9,
            "per_rank_tensor_GBps": tensor_bytes / mean_sec / 1e9,
            "nccl_ib_qps_per_connection": os.environ.get(
                "NCCL_IB_QPS_PER_CONNECTION", "default"
            ),
            "nccl_nchannels_per_net_peer": os.environ.get(
                "NCCL_NCHANNELS_PER_NET_PEER", "default"
            ),
            "pin_local_hca": not args.no_pin_local_hca,
        }

        for iteration, latency_ms in enumerate(collective_ms.tolist(), start=1):
            print(f"iteration={iteration:02d} latency_ms={latency_ms:.3f}")
        print("RESULT_JSON " + json.dumps(result, sort_keys=True))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
