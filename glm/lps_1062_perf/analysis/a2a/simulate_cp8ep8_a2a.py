#!/usr/bin/env python3
"""Benchmark the CP8/EP8 token-dispatch AllToAllV on one 8-GPU node.

Run with:

    torchrun --standalone --nproc_per_node=8 simulate_cp8ep8_a2a.py

Each trial queues a NCCL barrier immediately before the AllToAllV so the GPU
streams enter the measured operation together as closely as NCCL permits.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from datetime import timedelta

import torch
import torch.distributed as dist


ROUTING_MATRIX = [
    [30281, 356, 30692, 13472, 416, 32459, 2538, 20858],
    [30120, 371, 30913, 13572, 416, 32337, 2453, 20890],
    [30110, 361, 30946, 13554, 394, 32367, 2506, 20834],
    [30041, 331, 31111, 13543, 379, 32292, 2552, 20823],
    [30099, 314, 31151, 13507, 357, 32239, 2569, 20836],
    [30040, 356, 31125, 13572, 369, 32431, 2449, 20730],
    [30349, 331, 30995, 13541, 327, 32149, 2538, 20842],
    [30090, 352, 30899, 13567, 384, 32374, 2576, 20830],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument(
        "--routing",
        choices=("exact", "balanced"),
        default="exact",
        help="Use the traced routing matrix or an equal 16384-to-each-rank matrix.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.hidden_size < 1:
        raise ValueError("--hidden-size must be positive")


def routing_matrix(mode: str) -> list[list[int]]:
    if mode == "balanced":
        return [[16384] * 8 for _ in range(8)]
    return ROUTING_MATRIX


def gather_rank_tensor(local: torch.Tensor, world_size: int) -> torch.Tensor | None:
    gathered = torch.empty(
        world_size * local.numel(), dtype=local.dtype, device=local.device
    )
    dist.all_gather_into_tensor(gathered, local.contiguous())
    if dist.get_rank() != 0:
        return None
    return gathered.view(world_size, *local.shape).cpu()


def main() -> None:
    args = parse_args()
    validate_args(args)

    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Launch with torchrun --standalone --nproc_per_node=8")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        timeout=timedelta(minutes=5),
        device_id=torch.device("cuda", local_rank),
    )

    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != 8:
            raise RuntimeError(f"This benchmark requires exactly 8 ranks, got {world_size}")

        matrix = routing_matrix(args.routing)
        send_splits = matrix[rank]
        recv_splits = [matrix[source][rank] for source in range(world_size)]
        if sum(send_splits) != 131072:
            raise RuntimeError(f"rank {rank} send splits sum to {sum(send_splits)}, not 131072")

        dtype = torch.bfloat16
        input_tensor = torch.full(
            (sum(send_splits), args.hidden_size),
            fill_value=rank,
            dtype=dtype,
            device=local_rank,
        )
        output_tensor = torch.empty(
            (sum(recv_splits), args.hidden_size),
            dtype=dtype,
            device=local_rank,
        )

        def run_once() -> tuple[float, int]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            # Queue the barrier and benchmark back-to-back on the same stream.
            # The barrier removes model-side arrival skew from the experiment.
            dist.barrier()
            issue_ns = time.perf_counter_ns()
            start.record()
            dist.all_to_all_single(
                output_tensor,
                input_tensor,
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
            )
            end.record()
            end.synchronize()
            return start.elapsed_time(end), issue_ns

        for _ in range(args.warmups):
            run_once()

        elapsed_ms: list[float] = []
        issue_ns: list[int] = []
        for _ in range(args.repeats):
            elapsed, issued = run_once()
            elapsed_ms.append(elapsed)
            issue_ns.append(issued)

        elapsed_tensor = torch.tensor(elapsed_ms, dtype=torch.float64, device=local_rank)
        issue_tensor = torch.tensor(issue_ns, dtype=torch.int64, device=local_rank)
        all_elapsed = gather_rank_tensor(elapsed_tensor, world_size)
        all_issues = gather_rank_tensor(issue_tensor, world_size)

        if rank == 0:
            assert all_elapsed is not None
            assert all_issues is not None
            bytes_per_token = args.hidden_size * torch.empty((), dtype=dtype).element_size()
            print(
                f"routing={args.routing} hidden={args.hidden_size} dtype={dtype} "
                f"warmups={args.warmups} repeats={args.repeats}"
            )
            print(
                f"{'rank':>4} {'sent':>8} {'received':>8} {'remote GB':>10} "
                f"{'median ms':>10} {'min ms':>10} {'max ms':>10}"
            )
            for current_rank in range(world_size):
                sent = sum(matrix[current_rank])
                received = sum(matrix[source][current_rank] for source in range(world_size))
                self_tokens = matrix[current_rank][current_rank]
                remote_tokens = max(sent - self_tokens, received - self_tokens)
                remote_gb = remote_tokens * bytes_per_token / 1e9
                samples = all_elapsed[current_rank].tolist()
                print(
                    f"{current_rank:>4} {sent:>8} {received:>8} {remote_gb:>10.3f} "
                    f"{statistics.median(samples):>10.3f} "
                    f"{min(samples):>10.3f} {max(samples):>10.3f}"
                )

            print()
            print(
                f"{'trial':>5} {'issue skew ms':>14} {'max rank ms':>12} "
                f"{'min rank ms':>12}"
            )
            for trial in range(args.repeats):
                starts = all_issues[:, trial]
                durations = all_elapsed[:, trial]
                issue_skew_ms = (starts.max() - starts.min()).item() / 1e6
                print(
                    f"{trial:>5} {issue_skew_ms:>14.3f} "
                    f"{durations.max().item():>12.3f} {durations.min().item():>12.3f}"
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
