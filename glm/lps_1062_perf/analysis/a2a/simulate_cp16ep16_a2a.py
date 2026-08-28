#!/usr/bin/env python3
"""Benchmark the CP16/EP16 token-dispatch AllToAllV on 2x8 GPUs.

Launch one torchrun process on each node with matching rendezvous arguments:

    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 ... simulate_cp16ep16_a2a.py
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 ... simulate_cp16ep16_a2a.py

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
    [4914, 10228, 138, 21, 14243, 1160, 6577, 112, 200, 2, 10747, 5462, 1282, 0, 9518, 932],
    [4909, 10229, 156, 41, 14167, 1121, 6661, 122, 212, 2, 10807, 5443, 1257, 0, 9464, 945],
    [4864, 10238, 163, 33, 14264, 1232, 6674, 128, 192, 0, 10717, 5391, 1202, 0, 9542, 896],
    [4860, 10157, 156, 19, 14229, 1189, 6653, 119, 222, 2, 10806, 5421, 1251, 0, 9510, 942],
    [4815, 10206, 155, 27, 14293, 1161, 6662, 117, 191, 1, 10713, 5541, 1267, 0, 9515, 872],
    [4838, 10250, 150, 29, 14270, 1222, 6640, 135, 201, 1, 10826, 5286, 1239, 0, 9539, 910],
    [4850, 10183, 134, 21, 14422, 1143, 6693, 121, 195, 1, 10765, 5322, 1298, 0, 9502, 886],
    [4879, 10129, 150, 26, 14396, 1149, 6580, 148, 182, 1, 10857, 5351, 1253, 0, 9490, 945],
    [4853, 10259, 130, 25, 14411, 1198, 6607, 111, 181, 2, 10725, 5404, 1244, 0, 9513, 873],
    [4878, 10110, 137, 22, 14314, 1225, 6663, 127, 173, 1, 10730, 5381, 1325, 0, 9565, 885],
    [4817, 10230, 165, 24, 14359, 1205, 6721, 115, 177, 0, 10780, 5346, 1273, 0, 9475, 849],
    [4852, 10143, 144, 23, 14369, 1192, 6611, 125, 192, 0, 10821, 5483, 1177, 0, 9473, 931],
    [4942, 10251, 133, 21, 14300, 1149, 6667, 130, 171, 0, 10750, 5332, 1308, 0, 9492, 890],
    [4889, 10268, 140, 37, 14345, 1199, 6640, 104, 154, 2, 10717, 5349, 1231, 0, 9553, 908],
    [4834, 10203, 133, 30, 14396, 1165, 6616, 127, 199, 2, 10755, 5433, 1258, 0, 9506, 879],
    [4827, 10229, 161, 29, 14176, 1161, 6687, 136, 182, 1, 10823, 5363, 1316, 0, 9516, 929],
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
        help="Use the traced routing matrix or an equal 4096-to-each-rank matrix.",
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
        return [[4096] * 16 for _ in range(16)]
    return ROUTING_MATRIX


def all_gather_rank_tensor(local: torch.Tensor, world_size: int) -> torch.Tensor | None:
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
        raise RuntimeError("Launch with torchrun on both 8-GPU nodes")

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
        if world_size != 16:
            raise RuntimeError(f"This benchmark requires exactly 16 ranks, got {world_size}")

        matrix = routing_matrix(args.routing)
        send_splits = matrix[rank]
        recv_splits = [matrix[source][rank] for source in range(world_size)]
        if sum(send_splits) != 65536:
            raise RuntimeError(f"rank {rank} send splits sum to {sum(send_splits)}, not 65536")

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
        all_elapsed = all_gather_rank_tensor(elapsed_tensor, world_size)
        all_issues = all_gather_rank_tensor(issue_tensor, world_size)

        if rank == 0:
            assert all_elapsed is not None
            assert all_issues is not None
            bytes_per_token = args.hidden_size * torch.empty((), dtype=dtype).element_size()
            print(
                f"routing={args.routing} hidden={args.hidden_size} dtype={dtype} "
                f"warmups={args.warmups} repeats={args.repeats}"
            )
            print(
                f"{'rank':>4} {'recv':>8} {'local GB':>10} {'cross GB':>10} "
                f"{'median ms':>10} {'min ms':>10} {'max ms':>10}"
            )
            for current_rank in range(world_size):
                node_start = 0 if current_rank < 8 else 8
                node_end = node_start + 8
                remote_node_start = 8 - node_start
                remote_node_end = remote_node_start + 8

                local_send = sum(matrix[current_rank][node_start:node_end]) - matrix[current_rank][current_rank]
                local_recv = sum(matrix[source][current_rank] for source in range(node_start, node_end)) - matrix[current_rank][current_rank]
                cross_send = sum(matrix[current_rank][remote_node_start:remote_node_end])
                cross_recv = sum(matrix[source][current_rank] for source in range(remote_node_start, remote_node_end))
                local_gb = max(local_send, local_recv) * bytes_per_token / 1e9
                cross_gb = max(cross_send, cross_recv) * bytes_per_token / 1e9
                received = sum(matrix[source][current_rank] for source in range(world_size))
                samples = all_elapsed[current_rank].tolist()
                print(
                    f"{current_rank:>4} {received:>8} {local_gb:>10.3f} {cross_gb:>10.3f} "
                    f"{statistics.median(samples):>10.3f} "
                    f"{min(samples):>10.3f} {max(samples):>10.3f}"
                )

            print()
            print(
                f"{'trial':>5} {'node0 skew':>11} {'node1 skew':>11} "
                f"{'max rank ms':>12} {'min rank ms':>12}"
            )
            for trial in range(args.repeats):
                starts = all_issues[:, trial]
                durations = all_elapsed[:, trial]
                node0_skew_ms = (starts[:8].max() - starts[:8].min()).item() / 1e6
                node1_skew_ms = (starts[8:].max() - starts[8:].min()).item() / 1e6
                print(
                    f"{trial:>5} {node0_skew_ms:>11.3f} {node1_skew_ms:>11.3f} "
                    f"{durations.max().item():>12.3f} {durations.min().item():>12.3f}"
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
