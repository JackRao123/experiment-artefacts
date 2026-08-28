import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-mb", type=float, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


def transfer(buffer: torch.Tensor, rank: int) -> None:
    if rank == 0:
        work = dist.isend(buffer, dst=1)
    else:
        work = dist.irecv(buffer, src=0)
    work.wait()
    torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    if args.payload_mb <= 0 or args.warmup < 0 or args.iterations < 1:
        raise ValueError("payload-mb and iterations must be positive; warmup cannot be negative")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    if dist.get_world_size() != 2:
        raise RuntimeError("This benchmark requires exactly two NCCL ranks")

    payload_bytes = round(args.payload_mb * 1024 * 1024)
    buffer = torch.empty(payload_bytes, dtype=torch.uint8, device="cuda")
    buffer.fill_(rank)

    for _ in range(args.warmup):
        dist.barrier()
        transfer(buffer, rank)

    samples_ms: list[float] = []
    for _ in range(args.iterations):
        dist.barrier()
        torch.cuda.synchronize()
        started = time.perf_counter()
        transfer(buffer, rank)
        elapsed_ms = (time.perf_counter() - started) * 1_000

        elapsed = torch.tensor(elapsed_ms, dtype=torch.float64, device="cuda")
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        if rank == 0:
            samples_ms.append(elapsed.item())

    if rank == 0:
        ordered = sorted(samples_ms)
        median_ms = statistics.median(samples_ms)
        p95_ms = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
        result = {
            "payload_bytes": payload_bytes,
            "payload_GB": payload_bytes / 1e9,
            "iterations": args.iterations,
            "min_ms": min(samples_ms),
            "median_ms": median_ms,
            "p95_ms": p95_ms,
            "max_ms": max(samples_ms),
            "median_GBps": payload_bytes / 1e9 / (median_ms / 1_000),
            "min_latency_GBps": payload_bytes / 1e9 / (min(samples_ms) / 1_000),
        }
        print(f"RESULT_JSON={json.dumps(result, sort_keys=True)}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
