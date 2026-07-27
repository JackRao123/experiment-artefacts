#!/usr/bin/env python3
"""Sample per-GPU NVML memory at sub-second resolution.

Run one process per node. Each process writes a host-specific CSV so a Slurm
step can profile every trainer node without coordinating writers.
"""

from __future__ import annotations

import argparse
import csv
import signal
import socket
import time
from pathlib import Path

import pynvml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--interval-ms", type=float, default=20.0)
    args = parser.parse_args()
    if args.interval_ms <= 0:
        parser.error("--interval-ms must be positive")

    stopping = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    pynvml.nvmlInit()
    try:
        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(index)
            for index in range(pynvml.nvmlDeviceGetCount())
        ]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output = args.output_dir / f"{socket.gethostname()}.csv"
        interval_ns = int(args.interval_ms * 1_000_000)
        deadline_ns = time.monotonic_ns()

        with output.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "time_ns",
                    "monotonic_ns",
                    "gpu",
                    "used_bytes",
                    "free_bytes",
                    "total_bytes",
                    "gpu_util_percent",
                ]
            )
            while not stopping:
                wall_ns = time.time_ns()
                monotonic_ns = time.monotonic_ns()
                for index, handle in enumerate(handles):
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    writer.writerow(
                        [
                            wall_ns,
                            monotonic_ns,
                            index,
                            memory.used,
                            memory.free,
                            memory.total,
                            utilization.gpu,
                        ]
                    )
                file.flush()
                deadline_ns += interval_ns
                delay_ns = deadline_ns - time.monotonic_ns()
                if delay_ns > 0:
                    time.sleep(delay_ns / 1_000_000_000)
                else:
                    deadline_ns = time.monotonic_ns()
    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
