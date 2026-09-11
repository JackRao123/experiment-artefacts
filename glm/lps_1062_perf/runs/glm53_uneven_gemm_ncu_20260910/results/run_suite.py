"""Run independent shape replays on the pod's eight GPUs; no distributed trainer."""
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).parent
jobs = []
for rank in range(8):
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(rank), "CUDA_DEVICE_MAX_CONNECTIONS": "32", "OMP_NUM_THREADS": "1"}
    log = (root / f"timing.rank{rank}.log").open("w")
    command = [sys.executable, str(root / "run_probe.py"), "--rank", str(rank),
               "--output", str(root / f"timing.rank{rank}.json")]
    process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
    jobs.append((rank, process, log))
    print(f"Started GPU {rank}, PID {process.pid}", flush=True)
failed = []
for rank, process, log in jobs:
    code = process.wait()
    log.close()
    print(f"GPU {rank} completed, exit={code}", flush=True)
    if code:
        failed.append(rank)
if failed:
    raise RuntimeError(f"Failed replay ranks: {failed}")
