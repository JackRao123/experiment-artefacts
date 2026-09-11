"""Whole-range NCU replay preserves TE multi-stream launch dependencies."""
import concurrent.futures
import itertools
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).parent
metrics = ",".join([
    "gpu__time_duration.sum", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed", "sm__warps_active.avg.pct_of_peak_sustained_active",
    "dram__bytes_read.sum", "dram__bytes_write.sum", "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__ops_path_tensor_src_bf16_dst_fp32.sum",
])


def partition(gpu, layouts):
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CUDA_DEVICE_MAX_CONNECTIONS": "32", "OMP_NUM_THREADS": "1"}
    for config, layer, rank in layouts:
        phases = ("forward",)
        for projection, implementation, phase in itertools.product(("gate_up", "down"), ("te", "grouped"), phases):
            name = f"{config.lower()}.L{layer}.rank{rank}.{projection}.{implementation}.{phase}.gpu{gpu}.range"
            report = root / f"{name}.ncu-rep"
            if not report.exists():
                command = ["ncu", "--replay-mode", "app-range", "--nvtx", "--nvtx-include", "profile/",
                           "--clock-control", "none", "--pipeline-boost-state", "dynamic", "--cache-control", "all",
                           "--import-sass", "no", "--apply-rules", "no", "--metrics", metrics, "-o", str(root / name),
                           sys.executable, str(root / "run_probe.py"), "--mode", "profile", "--config", config,
                           "--rank", str(rank), "--layer", str(layer), "--projection", projection,
                           "--implementation", implementation, "--phase", phase]
                started = time.monotonic()
                print(f"START {name}", flush=True)
                with (root / f"{name}.log").open("w") as log:
                    process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        status = process.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise
                    if status:
                        raise subprocess.CalledProcessError(status, command)
                print(f"DONE {name} elapsed={time.monotonic() - started:.1f}s", flush=True)
            with (root / f"{name}.csv").open("w") as csv:
                subprocess.run(["ncu", "--import", str(report), "--page", "raw", "--csv"], stdout=csv, check=True)


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    jobs = [pool.submit(partition, 0, [("CP8EP1", 1, 0), ("CP8EP8", 1, 0)]),
            pool.submit(partition, 1, [("CP8EP1", 2, 0), ("CP8EP8", 2, 4)])]
    for job in jobs:
        job.result()
