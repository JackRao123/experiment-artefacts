#!/usr/bin/env python3
"""Poll per-GPU memory via pynvml at ~10 Hz, append JSONL lines.

Usage: nvml_poller.py <out_path> <run_id>
Each line: {"ts": unix, "run": run_id, "host": hostname, "used": [bytes per GPU], "total": bytes}
Same data source as the dp_worker /debug/nvml endpoint (pynvml driver
counters), but node-local so it works regardless of where HTTP binds.
"""

import json
import socket
import sys
import time

import pynvml


def main() -> None:
    out_path, run_id = sys.argv[1], sys.argv[2]
    host = socket.gethostname()
    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    total = pynvml.nvmlDeviceGetMemoryInfo(handles[0]).total
    with open(out_path, "a", buffering=1) as f:  # line-buffered
        while True:
            t0 = time.time()
            used = [pynvml.nvmlDeviceGetMemoryInfo(h).used for h in handles]
            f.write(
                json.dumps({"ts": t0, "run": run_id, "host": host, "used": used, "total": total})
                + "\n"
            )
            time.sleep(max(0.0, 0.1 - (time.time() - t0)))


if __name__ == "__main__":
    main()
