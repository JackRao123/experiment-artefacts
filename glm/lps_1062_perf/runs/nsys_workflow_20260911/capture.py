"""Warmup, controls and separate all-rank nsys timing/metric captures.

Run inside the profiler pod after its devbox-up waiter reports healthy.
No source patches or model changes; requests use the original driver helpers.
"""
import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
from pathlib import Path

import httpx
import profile_driver as driver

parser = argparse.ArgumentParser()
parser.add_argument("case")
parser.add_argument("--warmups", type=int, default=3)
parser.add_argument("--controls", type=int, default=5)
parser.add_argument("--trace-steps", type=int, default=3)
parser.add_argument("--metrics", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parent
folder = root / args.case
if (folder / "benchmark.json").exists():
    raise FileExistsError(f"Preserve the existing result; create a new case instead: {folder}")
driver.OUT_DIR = folder
config = json.loads((folder / "trainer-config.json").read_text())
seq = config["max_seq_len"]
datums = [driver.make_datum(random.Random(0xB300), seq)]
session = f"glm53-{args.case}-0911"
windows = []
result = {"case": args.case, "config": config, "sequence_length": seq, "num_gpus": 8,
          "physical_gpu_by_rank": {str(i): i for i in range(8)},
          "tokens_per_step": seq, "step_definition": "forward_backward HTTP request; optimizer separate",
          "input_sha256": hashlib.sha256(json.dumps(datums, sort_keys=True).encode()).hexdigest(),
          "windows": windows, "nsys_version": subprocess.check_output(["nsys", "--version"], text=True).strip()}
source = Path("/root/glm53-pr1355-repro-20260910/trainers")
result["source_revisions"] = {label: subprocess.check_output(["git", "-C", str(source / relative), "rev-parse", "HEAD"], text=True).strip()
                              for label, relative in (("trainers", "."), ("bridge", "server-megatron-bridge/vendor/megatron-bridge"),
                              ("core", "server-megatron-bridge/vendor/megatron-bridge/3rdparty/Megatron-LM"))}
result["runtime_options"] = {"fsdp": True, "prefetch": True, "persistent_buffers": True,
                             "grouped_mm": args.case.endswith("grouped"), "cuda_graphs": False,
                             "lm_head_chunk": 4096, "memory_efficient_lm_head": False,
                             "capture": "software CUDA/NVTX; event tracing enabled; no CPU/callstack sampling"}
result["gpu_inventory"] = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv"], text=True)


def save():
    controls = [w["fb_elapsed_s"] for w in windows if w["phase"] == "control"]
    if controls:
        result["control"] = {"n": len(controls), "mean_s": statistics.mean(controls),
                             "median_s": statistics.median(controls), "min_s": min(controls), "max_s": max(controls),
                             "tps_per_gpu": seq / 8 / statistics.mean(controls)}
    (folder / "benchmark.json").write_text(json.dumps(result, indent=2))


with httpx.Client(base_url=driver.BASE_URL, timeout=60) as client:
    result["initial_status"] = client.get("/status").json()

    def window(phase, index):
        value = driver.drive_window(client, args.case, index, datums, seq, 8, phase)
        assert math.isfinite(value["loss"]) and math.isfinite(value["grad_norm"]), value
        windows.append(value)
        save()

    for phase, count in (("warmup", args.warmups), ("control", args.controls)):
        for index in range(count):
            window(phase, index)
    for label, count, metrics in (("timing", args.trace_steps, False), ("metrics", 1 if args.metrics else 0, True)):
        if not count:
            continue
        output = folder / label
        if output.with_suffix(".nsys-rep").exists():
            raise FileExistsError(output)
        command = ["nsys", "start", f"--session={session}", "--sample=none", "--cpuctxsw=none",
                   "--backtrace=none", "--stats=false", f"--output={output}"]
        if metrics:
            command += ["--gpu-metrics-devices=all", "--gpu-metrics-frequency=10000"]
        subprocess.run(command, check=True)
        try:
            for index in range(count):
                window(label, index)
        finally:
            subprocess.run(["nsys", "stop", f"--session={session}"], check=True)
        subprocess.run(["nsys", "export", "--type=sqlite", f"--output={output}.sqlite", f"{output}.nsys-rep"], check=True)
    result["final_status"] = client.get("/status").json()
    save()
