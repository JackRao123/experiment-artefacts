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
import time
from pathlib import Path

import httpx
import profile_driver as driver

parser = argparse.ArgumentParser()
parser.add_argument("case")
parser.add_argument("--warmups", type=int, default=3)
parser.add_argument("--controls", type=int, default=5)
parser.add_argument("--trace-steps", type=int, default=1)
parser.add_argument("--metrics", action="store_true")
parser.add_argument("--continue-case", help="Validate an already-loaded completed case without reinitializing model state")
parser.add_argument("--validation-controls", type=int, default=0, help="Additional unprofiled steps after captures; excluded from the five-control headline")
parser.add_argument("--fb-only", action="store_true", help="No optimizer requests; gradients accumulate across warmup and the measured FB")
args = parser.parse_args()
if args.fb_only and (args.warmups, args.controls, args.trace_steps, args.metrics, args.validation_controls, args.continue_case) != (1, 0, 1, False, 0, None):
    parser.error("--fb-only requires one warmup, zero controls, one timing trace, no metrics/validation/continuation")
root = Path(__file__).resolve().parent
folder = root / args.case
if (folder / "benchmark.json").exists():
    raise FileExistsError(f"Preserve the existing result; create a new case instead: {folder}")
driver.OUT_DIR = folder
config = json.loads((folder / "trainer-config.json").read_text())
seq = config["max_seq_len"]
datums = [driver.make_datum(random.Random(0xB300), seq)]
session = f"glm53-{args.continue_case or args.case}-0911"
windows = []
result = {"case": args.case, "config": config, "sequence_length": seq, "num_gpus": 8,
          "fb_only": args.fb_only,
          "continued_from_case": args.continue_case,
          "requested_windows": {"warmups": args.warmups, "controls": args.controls, "trace_steps": args.trace_steps, "metrics": args.metrics, "validation_controls": args.validation_controls},
          "capture_complete": False, "artifacts": {},
          "physical_gpu_by_rank": {str(i): i for i in range(8)},
          "tokens_per_step": seq, "step_definition": "forward_backward HTTP request; optimizer separate",
          "input_sha256": hashlib.sha256(json.dumps(datums, sort_keys=True).encode()).hexdigest(),
          "windows": windows, "nsys_version": subprocess.check_output(["nsys", "--version"], text=True).strip()}
source = Path("/root/glm53-pr1355-repro-20260910/trainers")
if args.fb_only:
    result["step_definition"] = "single profiled forward_backward HTTP request; no optimizer; one untimed FB warmup"
    result["optimizer_steps"] = 0
result["source_revisions"] = {label: subprocess.check_output(["git", "-C", str(source / relative), "rev-parse", "HEAD"], text=True).strip()
                              for label, relative in (("trainers", "."), ("bridge", "server-megatron-bridge/vendor/megatron-bridge"),
                              ("core", "server-megatron-bridge/vendor/megatron-bridge/3rdparty/Megatron-LM"))}
result["runtime_options"] = json.loads((folder / "run_options.json").read_text())
result["gpu_inventory"] = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv"], text=True)
result["resource_limits"] = {name: (Path("/sys/fs/cgroup") / name).read_text().strip()
                             for name in ("cpu.max", "memory.max") if (Path("/sys/fs/cgroup") / name).exists()}


def save():
    controls = [w["fb_elapsed_s"] for w in windows if w["phase"] == "control"]
    if controls:
        result["control"] = {"n": len(controls), "mean_s": statistics.mean(controls),
                             "median_s": statistics.median(controls), "min_s": min(controls), "max_s": max(controls),
                             "sd_s": statistics.stdev(controls) if len(controls)>1 else None,
                             "tps_per_gpu": seq / 8 / statistics.mean(controls)}
    (folder / "benchmark.json").write_text(json.dumps(result, indent=2))


def nsys_command(command, label, *, timeout):
    # Preserve importer diagnostics without streaming thousands of progress
    # carriage returns into the interactive session.
    log = folder / f"nsys-{label}.log"
    print(f"nsys {label}; diagnostics: {log}", flush=True)
    with log.open("w") as stream:
        subprocess.run(command, check=True, timeout=timeout, stdout=stream, stderr=subprocess.STDOUT)


with httpx.Client(base_url=driver.BASE_URL, timeout=60) as client:
    result["initial_status"] = client.get("/status").json()
    assert result["initial_status"]["mode"] == "training"
    assert result["initial_status"]["world_size"] == result["num_gpus"]
    for status_key, config_key in (
        ("model_id", "base_model"), ("max_seq_len", "max_seq_len"),
        ("tensor_parallel_size", "tensor_parallel_size"),
        ("pipeline_parallel_size", "pipeline_parallel_size"),
        ("expert_parallel_size", "expert_parallel_size"),
        ("context_parallel_size", "context_parallel_size"),
        ("expert_tensor_parallel_size", "expert_tensor_parallel_size"),
    ):
        assert result["initial_status"][status_key] == config[config_key], f"Wrong running trainer: {status_key}"
    if args.continue_case:
        previous = json.loads((root / args.continue_case / "benchmark.json").read_text())
        assert previous["capture_complete"], "Only continue a finalized case"
        for key in ("config", "runtime_options", "source_revisions", "input_sha256"):
            assert previous[key] == result[key], f"Continuation changes {key}"
        for key in ("step", "model_id", "world_size", "expert_parallel_size", "context_parallel_size", "max_seq_len"):
            assert previous["final_status"][key] == result["initial_status"][key], f"Running model changed: {key}"

    def window(phase, index):
        if args.fb_only:
            start = time.perf_counter()
            response = driver.submit_and_wait(client, "/forward_backward", {"data":datums}, driver.FB_TIMEOUT_S)
            elapsed = time.perf_counter()-start
            value = {"phase":phase, "window_index":index, "fb_elapsed_s":elapsed,
                     "fb_tps_per_gpu":seq/8/elapsed, "loss":response["loss"]}
            assert math.isfinite(value["loss"]), value
            if phase != "warmup":
                result["single_fb_measurement"] = value
            print(f"{phase}: {elapsed:.4f}s, {seq/8/elapsed:.1f} TPS/GPU, no optimizer", flush=True)
        else:
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
        nsys_command(command, f"{label}-start", timeout=180)
        # A profiled request taking >5x the steady control is a failed capture,
        # not an observation to average into model performance.
        driver.FB_TIMEOUT_S = 120.0 if args.fb_only else max(60.0, 5 * result["control"]["median_s"])
        driver.OP_TIMEOUT_S = driver.FB_TIMEOUT_S
        try:
            for index in range(count):
                window(label, index)
        except BaseException as error:
            result.setdefault("capture_failures", []).append({"phase": label, "error": repr(error)})
            save()
            raise
        finally:
            nsys_command(["nsys", "stop", f"--session={session}"], f"{label}-stop", timeout=900)
        nsys_command(["nsys", "export", "--type=sqlite", f"--output={output}.sqlite", f"{output}.nsys-rep"], f"{label}-export", timeout=900)
        for path in (output.with_suffix(".nsys-rep"), output.with_suffix(".sqlite")):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            result["artifacts"][path.name] = {"sha256": digest, "bytes": path.stat().st_size}
        save()
    # Keep robustness validation separate from the initial matched timing window.
    for index in range(args.validation_controls):
        window("validation", index)
    result["final_status"] = client.get("/status").json()
    if args.fb_only:
        assert result["final_status"]["step"] == result["initial_status"]["step"]
    result["capture_complete"] = True
    save()
