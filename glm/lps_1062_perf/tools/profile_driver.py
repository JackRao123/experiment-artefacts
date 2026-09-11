"""Lightweight timing and profiling driver.

All windows reuse the same --datums datums x --seq-len tokens each
(defaults 1 x 131,072), synthetic random tokens, rng seed 0xB300.

Protocol:
  1. warmup  (untraced): absorbs one-time costs (cudnn/DSA autotune at the
     shape, NCCL channel setup, allocator warmup, optimizer lazy init).
  2. control (unprofiled): clean steady-state timing. Profilers add overhead,
     so the headline tok/s/GPU + MFU come from these windows alone.
  3. optional memory profile (--memory-profile):
     memory_profile/start -> fb+optim -> memory_profile/stop.
  4. optional Kineto runtime profile (--kineto-runtime-profile):
      runtime_profile/start -> fb+optim -> runtime_profile/stop. The
      .pt.trace.json covers exactly one full step.
  5. optional Nsight Systems runtime profile (--nsys-runtime-profile):
     nsys start -> fb+optim -> nsys stop.

Kineto historically added ~5-8% overhead so the runtime-profiled window is for attribution,
not the timing headline.

Usage (on the node where the trainer HTTP is up, port 8001):
    python3 profile_driver.py --label expX-131k-d1 [--seq-len 131072] \
        [--datums 1] [--num-gpus 16] [--control-repeats 1] \
        [--memory-profile] [--kineto-runtime-profile | --nsys-runtime-profile SESSION]

Writes /root/.cache/user_artifacts/lps1062_bench/<label>.json + SUMMARY line.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import httpx
from mfu import hfu, mfu3x

BASE_URL = "http://127.0.0.1:8001"
VOCAB_SIZE = 154_880  # GLM-5.2 vocab
FB_TIMEOUT_S = 3600.0  # warmup step includes cudnn/DSA autotune
OP_TIMEOUT_S = 3600.0
OUT_DIR = Path("/root/.cache/user_artifacts/lps1062_bench")
# The server default of 100K entries retained only ~10 seconds on a 131K
# snapshot. The trade-off is intentionally asymmetric: exhausting this ring
# silently discards the oldest events and can invalidate the memory timeline,
# while 5M entries cost only ~630 MB/rank on hosts with ~2 TB of RAM.
MEMORY_PROFILE_MAX_ENTRIES = 5_000_000


def make_datum(rng: random.Random, seq_len: int) -> dict:
    tokens = [rng.randrange(VOCAB_SIZE) for _ in range(seq_len)]
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
        "loss_fn_inputs": {},
    }


def make_datums(rng: random.Random, args: argparse.Namespace) -> list[dict]:
    return [make_datum(rng, args.seq_len) for _ in range(args.datums)]


def submit_and_wait(client: httpx.Client, path: str, body: dict, timeout: float) -> dict:
    r = client.post(path, json=body, headers={"Idempotency-Key": uuid.uuid4().hex}, timeout=60.0)
    if r.status_code != 202:
        raise RuntimeError(f"{path} submit failed: {r.status_code} {r.text[:2000]}")
    operation_id = r.json()["operation_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rr = client.get(f"/operations/{operation_id}", timeout=60.0)
        if rr.status_code == 408:
            continue
        rr.raise_for_status()
        body = rr.json()
        if body.get("status") == "done":
            return body["result"]
        if body.get("status") == "error":
            raise RuntimeError(f"{path} op {operation_id} errored: {body.get('error', '')[:2000]}")
    raise TimeoutError(f"{path} op {operation_id} did not finish in {timeout}s")


def drive_window(client: httpx.Client, label: str, index: int, datums: list[dict],
                  seq_len: int, num_gpus: int, tag: str) -> dict:
    n_tokens = seq_len * len(datums)
    t0 = time.perf_counter()
    fb = submit_and_wait(client, "/forward_backward", {"data": datums}, FB_TIMEOUT_S)
    fb_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    opt = submit_and_wait(client, "/optim_step", {"adam_params": {"learning_rate": 1e-5}}, OP_TIMEOUT_S)
    opt_s = time.perf_counter() - t0
    metrics = (opt or {}).get("metrics") or {}
    rec = {
        "label": label,
        "window_index": index,
        "phase": tag,
        "datums": len(datums),
        "num_tokens": n_tokens,
        "fb_elapsed_s": fb_s,
        "fb_tps_per_gpu": n_tokens / fb_s / num_gpus,
        "optim_elapsed_s": opt_s,
        "loss": (fb or {}).get("loss"),
        "grad_norm": metrics.get("grad_norm"),
        "server_step_seconds": metrics.get("step_seconds"),
        "peak_reserved_bytes": metrics.get("peak_reserved_bytes"),
        "peak_allocated_bytes": metrics.get("peak_allocated_bytes"),
        "device_total_bytes": metrics.get("device_total_bytes"),
    }
    print(
        f"[{label} {tag}{index}] fb={fb_s:.1f}s ({rec['fb_tps_per_gpu']:.0f} tok/s/GPU) "
        f"optim={opt_s:.1f}s loss={rec['loss']} gn={rec['grad_norm']}",
        flush=True,
    )
    return rec


def run_nsys(args: list[str]) -> None:
    command = ["nsys", *args]
    if shutil.which("srun"):
        result = subprocess.run(
            ["squeue", "-h", "--name=devbox_trainer", "--states=RUNNING", "--format=%A %D"],
            check=True,
            capture_output=True,
            text=True,
        )
        jobs = result.stdout.splitlines()
        if len(jobs) != 1:
            raise RuntimeError(f"expected one running devbox_trainer job, found {len(jobs)}")
        job_id, nodes = jobs[0].split()
        command = [
            "srun",
            f"--jobid={job_id}",
            "--overlap",
            f"--nodes={nodes}",
            f"--ntasks={nodes}",
            "--ntasks-per-node=1",
            *command,
        ]
    subprocess.run(command, check=True)


def main() -> None:
    total_started = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--seq-len", type=int, default=131_072)
    ap.add_argument("--datums", type=int, default=1,
                    help="datums per window; each is --seq-len tokens, so a "
                         "window/step is --datums x --seq-len tokens")
    ap.add_argument("--num-gpus", type=int, default=16)
    ap.add_argument("--lora-rank", type=int, default=32,
                    help="LoRA rank of the run (mfu.py: adapter FLOPs scale with rank)")
    ap.add_argument("--control-repeats", type=int, default=1,
                    help="untraced control windows; >1 for a variance estimate on the headline")
    ap.add_argument("--memory-profile", action="store_true",
                    help="capture one memory-profiled fb+optim step")
    runtime_profiles = ap.add_mutually_exclusive_group()
    runtime_profiles.add_argument("--kineto-runtime-profile", action="store_true",
                                  help="capture one Kineto-profiled fb+optim step")
    runtime_profiles.add_argument("--nsys-runtime-profile", metavar="SESSION",
                                  help="capture one Nsight Systems-profiled fb+optim step")
    args = ap.parse_args()

    rng = random.Random(0xB300)
    datums = make_datums(rng, args)
    tokens_per_step = args.seq_len * args.datums
    out: dict = {
        "label": args.label,
        "seq_len": args.seq_len,
        "num_gpus": args.num_gpus,
        "datums_per_window": args.datums,
        "tokens_per_step": tokens_per_step,
        "memory_profile_enabled": args.memory_profile,
        "kineto_runtime_profile_enabled": args.kineto_runtime_profile,
        "nsys_runtime_profile_session": args.nsys_runtime_profile,
        "started": time.strftime("%F %T"),
    }

    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        status = client.get("/status").json()
        out["initial_status"] = status
        print(f"[status] world_size={status.get('world_size')} dp={status.get('data_parallel_size')}", flush=True)

        windows = [drive_window(client, args.label, 0, datums,
                                args.seq_len, args.num_gpus, "warmup")]

        for i in range(args.control_repeats):
            windows.append(drive_window(client, args.label, i, datums,
                                        args.seq_len, args.num_gpus, "control"))

        if args.memory_profile:
            print(
                f"[profile] memory_profile/start max_entries={MEMORY_PROFILE_MAX_ENTRIES}",
                flush=True,
            )
            out["memory_profile_start"] = submit_and_wait(
                client,
                "/memory_profile/start",
                {"max_entries": MEMORY_PROFILE_MAX_ENTRIES},
                OP_TIMEOUT_S,
            )
            try:
                windows.append(drive_window(
                    client, args.label, 0, datums,
                    args.seq_len, args.num_gpus, "memory_profile",
                ))
            finally:
                print(
                    "[profile] memory_profile/stop (snapshot dump can take minutes)",
                    flush=True,
                )
                out["memory_profile_stop"] = submit_and_wait(
                    client, "/memory_profile/stop", {}, OP_TIMEOUT_S
                )

        if args.kineto_runtime_profile:
            print("[profile] runtime_profile/start", flush=True)
            out["kineto_runtime_profile_start"] = submit_and_wait(
                client, "/runtime_profile/start", {}, OP_TIMEOUT_S
            )
            try:
                windows.append(drive_window(
                    client, args.label, 0, datums,
                    args.seq_len, args.num_gpus, "kineto_runtime_profile",
                ))
            finally:
                print(
                    "[profile] runtime_profile/stop (kineto flush can take minutes)",
                    flush=True,
                )
                out["kineto_runtime_profile_stop"] = submit_and_wait(
                    client, "/runtime_profile/stop", {}, OP_TIMEOUT_S
                )

        if args.nsys_runtime_profile:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            report = OUT_DIR / f"{args.label}-%h"
            print(f"[profile] nsys start session={args.nsys_runtime_profile}", flush=True)
            run_nsys([
                "start",
                f"--session={args.nsys_runtime_profile}",
                "--sample=process-tree",
                "--cpuctxsw=process-tree",
                "--backtrace=lbr",
                "--gpu-metrics-devices=all",
                "--gpu-metrics-frequency=10000",
                "--gpuctxsw=true",
                f"--output={report}",
                "--force-overwrite=true",
                "--stats=false",
            ])
            try:
                windows.append(drive_window(
                    client, args.label, 0, datums,
                    args.seq_len, args.num_gpus, "nsys_runtime_profile",
                ))
            finally:
                print(f"[profile] nsys stop session={args.nsys_runtime_profile}", flush=True)
                run_nsys(["stop", f"--session={args.nsys_runtime_profile}"])
            out["nsys_runtime_profile_output"] = f"{report}.nsys-rep"

        # /status can queue behind the profile flushes; be patient.
        out["final_status"] = client.get("/status", timeout=600.0).json()

    controls = [w for w in windows if w["phase"] == "control"]
    ctrl_fb = [w["fb_elapsed_s"] for w in controls]
    control_fb_s = sum(ctrl_fb) / len(ctrl_fb)
    tps_per_gpu = tokens_per_step / control_fb_s / args.num_gpus
    out["windows"] = windows
    aggregates = {
        "tokens_per_step": tokens_per_step,
        "control_fb_seconds_mean": control_fb_s,
        "control_optim_seconds_mean": (
            sum(w["optim_elapsed_s"] for w in controls) / len(controls)
        ),
        "control_tps_per_gpu": tps_per_gpu,
        "mfu3x": mfu3x(tps_per_gpu, args.seq_len, args.lora_rank),
        "hfu": hfu(tps_per_gpu, args.seq_len, args.lora_rank),
        "peak_reserved_bytes": max(
            int(w["peak_reserved_bytes"])
            for w in windows
            if w.get("peak_reserved_bytes") is not None
        ),
        "peak_allocated_bytes": max(
            int(w["peak_allocated_bytes"])
            for w in windows
            if w.get("peak_allocated_bytes") is not None
        ),
        "device_total_bytes": max(
            int(w["device_total_bytes"])
            for w in windows
            if w.get("device_total_bytes") is not None
        ),
    }
    if args.memory_profile:
        memory_profile = next(w for w in windows if w["phase"] == "memory_profile")
        aggregates["memory_profile_fb_seconds"] = memory_profile["fb_elapsed_s"]
        aggregates["memory_profile_optim_seconds"] = memory_profile["optim_elapsed_s"]
    if args.kineto_runtime_profile:
        runtime_profile = next(w for w in windows if w["phase"] == "kineto_runtime_profile")
        aggregates["kineto_runtime_profile_fb_seconds"] = runtime_profile["fb_elapsed_s"]
        aggregates["kineto_runtime_profile_optim_seconds"] = runtime_profile["optim_elapsed_s"]
    if args.nsys_runtime_profile:
        runtime_profile = next(w for w in windows if w["phase"] == "nsys_runtime_profile")
        aggregates["nsys_runtime_profile_fb_seconds"] = runtime_profile["fb_elapsed_s"]
        aggregates["nsys_runtime_profile_optim_seconds"] = runtime_profile["optim_elapsed_s"]
    out["aggregates"] = aggregates
    out["total_elapsed_seconds"] = time.perf_counter() - total_started

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.label}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[done] -> {path}", flush=True)
    agg = out["aggregates"]
    summary = (
        f"SUMMARY {args.label}: {agg['control_tps_per_gpu']:.0f} tok/s/GPU | "
        f"control fb={agg['control_fb_seconds_mean']:.1f}s "
        f"optim={agg['control_optim_seconds_mean']:.1f}s "
        f"({agg['tokens_per_step']} tok) | "
        f"mfu3x {100 * agg['mfu3x']:.1f}% | hfu {100 * agg['hfu']:.1f}%"
    )
    if args.memory_profile:
        summary += (
            f" | memory-profile fb={agg['memory_profile_fb_seconds']:.1f}s "
            f"optim={agg['memory_profile_optim_seconds']:.1f}s"
        )
    if args.kineto_runtime_profile:
        summary += (
            f" | kineto-runtime-profile fb={agg['kineto_runtime_profile_fb_seconds']:.1f}s "
            f"optim={agg['kineto_runtime_profile_optim_seconds']:.1f}s"
        )
    if args.nsys_runtime_profile:
        summary += (
            f" | nsys-runtime-profile fb={agg['nsys_runtime_profile_fb_seconds']:.1f}s "
            f"optim={agg['nsys_runtime_profile_optim_seconds']:.1f}s"
        )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
