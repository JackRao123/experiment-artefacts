#!/usr/bin/env python3
"""Lightweight trace+timing driver: warmup -> kineto-traced step -> untraced control.

All windows are the same shape: --datums datums x --seq-len tokens each
(defaults 1 x 131,072), synthetic random tokens, rng seed 0xB300.

Protocol:
  1. warmup  (untraced): absorbs one-time costs (cudnn/DSA autotune at the
     shape, NCCL channel setup, allocator warmup, optimizer lazy init).
  2. traced  (kineto on): runtime_profile/start -> fb+optim -> runtime_profile/stop.
     The .pt.trace.json covers exactly one full step. This window is for
     attribution, NOT for the timing headline.
  3. control (untraced): clean steady-state timing. Kineto adds ~5-8% overhead
     (Aug-6 262k baseline: 76.2s traced vs 70.8s untraced = 7.6%), so the
     headline tok/s/GPU + MFU come from THIS window alone.

memory_profile spans all windows (start before warmup, stop after the last
control) so the allocator HWM reflects steady state.

NOTE: with the default --datums 1, 1 datum x 131,072 = 131,072 tokens/step —
a different operating point from the ~524,288-token standard of the existing
LPS-1062 rows. Pass --datums 4 at 131k for the 524,288-token/step standard
operating point (comparable to the anchor table).

Usage (on the node where the trainer HTTP is up, port 8001):
    python3 profile_driver.py --label expX-131k-d1 [--seq-len 131072] \
        [--datums 1] [--num-gpus 16] [--control-repeats 1]

Writes /root/.cache/user_artifacts/lps1062_bench/<label>.json + SUMMARY line.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from pathlib import Path

import httpx

from mfu import PEAK_FLOPS_GPU, hfu, mfu3x

BASE_URL = "http://127.0.0.1:8001"
VOCAB_SIZE = 154_880  # GLM-5.2 vocab
FB_TIMEOUT_S = 3600.0  # warmup step includes cudnn/DSA autotune
OP_TIMEOUT_S = 3600.0
OUT_DIR = Path("/root/.cache/user_artifacts/lps1062_bench")


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
    }
    print(
        f"[{label} {tag}{index}] fb={fb_s:.1f}s ({rec['fb_tps_per_gpu']:.0f} tok/s/GPU) "
        f"optim={opt_s:.1f}s loss={rec['loss']} gn={rec['grad_norm']}",
        flush=True,
    )
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--seq-len", type=int, default=131_072)
    ap.add_argument("--datums", type=int, default=1,
                    help="datums per window; each is --seq-len tokens, so a "
                         "window/step is --datums x --seq-len tokens")
    ap.add_argument("--num-gpus", type=int, default=16)
    ap.add_argument("--lora-rank", type=int, default=32,
                    help="LoRA rank of the run (mfu.py: adapter FLOPs scale with rank)")
    ap.add_argument("--max-entries", type=int, default=1_000_000,
                    help="allocator event-ring size for memory_profile/start. The "
                         "server default (100,000) covered only ~10 s on a 131k "
                         "snapshot — under half a d2 step — so the replay peak came "
                         "out far below dump-time live and the composition was "
                         "window-local. 1M events is ~4 steps of headroom at ~126 MB "
                         "per rank.")
    ap.add_argument("--control-repeats", type=int, default=1,
                    help="untraced control windows; >1 for a variance estimate on the headline")
    args = ap.parse_args()

    rng = random.Random(0xB300)
    tokens_per_step = args.seq_len * args.datums
    out: dict = {
        "label": args.label,
        "seq_len": args.seq_len,
        "num_gpus": args.num_gpus,
        "datums_per_window": args.datums,
        "tokens_per_step": tokens_per_step,
        "started": time.strftime("%F %T"),
    }

    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        status = client.get("/status").json()
        out["initial_status"] = status
        print(f"[status] world_size={status.get('world_size')} dp={status.get('data_parallel_size')}", flush=True)

        print(f"[profile] memory_profile/start max_entries={args.max_entries}", flush=True)
        out["memory_profile_start"] = submit_and_wait(
            client, "/memory_profile/start", {"max_entries": args.max_entries}, OP_TIMEOUT_S
        )

        windows = [drive_window(client, args.label, 0, make_datums(rng, args),
                                args.seq_len, args.num_gpus, "warmup")]

        print("[profile] runtime_profile/start", flush=True)
        out["runtime_profile_start"] = submit_and_wait(client, "/runtime_profile/start", {}, OP_TIMEOUT_S)

        windows.append(drive_window(client, args.label, 0, make_datums(rng, args),
                                    args.seq_len, args.num_gpus, "traced"))

        print("[profile] runtime_profile/stop (sync kineto flush, can take minutes)", flush=True)
        out["runtime_profile_stop"] = submit_and_wait(client, "/runtime_profile/stop", {}, OP_TIMEOUT_S)

        for i in range(args.control_repeats):
            windows.append(drive_window(client, args.label, i, make_datums(rng, args),
                                        args.seq_len, args.num_gpus, "control"))

        print("[profile] memory_profile/stop (snapshot dump, can take minutes)", flush=True)
        out["memory_profile_stop"] = submit_and_wait(client, "/memory_profile/stop", {}, OP_TIMEOUT_S)

        # /status can queue behind the profile flushes; be patient.
        out["final_status"] = client.get("/status", timeout=600.0).json()

    controls = [w for w in windows if w["phase"] == "control"]
    traced = next(w for w in windows if w["phase"] == "traced")
    ctrl_fb = [w["fb_elapsed_s"] for w in controls]
    control_fb_s = sum(ctrl_fb) / len(ctrl_fb)
    tps_per_gpu = tokens_per_step / control_fb_s / args.num_gpus
    overhead_pct = (traced["fb_elapsed_s"] - control_fb_s) / control_fb_s * 100
    out["windows"] = windows
    out["aggregates"] = {
        "tokens_per_step": tokens_per_step,
        "control_fb_seconds_mean": control_fb_s,
        "control_tps_per_gpu": tps_per_gpu,
        "mfu3x": mfu3x(tps_per_gpu, args.seq_len, args.lora_rank),
        "hfu": hfu(tps_per_gpu, args.seq_len, args.lora_rank),
        "traced_fb_seconds": traced["fb_elapsed_s"],
        "kineto_overhead_pct": overhead_pct,
        "optim_s_mean": sum(w["optim_elapsed_s"] for w in controls) / len(controls),
        "peak_gpu_memory_bytes": max((out["final_status"].get("gpu_memory") or {}).values(), default=0),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.label}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[done] -> {path}", flush=True)
    agg = out["aggregates"]
    print(
        f"SUMMARY {args.label}: {agg['control_tps_per_gpu']:.0f} tok/s/GPU | "
        f"step {agg['control_fb_seconds_mean']:.1f}s ({agg['tokens_per_step']} tok) | "
        f"mfu3x {100 * agg['mfu3x']:.1f}% | hfu {100 * agg['hfu']:.1f}% | "
        f"traced {agg['traced_fb_seconds']:.1f}s (+{agg['kineto_overhead_pct']:.1f}% kineto overhead) | "
        f"peak_mem {agg['peak_gpu_memory_bytes'] / 2**30:.0f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
