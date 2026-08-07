#!/usr/bin/env python3
"""Lean apples-to-apples benchmark driver for LPS-1062 optimization iterations.

Same synthetic data + window structure as the baseline profile run
(profile_driver.py): rng seed 0xB300, 1 warmup window (1 datum), then
MAIN_REPEATS main windows (2 datums = 524,288 tokens each). No profilers.

Usage (on the devbox leader, trainer HTTP up on :8001):
    python3 bench_driver.py --label exp01-flex [--repeats 2]

Writes /root/.cache/user_artifacts/lps1062_bench/<label>.json and prints a
one-line summary for the notebook.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8001"
SEQ_LEN = 262_144
VOCAB_SIZE = 154_880
NUM_GPUS = 16
WARMUP_DATUMS = 1
MAIN_DATUMS = 2
FB_TIMEOUT_S = 3600.0
OUT_DIR = Path("/root/.cache/user_artifacts/lps1062_bench")

# MFU constants (mfu_calc.py): fwd FLOPs/token, B300 dense bf16 peak/GPU
FWD_FLOPS_PER_TOK = 118.3e9
PEAK_FLOPS_GPU = 2.5e15

# Baseline canaries (q480z53, 0e0b65a6): loss/grad_norm per window
CANARY = [(12.3556, 0.9396), (12.3392, 0.9408), (12.3105, 0.6931)]


def make_datum(rng: random.Random) -> dict:
    tokens = [rng.randrange(VOCAB_SIZE) for _ in range(SEQ_LEN)]
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
        "loss_fn_inputs": {},
    }


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


def drive_window(client: httpx.Client, label: str, index: int, datums: list[dict]) -> dict:
    n_tokens = SEQ_LEN * len(datums)
    t0 = time.perf_counter()
    fb = submit_and_wait(client, "/forward_backward", {"data": datums}, FB_TIMEOUT_S)
    fb_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    opt = submit_and_wait(client, "/optim_step", {"adam_params": {"learning_rate": 1e-5}}, FB_TIMEOUT_S)
    opt_s = time.perf_counter() - t0
    metrics = (opt or {}).get("metrics") or {}
    rec = {
        "label": label,
        "window_index": index,
        "num_tokens": n_tokens,
        "fb_elapsed_s": fb_s,
        "fb_tps_per_gpu": n_tokens / fb_s / NUM_GPUS,
        "optim_elapsed_s": opt_s,
        "loss": (fb or {}).get("loss"),
        "grad_norm": metrics.get("grad_norm"),
    }
    print(
        f"[{label} {index}] fb={fb_s:.1f}s ({rec['fb_tps_per_gpu']:.0f} tok/s/GPU) "
        f"optim={opt_s:.1f}s loss={rec['loss']} gn={rec['grad_norm']}",
        flush=True,
    )
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--repeats", type=int, default=2, help="main windows (2 datums each)")
    ap.add_argument("--hw-passes", type=float, default=None,
                    help="fwd-equivalent passes actually run per token for HFU "
                    "(4=full recompute, 3=no recompute); omit to skip HFU")
    args = ap.parse_args()

    rng = random.Random(0xB300)
    out: dict = {"label": args.label, "seq_len": SEQ_LEN, "started": time.strftime("%F %T")}

    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        status = client.get("/status").json()
        out["initial_status"] = status
        print(f"[status] world_size={status.get('world_size')}", flush=True)

        windows = [drive_window(client, "warmup", 0, [make_datum(rng) for _ in range(WARMUP_DATUMS)])]
        for i in range(args.repeats):
            windows.append(drive_window(client, "main", i, [make_datum(rng) for _ in range(MAIN_DATUMS)]))

        final_status = client.get("/status", timeout=600.0).json()
        out["final_status"] = final_status

    mains = windows[1:]
    fb = [w["fb_elapsed_s"] for w in mains]
    toks = sum(w["num_tokens"] for w in mains)
    tps = toks / sum(fb)
    mfu3x = tps * 3 * FWD_FLOPS_PER_TOK / (NUM_GPUS * PEAK_FLOPS_GPU)
    agg = {
        "step_s_mean": sum(fb) / len(fb),
        "tps": tps,
        "tps_per_gpu": tps / NUM_GPUS,
        "mfu3x": mfu3x,
        "hfu": (tps * args.hw_passes * FWD_FLOPS_PER_TOK / (NUM_GPUS * PEAK_FLOPS_GPU))
        if args.hw_passes else None,
        "optim_s_mean": sum(w["optim_elapsed_s"] for w in mains) / len(mains),
        "peak_gpu_memory_bytes": max((final_status.get("gpu_memory") or {}).values(), default=0),
    }
    out["windows"] = windows
    out["aggregates"] = agg

    # correctness canary vs baseline
    drift = []
    for w, (bl, bg) in zip(windows, CANARY):
        if w["loss"] is not None:
            drift.append(f"{w['label']}{w['window_index']}: dloss={w['loss']-bl:+.4f} dgn={(w['grad_norm'] or 0)-bg:+.4f}")
    out["canary_drift_vs_baseline"] = drift

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.label}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[done] -> {path}", flush=True)
    print(
        f"SUMMARY {args.label}: {agg['tps_per_gpu']:.0f} tok/s/GPU | "
        f"step {agg['step_s_mean']:.1f}s | mfu3x {100 * mfu3x:.1f}% | "
        f"peak_mem {agg['peak_gpu_memory_bytes'] / 2**30:.0f} GiB",
        flush=True,
    )
    for d in drift:
        print("CANARY", d, flush=True)


if __name__ == "__main__":
    main()
