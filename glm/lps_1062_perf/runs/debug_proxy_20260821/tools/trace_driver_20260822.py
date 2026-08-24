#!/usr/bin/env python3
"""Kineto trace capture driver for the 4-layer offload-overlap investigation.

Protocol (trainer already healthy on :8001):
  1. one warmup window (fb + optim)
  2. N untraced main windows (timing reference)
  3. POST /runtime_profile/start
  4. one traced main window (fb + optim)
  5. POST /runtime_profile/stop  -> flushes .pt.trace.json
  6. one post-trace untraced window (profiler-perturbation check)

Writes /root/.cache/user_artifacts/lps1062_traces/<label>.json with all
window timings plus the trace file list reported by the stop op.
Same synthetic-token protocol as bench_driver2c.py (seed 0xB300).
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
VOCAB_SIZE = 154_880
FB_TIMEOUT_S = 3600.0
OUT_DIR = Path("/root/.cache/user_artifacts/lps1062_traces")


def make_datum(rng: random.Random, seq_len: int) -> dict:
    tokens = [rng.randrange(VOCAB_SIZE) for _ in range(seq_len)]
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
        "loss_fn_inputs": {},
    }


def submit_and_wait(client: httpx.Client, path: str, body: dict | None,
                    timeout: float) -> dict:
    r = client.post(path, json=body, headers={"Idempotency-Key": uuid.uuid4().hex},
                    timeout=60.0)
    if r.status_code != 202:
        raise RuntimeError(f"{path} submit failed: {r.status_code} {r.text[:2000]}")
    operation_id = r.json()["operation_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rr = client.get(f"/operations/{operation_id}", timeout=60.0)
        if rr.status_code == 408:
            continue
        rr.raise_for_status()
        ob = rr.json()
        if ob.get("status") == "done":
            return ob.get("result") or {}
        if ob.get("status") == "error":
            raise RuntimeError(f"{path} op errored: {ob.get('error', '')[:2000]}")
    raise TimeoutError(f"{path} op {operation_id} did not finish in {timeout}s")


def window(client: httpx.Client, label: str, tag: str, idx: int,
           datums: list[dict], n_tokens: int) -> dict:
    t0 = time.perf_counter()
    fb = submit_and_wait(client, "/forward_backward", {"data": datums}, FB_TIMEOUT_S)
    fb_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    opt = submit_and_wait(client, "/optim_step",
                          {"adam_params": {"learning_rate": 1e-5}}, FB_TIMEOUT_S)
    opt_s = time.perf_counter() - t0
    metrics = (opt or {}).get("metrics") or {}
    rec = {
        "phase": tag,
        "window_index": idx,
        "num_tokens": n_tokens,
        "fb_elapsed_s": fb_s,
        "fb_tps_per_gpu": n_tokens / fb_s,
        "optim_elapsed_s": opt_s,
        "loss": (fb or {}).get("loss"),
        "grad_norm": metrics.get("grad_norm"),
    }
    print(f"[{label} {tag}{idx}] fb={fb_s:.3f}s ({rec['fb_tps_per_gpu']:.0f} tok/s) "
          f"optim={opt_s:.3f}s loss={rec['loss']} gn={rec['grad_norm']}", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--pre-windows", type=int, default=2,
                    help="untraced main windows before the traced one")
    args = ap.parse_args()

    rng = random.Random(0xB300)
    out: dict = {"label": args.label, "seq_len": args.seq_len,
                 "started": time.strftime("%F %T"), "windows": []}

    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        status = client.get("/status").json()
        out["initial_status"] = status
        print(f"[status] world_size={status.get('world_size')}", flush=True)

        out["windows"].append(window(client, args.label, "warmup", 0,
                                     [make_datum(rng, args.seq_len)], args.seq_len))
        for i in range(args.pre_windows):
            out["windows"].append(window(client, args.label, "main", i,
                                         [make_datum(rng, args.seq_len)],
                                         args.seq_len))

        start = submit_and_wait(client, "/runtime_profile/start", {}, 300.0)
        out["profile_start"] = start
        print(f"[profile] start: {start}", flush=True)

        out["windows"].append(window(client, args.label, "traced", 0,
                                     [make_datum(rng, args.seq_len)], args.seq_len))

        stop = submit_and_wait(client, "/runtime_profile/stop", {}, 600.0)
        out["profile_stop"] = stop
        print(f"[profile] stop: files={stop.get('files')} "
              f"size={stop.get('size_bytes')} local={stop.get('local_path')}",
              flush=True)

        out["windows"].append(window(client, args.label, "posttrace", 0,
                                     [make_datum(rng, args.seq_len)], args.seq_len))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.label}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[done] -> {path}", flush=True)


if __name__ == "__main__":
    main()
