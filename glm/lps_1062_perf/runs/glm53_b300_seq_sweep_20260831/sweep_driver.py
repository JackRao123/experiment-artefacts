#!/usr/bin/env python3
"""Increasing-length GLM-5.3 TPS and peak-memory sweep."""

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
SEQ_LENS = (131_072, 196_608, 262_144, 327_680, 393_216)
NUM_GPUS = 8
TIMEOUT_S = 3_600.0


def make_datum(rng: random.Random, seq_len: int) -> dict:
    return {
        "model_input": {
            "chunks": [
                {
                    "type": "encoded_text",
                    "tokens": [rng.randrange(VOCAB_SIZE) for _ in range(seq_len)],
                }
            ]
        },
        "loss_fn_inputs": {},
    }


def submit_and_wait(client: httpx.Client, path: str, body: dict) -> dict:
    response = client.post(
        path,
        json=body,
        headers={"Idempotency-Key": uuid.uuid4().hex},
        timeout=60.0,
    )
    if response.status_code != 202:
        raise RuntimeError(f"{path} submit failed: {response.status_code} {response.text[:2000]}")
    operation_id = response.json()["operation_id"]
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        result = client.get(f"/operations/{operation_id}", timeout=60.0)
        if result.status_code == 408:
            continue
        result.raise_for_status()
        payload = result.json()
        if payload.get("status") == "done":
            return payload["result"]
        if payload.get("status") == "error":
            raise RuntimeError(f"{path} failed: {payload.get('error', '')[:2000]}")
    raise TimeoutError(f"{path} did not finish in {TIMEOUT_S}s")


def run_step(
    client: httpx.Client,
    rng: random.Random,
    seq_len: int,
    phase: str,
    index: int,
) -> dict:
    datum = make_datum(rng, seq_len)
    started = time.perf_counter()
    forward = submit_and_wait(client, "/forward_backward", {"data": [datum]})
    fb_seconds = time.perf_counter() - started
    started = time.perf_counter()
    optim = submit_and_wait(
        client,
        "/optim_step",
        {"adam_params": {"learning_rate": 1e-5}},
    )
    optim_seconds = time.perf_counter() - started
    metrics = (optim or {}).get("metrics") or {}
    record = {
        "seq_len": seq_len,
        "phase": phase,
        "index": index,
        "fb_seconds": fb_seconds,
        "tps_per_gpu": seq_len / fb_seconds / NUM_GPUS,
        "optim_seconds": optim_seconds,
        "loss": (forward or {}).get("loss"),
        "grad_norm": metrics.get("grad_norm"),
        "server_step_seconds": metrics.get("step_seconds"),
        "peak_allocated_bytes": metrics.get("peak_allocated_bytes"),
        "peak_reserved_bytes": metrics.get("peak_reserved_bytes"),
        "device_total_bytes": metrics.get("device_total_bytes"),
    }
    print(
        f"[{seq_len} {phase}{index}] fb={fb_seconds:.3f}s "
        f"tps/gpu={record['tps_per_gpu']:.1f} "
        f"peak_alloc={record['peak_allocated_bytes'] / 2**30:.2f}GiB "
        f"peak_reserved={record['peak_reserved_bytes'] / 2**30:.2f}GiB",
        flush=True,
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-repeats", type=int, default=3)
    args = parser.parse_args()

    rng = random.Random(0xB300)
    output = {
        "sequence_lengths": list(SEQ_LENS),
        "num_gpus": NUM_GPUS,
        "control_repeats": args.control_repeats,
        "started": time.strftime("%F %T"),
        "windows": [],
        "results": [],
    }

    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        output["initial_status"] = client.get("/status").json()
        for seq_len in SEQ_LENS:
            output["windows"].append(run_step(client, rng, seq_len, "warmup", 0))
            controls = []
            for index in range(args.control_repeats):
                record = run_step(client, rng, seq_len, "control", index)
                controls.append(record)
                output["windows"].append(record)
            stable = controls[-2:]
            stable_seconds = sum(row["fb_seconds"] for row in stable) / len(stable)
            result = {
                "seq_len": seq_len,
                "control_tps_per_gpu": seq_len
                / (sum(row["fb_seconds"] for row in controls) / len(controls))
                / NUM_GPUS,
                "stable_last_two_tps_per_gpu": seq_len / stable_seconds / NUM_GPUS,
                "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in controls),
                "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in controls),
                "device_total_bytes": max(row["device_total_bytes"] for row in controls),
            }
            output["results"].append(result)
            print(f"RESULT {json.dumps(result, sort_keys=True)}", flush=True)
        output["final_status"] = client.get("/status", timeout=600.0).json()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"[done] {args.output}", flush=True)


if __name__ == "__main__":
    main()
