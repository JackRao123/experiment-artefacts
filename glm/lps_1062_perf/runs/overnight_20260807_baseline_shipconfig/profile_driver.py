#!/usr/bin/env python3
"""Synthetic-data benchmark + profiling driver for the GLM-5.2 B300 devbox trainer.

Mirrors tests/benchmarking's window structure (warmup fb+optim, then main
fb+optim windows) but with random-token synthetic data, and toggles the
trainer's HTTP profilers around the measured windows:

  - memory profile: on before warmup, off at the end (all ranks write
    memory.rank<N>.pickle under the profile output dir)
  - runtime (kineto) profile: on before main window 1, off right after its
    optim_step -> the .pt.trace.json covers exactly one full training step

Run on the leader (rank 0 serves HTTP on 127.0.0.1:8001).
"""

from __future__ import annotations

import json
import random
import time
import uuid
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8001"
SEQ_LEN = 262_144
VOCAB_SIZE = 154_880  # GLM-5.2 vocab
NUM_GPUS = 16
WARMUP_DATUMS = 1
MAIN_DATUMS = 2
MAIN_REPEATS = 2
FB_TIMEOUT_S = 3600.0  # warmup step includes cudnn/DSA autotune at 256k
OP_TIMEOUT_S = 3600.0
RESULTS_PATH = Path("/root/.cache/user_artifacts/glm52-b300-profile-results.json")


def make_datum(rng: random.Random) -> dict:
    tokens = [rng.randrange(VOCAB_SIZE) for _ in range(SEQ_LEN)]
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
        "loss_fn_inputs": {},
    }


def submit_and_wait(client: httpx.Client, path: str, body: dict, timeout: float) -> dict:
    """POST a long-running op and long-poll /operations/{id} until done."""
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
    opt = submit_and_wait(
        client, "/optim_step", {"adam_params": {"learning_rate": 1e-5}}, OP_TIMEOUT_S
    )
    opt_s = time.perf_counter() - t0

    metrics = (opt or {}).get("metrics") or {}
    rec = {
        "label": label,
        "window_index": index,
        "datums": len(datums),
        "num_tokens": n_tokens,
        "fb_elapsed_s": fb_s,
        "fb_tps": n_tokens / fb_s,
        "fb_tps_per_gpu": n_tokens / fb_s / NUM_GPUS,
        "optim_elapsed_s": opt_s,
        "loss": (fb or {}).get("loss"),
        "grad_norm": metrics.get("grad_norm"),
        "server_step_seconds": metrics.get("step_seconds"),
        "server_tps": metrics.get("tps"),
    }
    print(
        f"[{label} {index}] fb={fb_s:.1f}s ({rec['fb_tps']:.0f} tok/s, "
        f"{rec['fb_tps_per_gpu']:.0f} tok/s/GPU) optim={opt_s:.1f}s loss={rec['loss']}",
        flush=True,
    )
    return rec


def main() -> None:
    rng = random.Random(0xB300)
    out: dict = {"config": {
        "model": "zai-org/GLM-5.2-FP8",
        "seq_len": SEQ_LEN,
        "num_gpus": NUM_GPUS,
        "parallelism": {"tp": 1, "pp": 1, "ep": 16, "cp": 16, "etp": 1},
        "lora_rank": 32,
        "lora_alpha": 64,
        "data": "synthetic random tokens (seed 0xB300)",
        "warmup_datums": WARMUP_DATUMS,
        "main_datums": MAIN_DATUMS,
        "main_repeats": MAIN_REPEATS,
    }}

    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        status = client.get("/status").json()
        out["initial_status"] = status
        print(f"[status] world_size={status.get('world_size')} dp={status.get('data_parallel_size')}", flush=True)

        print("[profile] memory_profile/start", flush=True)
        mem_start = submit_and_wait(client, "/memory_profile/start", {}, OP_TIMEOUT_S)
        out["memory_profile_start"] = mem_start

        warmup = drive_window(client, "warmup", 0, [make_datum(rng) for _ in range(WARMUP_DATUMS)])

        print("[profile] runtime_profile/start", flush=True)
        rt_start = submit_and_wait(client, "/runtime_profile/start", {}, OP_TIMEOUT_S)
        out["runtime_profile_start"] = rt_start

        mains = []
        mains.append(drive_window(client, "main", 0, [make_datum(rng) for _ in range(MAIN_DATUMS)]))

        print("[profile] runtime_profile/stop (sync kineto flush, can take minutes)", flush=True)
        rt_stop = submit_and_wait(client, "/runtime_profile/stop", {}, OP_TIMEOUT_S)
        out["runtime_profile_stop"] = rt_stop

        mains.append(drive_window(client, "main", 1, [make_datum(rng) for _ in range(MAIN_DATUMS)]))

        print("[profile] memory_profile/stop (snapshot dump, can take minutes)", flush=True)
        mem_stop = submit_and_wait(client, "/memory_profile/stop", {}, OP_TIMEOUT_S)
        out["memory_profile_stop"] = mem_stop

        # /status can queue behind the profile flushes; be patient.
        final_status = client.get("/status", timeout=600.0).json()
        out["final_status"] = final_status

    out["windows"] = [warmup, *mains]
    main_fb = [w["fb_elapsed_s"] for w in mains]
    main_tokens = sum(w["num_tokens"] for w in mains)
    out["aggregates"] = {
        "main_fb_seconds_mean": sum(main_fb) / len(main_fb),
        "main_tps": main_tokens / sum(main_fb),
        "main_tps_per_gpu": main_tokens / sum(main_fb) / NUM_GPUS,
        "main_optim_seconds_mean": sum(w["optim_elapsed_s"] for w in mains) / len(mains),
        "peak_gpu_memory_bytes": max((final_status.get("gpu_memory") or {}).values(), default=0),
    }
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"[done] results -> {RESULTS_PATH}", flush=True)
    print(json.dumps(out["aggregates"], indent=2), flush=True)


if __name__ == "__main__":
    main()
