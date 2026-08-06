#!/usr/bin/env python3
"""Repeat the seven-datum LPS-1003 payload and retain every logprob."""

import gzip
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path


PAYLOAD = Path(sys.argv[1])
URL = sys.argv[2].rstrip("/")
OUTPUT = Path(sys.argv[3])
REPS = int(sys.argv[4])
EXPECTED_LENGTHS = [41380, 30060, 20895, 12168, 60180, 34646, 55181]

body = PAYLOAD.read_bytes()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def request(method: str, path: str, data: bytes | None = None, timeout: int = 180):
    req = urllib.request.Request(
        URL + path,
        method=method,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Idempotency-Key": uuid.uuid4().hex,
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.status, response.read()


def run_forward():
    status, raw = request("POST", "/forward", body)
    assert status == 202, f"submit {status}: {raw[:300]}"
    operation_id = json.loads(raw)["operation_id"]
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        try:
            _, raw = request("GET", f"/operations/{operation_id}", timeout=60)
        except Exception as exc:  # noqa: BLE001
            print(f"poll error: {exc}", flush=True)
            time.sleep(5)
            continue
        operation = json.loads(raw)
        if operation.get("status") == "done":
            return operation["result"]
        if operation.get("status") == "error":
            raise RuntimeError(str(operation.get("error"))[:2000])
        time.sleep(1)
    raise TimeoutError("operation did not complete within 900 seconds")


with gzip.open(OUTPUT, "xt", compresslevel=6) as output:
    for rep in range(REPS):
        started = time.time()
        result = run_forward()
        logprobs = []
        for item in result["loss_fn_outputs"]:
            values = item["logprobs"]
            logprobs.append(values["data"] if isinstance(values, dict) else values)
        lengths = [len(values) for values in logprobs]
        if lengths != EXPECTED_LENGTHS:
            raise RuntimeError(f"incomplete logprobs: got {lengths}, expected {EXPECTED_LENGTHS}")
        nlls = [-sum(values) / len(values) for values in logprobs]
        record = {
            "rep": rep,
            "ts": time.time(),
            "wall_s": time.time() - started,
            "nlls": nlls,
            "logprobs": logprobs,
        }
        output.write(json.dumps(record, separators=(",", ":")) + "\n")
        output.flush()
        state = "DESTROYED" if any(value > 4.4 for value in nlls[4:]) else "ok"
        print(
            f"rep={rep:03d} wall={record['wall_s']:.1f}s state={state} "
            + "nlls="
            + " ".join(f"{value:.4f}" for value in nlls),
            flush=True,
        )

print(f"SOAK DONE reps={REPS} output={OUTPUT}", flush=True)
