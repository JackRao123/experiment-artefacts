from __future__ import annotations

import json
import random
import time
import uuid
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8001"
SEQ_LEN = 262_144
NUM_GPUS = 16
VOCAB_SIZE = 154_880
DATUM_SEQUENCE = [4, 4, 4, 5, 6]
TIMEOUT_S = 3600.0
OUT_PATH = Path(
    "/root/.cache/user_artifacts/lps1062_bench/"
    "glm53-b200-pp2-262k-sequence.json"
)


def make_datum(rng: random.Random) -> dict:
    tokens = [rng.randrange(VOCAB_SIZE) for _ in range(SEQ_LEN)]
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
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


def main() -> None:
    rng = random.Random(0xB300)
    records = []
    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        status = client.get("/status").json()
        print(
            f"[status] world={status['world_size']} pp={status['pipeline_parallel_size']} "
            f"cp={status['context_parallel_size']} ep={status['expert_parallel_size']} "
            f"max_seq_len={status['max_seq_len']}",
            flush=True,
        )
        for index, datums in enumerate(DATUM_SEQUENCE):
            batch = [make_datum(rng) for _ in range(datums)]
            started = time.perf_counter()
            fb = submit_and_wait(client, "/forward_backward", {"data": batch})
            fb_seconds = time.perf_counter() - started
            optim = submit_and_wait(
                client,
                "/optim_step",
                {"adam_params": {"learning_rate": 1e-5}},
            )
            metrics = (optim or {}).get("metrics") or {}
            tokens = datums * SEQ_LEN
            record = {
                "index": index,
                "datums": datums,
                "tokens": tokens,
                "fb_seconds": fb_seconds,
                "tps_per_gpu": tokens / fb_seconds / NUM_GPUS,
                "loss": (fb or {}).get("loss"),
                "grad_norm": metrics.get("grad_norm"),
                "step_seconds": metrics.get("step_seconds"),
                "peak_allocated_bytes": metrics.get("peak_allocated_bytes"),
                "peak_reserved_bytes": metrics.get("peak_reserved_bytes"),
                "device_total_bytes": metrics.get("device_total_bytes"),
            }
            records.append(record)
            print(
                f"[step {index}] datums={datums} fb={fb_seconds:.1f}s "
                f"tps/gpu={record['tps_per_gpu']:.0f} "
                f"allocated={record['peak_allocated_bytes'] / 1e9:.2f}GB "
                f"reserved={record['peak_reserved_bytes'] / 1e9:.2f}GB",
                flush=True,
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"status": status, "records": records}, indent=2))
    print(f"[done] {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
