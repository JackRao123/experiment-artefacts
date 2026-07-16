#!/usr/bin/env python3
"""Forward-only scalar loss parity driver for GLM PP16 and CP32 topologies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


DATUM_LENGTHS = [65521, 32779, 8197]
TEMPERATURES = [0.7, 1.0, 1.3]
LOSS_CONFIGS: list[tuple[str, dict[str, float]]] = [
    (
        "dppo",
        {
            "dppo_mask_high": 0.2,
            "dppo_mask_low": 0.2,
            "adv_tau": 1.0,
            "kl_tau": 0.001,
        },
    ),
    ("importance_sampling", {}),
    (
        "ppo",
        {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
    ),
    (
        "cispo",
        {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
    ),
    ("dro", {"beta": 0.05}),
]


def tensor_data(data: list[int] | list[float], dtype: str) -> dict[str, Any]:
    return {"data": data, "dtype": dtype, "shape": [len(data)]}


def build_datums() -> list[dict[str, Any]]:
    datums: list[dict[str, Any]] = []
    for datum_index, (length, temperature) in enumerate(
        zip(DATUM_LENGTHS, TEMPERATURES, strict=True)
    ):
        tokens = [
            100 + ((997 * datum_index + position) % 30000)
            for position in range(length)
        ]
        targets = [
            100 + ((997 * datum_index + position + 1) % 30000)
            for position in range(length)
        ]
        logprobs = [
            -0.05 - 0.07 * ((position + 13 * datum_index) % 37)
            for position in range(length)
        ]
        prompt_length = length // 2
        advantages = [0.0] * length
        for position in range(prompt_length, length):
            if position % 11 == 0:
                advantages[position] = 0.0
            elif position % 5 == 0:
                advantages[position] = -0.5
            else:
                advantages[position] = 1.0
        advantages[-1] = 1.0
        temperatures = [temperature] * length

        datums.append(
            {
                "model_input": {
                    "chunks": [{"type": "encoded_text", "tokens": tokens}]
                },
                "loss_fn_inputs": {
                    "target_tokens": tensor_data(targets, "int64"),
                    "logprobs": tensor_data(logprobs, "float32"),
                    "advantages": tensor_data(advantages, "float32"),
                    "temperatures": tensor_data(temperatures, "float32"),
                },
            }
        )
    return datums


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trainer_provenance(trainer_src: Path) -> dict[str, Any]:
    sha = subprocess.check_output(
        ["git", "-C", str(trainer_src), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(trainer_src), "status", "--porcelain=v1"],
        text=True,
    )
    diff = subprocess.check_output(
        ["git", "-C", str(trainer_src), "diff", "--binary", "HEAD"]
    )
    return {
        "sha": sha,
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None,
        "modified_paths": [line[3:] for line in status.splitlines()],
    }


def submit_and_wait(
    client: Any, op_path: str, *, body: dict[str, Any], timeout: float
) -> dict[str, Any]:
    response = client.post(
        op_path,
        json=body,
        headers={"Idempotency-Key": uuid.uuid4().hex},
    )
    if response.status_code != 202:
        raise RuntimeError(
            f"{op_path} submit failed {response.status_code}: {response.text[:2000]}"
        )

    operation_id = response.json()["operation_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result_response = client.get(
            f"/operations/{operation_id}", timeout=min(timeout, 35.0)
        )
        if result_response.status_code == 408:
            continue
        result_response.raise_for_status()
        payload = result_response.json()
        if payload.get("status") == "done":
            return payload["result"]
        if payload.get("status") == "error":
            raise RuntimeError(
                f"{op_path} operation failed: {payload.get('error', '')[:8000]}"
            )
    raise TimeoutError(f"{op_path} exceeded {timeout}s")


def validate_result(loss_fn: str, result: dict[str, Any]) -> tuple[float, list[list[int]]]:
    loss = result.get("loss")
    if not isinstance(loss, (int, float)) or not math.isfinite(loss):
        raise ValueError(f"{loss_fn}: non-finite or missing scalar loss: {loss!r}")

    outputs = result.get("loss_fn_outputs")
    if not isinstance(outputs, list) or len(outputs) != len(DATUM_LENGTHS):
        raise ValueError(
            f"{loss_fn}: expected {len(DATUM_LENGTHS)} output rows, got "
            f"{type(outputs).__name__} length "
            f"{len(outputs) if isinstance(outputs, list) else 'n/a'}"
        )

    row_shapes: list[list[int]] = []
    for row_index, (output, expected_length) in enumerate(
        zip(outputs, DATUM_LENGTHS, strict=True)
    ):
        logprobs = output.get("logprobs") if isinstance(output, dict) else None
        data = logprobs.get("data") if isinstance(logprobs, dict) else None
        shape = logprobs.get("shape") if isinstance(logprobs, dict) else None
        if not isinstance(data, list) or len(data) != expected_length:
            raise ValueError(
                f"{loss_fn}: row {row_index} data length "
                f"{len(data) if isinstance(data, list) else 'n/a'}, "
                f"expected {expected_length}"
            )
        expected_shape = [expected_length]
        if shape != expected_shape:
            raise ValueError(
                f"{loss_fn}: row {row_index} shape {shape!r}, "
                f"expected {expected_shape!r}"
            )
        row_shapes.append(shape)
    return float(loss), row_shapes


def run(args: argparse.Namespace) -> None:
    import httpx

    config_path = Path(args.config).resolve()
    trainer_src = Path(args.trainer_src).resolve()
    datums = build_datums()
    results: dict[str, Any] = {
        "topology": args.topology,
        "trainer": trainer_provenance(trainer_src),
        "config": {
            "path": config_path.name,
            "sha256": file_sha256(config_path),
        },
        "payload": {
            "sha256": canonical_sha256(datums),
            "datum_lengths": DATUM_LENGTHS,
            "temperatures": TEMPERATURES,
        },
        "losses": {},
    }

    with httpx.Client(base_url=args.trainer_url, timeout=args.timeout) as client:
        client.get("/health", timeout=30.0).raise_for_status()
        status_response = client.get("/status", timeout=30.0)
        status_response.raise_for_status()
        results["status"] = status_response.json()

        for loss_fn, loss_fn_config in LOSS_CONFIGS:
            start = time.perf_counter()
            result = submit_and_wait(
                client,
                "/forward",
                body={
                    "data": datums,
                    "loss_fn": loss_fn,
                    "loss_fn_config": loss_fn_config,
                },
                timeout=args.timeout,
            )
            loss, row_shapes = validate_result(loss_fn, result)
            duration = time.perf_counter() - start
            results["losses"][loss_fn] = {
                "config": loss_fn_config,
                "loss": loss,
                "duration_seconds": duration,
                "row_shapes": row_shapes,
            }
            print(
                f"[parity] {loss_fn:20s} loss={loss:.9g} "
                f"duration={duration:.1f}s rows={row_shapes}",
                flush=True,
            )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"[parity] wrote {output_path}", flush=True)


def compare(path_a: str, path_b: str) -> int:
    a = json.loads(Path(path_a).read_text())
    b = json.loads(Path(path_b).read_text())
    if a["trainer"] != b["trainer"]:
        raise ValueError(
            f"trainer provenance mismatch: {a['trainer']} != {b['trainer']}"
        )
    if a["payload"]["sha256"] != b["payload"]["sha256"]:
        raise ValueError("payload hash mismatch")

    failed = False
    print("loss_fn             pp16_loss        cp32_loss        abs_delta       rel_delta verdict")
    for loss_fn, _ in LOSS_CONFIGS:
        loss_a = float(a["losses"][loss_fn]["loss"])
        loss_b = float(b["losses"][loss_fn]["loss"])
        abs_delta = abs(loss_a - loss_b)
        rel_delta = abs_delta / max(abs(loss_a), abs(loss_b), 1e-300)
        passed = math.isclose(loss_a, loss_b, rel_tol=1e-2, abs_tol=1e-5)
        failed |= not passed
        print(
            f"{loss_fn:20s} {loss_a:16.9g} {loss_b:16.9g} "
            f"{abs_delta:15.8g} {rel_delta:15.8g} "
            f"{'PASS' if passed else 'FAIL'}"
        )
    print(f"PARITY: {'FAIL' if failed else 'PASS'}")
    return int(failed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-url", default="http://127.0.0.1:8000")
    parser.add_argument("--trainer-src")
    parser.add_argument("--config")
    parser.add_argument("--topology")
    parser.add_argument("--out")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--compare", nargs=2, metavar=("PP16_JSON", "CP32_JSON"))
    args = parser.parse_args()
    if args.compare:
        sys.exit(compare(*args.compare))
    missing = [
        name
        for name in ("trainer_src", "config", "topology", "out")
        if not getattr(args, name)
    ]
    if missing:
        parser.error(f"required in run mode: {', '.join('--' + name.replace('_', '-') for name in missing)}")
    run(args)


if __name__ == "__main__":
    main()
