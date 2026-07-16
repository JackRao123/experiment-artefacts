#!/usr/bin/env python3
"""Forward-only DPO parity for GLM PP16/EP2/64k and CP32/EP32/131k."""

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

DATUM_LENGTHS = [48001, 48001]
BETA = 1.0e-4


def tensor_data(data: list[int] | list[float], dtype: str) -> dict[str, Any]:
    return {"data": data, "dtype": dtype, "shape": [len(data)]}


def build_datums() -> list[dict[str, Any]]:
    datums = []
    for datum_index, length in enumerate(DATUM_LENGTHS):
        tokens = [
            100 + ((7919 * datum_index + 104729 * position) % 30000)
            for position in range(length)
        ]
        targets = [
            100 + ((7919 * datum_index + 104729 * (position + 1)) % 30000)
            for position in range(length)
        ]
        prompt_length = length // 2
        weights = [0.0] * prompt_length + [1.0] * (length - prompt_length)
        ref_logprobs = [
            -5.0 - 0.01 * ((position + 11 * datum_index) % 101)
            for position in range(length)
        ]
        datums.append(
            {
                "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
                "loss_fn_inputs": {
                    "target_tokens": tensor_data(targets, "int64"),
                    "weights": tensor_data(weights, "float32"),
                    "ref_logprobs": tensor_data(ref_logprobs, "float32"),
                },
            }
        )
    return datums


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trainer_provenance(trainer_src: Path) -> dict[str, Any]:
    sha = subprocess.check_output(
        ["git", "-C", str(trainer_src), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(trainer_src), "status", "--porcelain=v1"],
        text=True,
    )
    diff = subprocess.check_output(
        ["git", "-C", str(trainer_src), "diff", "--binary", "HEAD"],
    )
    return {
        "sha": sha,
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None,
        "modified_paths": [line[3:] for line in status.splitlines()],
    }


def submit_and_wait(
    client: Any,
    path: str,
    *,
    body: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    response = client.post(
        path,
        json=body,
        headers={"Idempotency-Key": uuid.uuid4().hex},
    )
    if response.status_code != 202:
        raise RuntimeError(
            f"{path} submit failed {response.status_code}: {response.text[:2000]}"
        )
    operation_id = response.json()["operation_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = client.get(
            f"/operations/{operation_id}",
            timeout=min(timeout, 35.0),
        )
        if result.status_code == 408:
            continue
        result.raise_for_status()
        payload = result.json()
        if payload.get("status") == "done":
            return payload["result"]
        if payload.get("status") == "error":
            raise RuntimeError(f"{path} failed: {payload.get('error', '')[:8000]}")
    raise TimeoutError(path)


def _softplus_negative(logit: float) -> float:
    return max(0.0, -logit) + math.log1p(math.exp(-abs(logit)))


def validate_and_summarize(
    result: dict[str, Any],
    datums: list[dict[str, Any]],
) -> dict[str, Any]:
    loss = result.get("loss")
    if not isinstance(loss, (int, float)) or not math.isfinite(loss):
        raise ValueError(f"non-finite or missing DPO loss: {loss!r}")
    outputs = result.get("loss_fn_outputs")
    if not isinstance(outputs, list) or len(outputs) != 2:
        raise ValueError(f"expected two DPO logprob rows, got {outputs!r}")

    policy_sequence_logps = []
    reference_sequence_logps = []
    row_shapes = []
    active_finite = []
    for output, datum, expected_length in zip(
        outputs,
        datums,
        DATUM_LENGTHS,
        strict=True,
    ):
        tensor = output["logprobs"]
        values = tensor["data"]
        if tensor["shape"] != [expected_length] or len(values) != expected_length:
            raise ValueError(
                f"DPO row shape mismatch: shape={tensor['shape']} len={len(values)} "
                f"expected={expected_length}"
            )
        weights = datum["loss_fn_inputs"]["weights"]["data"]
        refs = datum["loss_fn_inputs"]["ref_logprobs"]["data"]
        active_values = [value for value, weight in zip(values, weights) if weight > 0]
        if not all(math.isfinite(value) for value in active_values):
            raise ValueError("DPO output contains non-finite active logprobs")
        policy_sequence_logps.append(
            sum(value * weight for value, weight in zip(values, weights))
        )
        reference_sequence_logps.append(
            sum(value * weight for value, weight in zip(refs, weights))
        )
        row_shapes.append(tensor["shape"])
        active_finite.append(len(active_values))

    logit = BETA * (
        (policy_sequence_logps[0] - reference_sequence_logps[0])
        - (policy_sequence_logps[1] - reference_sequence_logps[1])
    )
    reconstructed_loss = _softplus_negative(logit)
    return {
        "loss": float(loss),
        "row_shapes": row_shapes,
        "active_finite_counts": active_finite,
        "policy_sequence_logps": policy_sequence_logps,
        "reference_sequence_logps": reference_sequence_logps,
        "dpo_logit": logit,
        "reconstructed_loss": reconstructed_loss,
        "reconstruction_abs_delta": abs(float(loss) - reconstructed_loss),
    }


def run(args: argparse.Namespace) -> None:
    import httpx

    config_path = Path(args.config).resolve()
    trainer_src = Path(args.trainer_src).resolve()
    datums = build_datums()
    output: dict[str, Any] = {
        "topology": args.topology,
        "mode": "forward_backward" if args.backward else "forward",
        "trainer": trainer_provenance(trainer_src),
        "config": {
            "path": config_path.name,
            "sha256": file_sha256(config_path),
        },
        "payload": {
            "sha256": canonical_sha256(datums),
            "datum_lengths": DATUM_LENGTHS,
            "beta": BETA,
        },
    }

    with httpx.Client(base_url=args.trainer_url, timeout=args.timeout) as client:
        client.get("/health", timeout=30.0).raise_for_status()
        output["status"] = client.get("/status", timeout=30.0).json()
        started = time.perf_counter()
        result = submit_and_wait(
            client,
            "/forward_backward" if args.backward else "/forward",
            body={
                "data": datums,
                "loss_fn": "dpo",
                "loss_fn_config": {"beta": BETA},
            },
            timeout=args.timeout,
        )
        output["dpo"] = validate_and_summarize(result, datums)
        output["dpo"]["duration_seconds"] = time.perf_counter() - started
        if args.backward:
            optim = submit_and_wait(
                client,
                "/optim_step",
                body={
                    "adam_params": {
                        "learning_rate": 1e-10,
                        "beta1": 0.9,
                        "beta2": 0.95,
                    }
                },
                timeout=args.timeout,
            )
            output["optim_step"] = optim

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        f"[production-dpo] topology={args.topology} "
        f"loss={output['dpo']['loss']:.9g} "
        f"reconstruction_delta={output['dpo']['reconstruction_abs_delta']:.3e} "
        f"duration={output['dpo']['duration_seconds']:.1f}s",
        flush=True,
    )
    print(f"[production-dpo] wrote {path}", flush=True)


def compare(path_a: str, path_b: str) -> int:
    a = json.loads(Path(path_a).read_text())
    b = json.loads(Path(path_b).read_text())
    if a["trainer"] != b["trainer"]:
        raise ValueError("trainer provenance mismatch")
    if a["payload"] != b["payload"]:
        raise ValueError("payload mismatch")

    loss_a = float(a["dpo"]["loss"])
    loss_b = float(b["dpo"]["loss"])
    abs_delta = abs(loss_a - loss_b)
    rel_delta = abs_delta / max(abs(loss_a), abs(loss_b), 1e-300)
    reconstruction_ok = all(
        float(item["dpo"]["reconstruction_abs_delta"]) < 1e-3 for item in (a, b)
    )
    parity_ok = math.isclose(loss_a, loss_b, rel_tol=1e-2, abs_tol=1e-5)
    passed = reconstruction_ok and parity_ok
    print(
        f"PP16 loss={loss_a:.9g} CP32 loss={loss_b:.9g} "
        f"abs_delta={abs_delta:.3e} rel_delta={rel_delta:.3e}"
    )
    print(
        f"wire reconstruction deltas: "
        f"PP16={a['dpo']['reconstruction_abs_delta']:.3e} "
        f"CP32={b['dpo']['reconstruction_abs_delta']:.3e}"
    )
    print(f"PARITY: {'PASS' if passed else 'FAIL'}")
    return int(not passed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-url", default="http://127.0.0.1:8000")
    parser.add_argument("--trainer-src")
    parser.add_argument("--config")
    parser.add_argument("--topology")
    parser.add_argument("--out")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--backward", action="store_true")
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
        parser.error(f"missing run arguments: {', '.join(missing)}")
    run(args)


if __name__ == "__main__":
    main()
