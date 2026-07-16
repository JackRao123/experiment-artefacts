#!/usr/bin/env python3
"""CP1-vs-CP2 parity driver for DPO's sequence-level THD reduction."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from pathlib import Path

LOSS_CONFIGS = [
    ("cross_entropy", {}),
    ("dpo", {"beta": 0.1, "label_smoothing": 0.0}),
]
DATUM_LENGTHS = [173, 173, 96, 96, 41, 41]
VOCAB = 1800


def _rng(seed: int):
    state = seed & 0x7FFFFFFF

    def nxt() -> float:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF

    return nxt


def _tensor(data: list[int] | list[float], dtype: str) -> dict:
    return {"data": data, "dtype": dtype, "shape": [len(data)]}


def build_datums() -> list[dict]:
    datums = []
    for datum_index, length in enumerate(DATUM_LENGTHS):
        rnd = _rng(5000 + datum_index)
        tokens = [1 + int(rnd() * (VOCAB - 2)) for _ in range(length)]
        targets = tokens[1:] + [-100]
        completion_start = max(1, length - 24)
        weights = [0.0] * completion_start + [1.0] * (length - completion_start)
        weights[-1] = 0.0
        ref_logprobs = [
            -7.0 - 0.01 * ((position + 3 * datum_index) % 17)
            for position in range(length)
        ]
        datums.append(
            {
                "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
                "loss_fn_inputs": {
                    "target_tokens": _tensor(targets, "int64"),
                    "weights": _tensor(weights, "float32"),
                    "ref_logprobs": _tensor(ref_logprobs, "float32"),
                },
            }
        )
    return datums


def submit_and_wait(client, path: str, body: dict, timeout: float = 1800.0) -> dict:
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
        result = client.get(f"/operations/{operation_id}", timeout=35.0)
        if result.status_code == 408:
            continue
        result.raise_for_status()
        payload = result.json()
        if payload.get("status") == "done":
            return payload["result"]
        if payload.get("status") == "error":
            raise RuntimeError(f"{path} failed: {payload.get('error', '')[:8000]}")
    raise TimeoutError(path)


def run(args: argparse.Namespace) -> None:
    import httpx

    datums = build_datums()
    output: dict = {"datum_lengths": DATUM_LENGTHS, "losses": {}}
    with httpx.Client(base_url=args.trainer_url, timeout=1800.0) as client:
        client.get("/health", timeout=30.0).raise_for_status()
        output["status"] = client.get("/status", timeout=30.0).json()
        for loss_fn, config in LOSS_CONFIGS:
            started = time.perf_counter()
            fb = submit_and_wait(
                client,
                "/forward_backward",
                {"data": datums, "loss_fn": loss_fn, "loss_fn_config": config},
            )
            rows = [item["logprobs"]["data"] for item in fb.get("loss_fn_outputs", [])]
            optim = submit_and_wait(
                client,
                "/optim_step",
                {
                    "adam_params": {
                        "learning_rate": 1e-10,
                        "beta1": 0.9,
                        "beta2": 0.95,
                    }
                },
            )
            grad_norm = optim["metrics"]["grad_norm"]
            output["losses"][loss_fn] = {
                "loss": fb["loss"],
                "grad_norm": grad_norm,
                "row_lengths": [len(row) for row in rows],
                "logprobs": rows,
                "duration_seconds": time.perf_counter() - started,
            }
            print(
                f"[dpo-parity] {loss_fn} loss={fb['loss']:.9g} "
                f"grad_norm={grad_norm:.9g} rows={[len(row) for row in rows]}",
                flush=True,
            )

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"[dpo-parity] wrote {path}", flush=True)


def _mean_abs_diff(rows_a: list[list[float]], rows_b: list[list[float]]) -> float:
    differences = [
        abs(a - b)
        for row_a, row_b in zip(rows_a, rows_b, strict=True)
        for a, b in zip(row_a, row_b, strict=True)
        if a != 0.0 and b != 0.0
    ]
    return sum(differences) / max(len(differences), 1)


def compare(path_a: str, path_b: str) -> int:
    a = json.loads(Path(path_a).read_text())
    b = json.loads(Path(path_b).read_text())
    failed = False
    grad_ratios = {}
    for loss_fn, _ in LOSS_CONFIGS:
        result_a = a["losses"][loss_fn]
        result_b = b["losses"][loss_fn]
        if result_a["row_lengths"] != DATUM_LENGTHS:
            raise ValueError(f"{loss_fn}: CP1 row shape mismatch")
        if result_b["row_lengths"] != DATUM_LENGTHS:
            raise ValueError(f"{loss_fn}: CP2 row shape mismatch")
        loss_a = float(result_a["loss"])
        loss_b = float(result_b["loss"])
        loss_rel = abs(loss_a - loss_b) / max(abs(loss_a), abs(loss_b), 1e-12)
        logprob_mean = _mean_abs_diff(
            result_a["logprobs"],
            result_b["logprobs"],
        )
        grad_ratio = float(result_b["grad_norm"]) / float(result_a["grad_norm"])
        grad_ratios[loss_fn] = grad_ratio
        passed = (
            math.isfinite(loss_a)
            and math.isfinite(loss_b)
            and loss_rel < 1e-2
            and logprob_mean < 2e-2
        )
        failed |= not passed
        print(
            f"{loss_fn:14s} loss_cp1={loss_a:.9g} loss_cp2={loss_b:.9g} "
            f"rel={loss_rel:.3e} logprob_mean={logprob_mean:.3e} "
            f"grad_ratio={grad_ratio:.6g} {'PASS' if passed else 'FAIL'}"
        )

    ratio_rel = abs(grad_ratios["dpo"] / grad_ratios["cross_entropy"] - 1.0)
    ratio_passed = ratio_rel < 0.02
    failed |= not ratio_passed
    print(
        f"grad-ratio parity: dpo={grad_ratios['dpo']:.6g} "
        f"ce={grad_ratios['cross_entropy']:.6g} rel={ratio_rel:.3e} "
        f"{'PASS' if ratio_passed else 'FAIL'}"
    )
    print(f"PARITY: {'FAIL' if failed else 'PASS'}")
    return int(failed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", default="dpo_parity.json")
    parser.add_argument("--compare", nargs=2, metavar=("CP1_JSON", "CP2_JSON"))
    args = parser.parse_args()
    if args.compare:
        sys.exit(compare(*args.compare))
    run(args)


if __name__ == "__main__":
    main()
