#!/usr/bin/env python3
"""Production-shaped GLM-5.2 CP32 trainer smoke through the HTTP API."""

from __future__ import annotations

import argparse
import json
import math
import time
import uuid
from collections.abc import Mapping
from typing import Any

import httpx
from transformers import AutoTokenizer


def tensor_data(data: list[int] | list[float], dtype: str) -> dict[str, Any]:
    return {"data": data, "dtype": dtype, "shape": [len(data)]}


def synthetic_ramp_datum(seq_len: int, vocab_period: int = 30_000) -> tuple[dict, int]:
    tokens = [100 + (i % vocab_period) for i in range(seq_len)]
    targets = tokens[1:] + [-100]
    first_supervised = seq_len // 2
    weights = [0.0] * first_supervised + [1.0] * (seq_len - first_supervised)
    return (
        {
            "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
            "loss_fn_inputs": {
                "target_tokens": tensor_data(targets, "int64"),
                "weights": tensor_data(weights, "float32"),
            },
        },
        first_supervised,
    )


def text_overfit_datum(
    model: str, prompt: str, answer: str
) -> tuple[dict, int, list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=True,
        local_files_only=True,
    )
    rendered_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(rendered_prompt, Mapping):
        rendered_prompt = rendered_prompt["input_ids"]
    if rendered_prompt and isinstance(rendered_prompt[0], list):
        rendered_prompt = rendered_prompt[0]
    prompt_tokens = [int(token) for token in rendered_prompt]
    answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        answer_tokens.append(tokenizer.eos_token_id)
    tokens = list(prompt_tokens) + answer_tokens
    targets = tokens[1:] + [-100]
    first_supervised = len(prompt_tokens) - 1
    weights = [
        1.0 if first_supervised <= i < len(tokens) - 1 else 0.0
        for i in range(len(tokens))
    ]
    return (
        {
            "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
            "loss_fn_inputs": {
                "target_tokens": tensor_data(targets, "int64"),
                "weights": tensor_data(weights, "float32"),
            },
        },
        first_supervised,
        answer_tokens,
    )


def submit_and_wait(
    client: httpx.Client,
    path: str,
    body: dict[str, Any],
    *,
    timeout: float = 3600,
) -> dict[str, Any]:
    response = client.post(
        path,
        json=body,
        headers={"Idempotency-Key": uuid.uuid4().hex},
        timeout=60,
    )
    if response.status_code != 202:
        raise RuntimeError(f"{path} submit failed {response.status_code}: {response.text}")
    operation_id = response.json()["operation_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        poll = client.get(f"/operations/{operation_id}", timeout=35)
        if poll.status_code == 408:
            continue
        poll.raise_for_status()
        payload = poll.json()
        if payload.get("status") == "done":
            return payload["result"]
        if payload.get("status") == "error":
            raise RuntimeError(f"{path} failed: {payload.get('error')}")
        time.sleep(1)
    raise TimeoutError(f"{path} timed out after {timeout}s")


def check_forward_result(
    result: dict[str, Any],
    *,
    seq_len: int,
    first_supervised: int,
) -> dict[str, bool]:
    rows = result.get("loss_fn_outputs") or []
    row = (rows[0].get("logprobs") or {}).get("data") if len(rows) == 1 else None
    checks = {
        "finite_loss": math.isfinite(float(result["loss"])),
        "one_logprob_row": len(rows) == 1,
        "logprob_length": row is not None and len(row) == seq_len,
    }
    if row is None:
        return checks
    supervised = row[first_supervised : seq_len - 1]
    checks.update(
        {
            "supervised_logprobs_finite": bool(supervised)
            and all(math.isfinite(value) for value in supervised),
            "masked_logprobs_zero": all(
                value == 0.0 for value in row[:first_supervised]
            ),
            "last_logprob_zero": row[-1] == 0.0,
            "logprobs_reproduce_loss": abs(
                -sum(supervised) / len(supervised) - float(result["loss"])
            )
            / max(abs(float(result["loss"])), 1e-9)
            < 1e-4,
        }
    )
    return checks


def train_step(
    client: httpx.Client,
    datum: dict[str, Any],
    *,
    first_supervised: int,
    learning_rate: float,
    label: str,
) -> tuple[float, float]:
    started = time.perf_counter()
    forward = submit_and_wait(
        client,
        "/forward_backward",
        {"data": [datum], "loss_fn": "cross_entropy"},
    )
    checks = check_forward_result(
        forward,
        seq_len=len(datum["model_input"]["chunks"][0]["tokens"]),
        first_supervised=first_supervised,
    )
    if not all(checks.values()):
        raise AssertionError(f"{label} forward checks failed: {checks}")
    optim = submit_and_wait(
        client,
        "/optim_step",
        {
            "adam_params": {
                "learning_rate": learning_rate,
                "beta1": 0.9,
                "beta2": 0.95,
                "eps": 1e-12,
                "weight_decay": 0.0,
                "grad_clip_norm": 0.0,
            }
        },
    )
    loss = float(forward["loss"])
    grad_norm = float(optim["metrics"]["grad_norm"])
    if not math.isfinite(grad_norm):
        raise AssertionError(f"{label} non-finite grad_norm: {grad_norm}")
    print(
        f"{label}: loss={loss:.8f} grad_norm={grad_norm:.6g} "
        f"elapsed={time.perf_counter() - started:.1f}s checks={checks}",
        flush=True,
    )
    return loss, grad_norm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="zai-org/GLM-5.2-FP8")
    parser.add_argument("--long-steps", type=int, default=2)
    parser.add_argument("--overfit-steps", type=int, default=8)
    parser.add_argument("--run-id", default="pr592-smoke-98e11451")
    args = parser.parse_args()

    with httpx.Client(base_url=args.trainer_url, timeout=3600) as client:
        client.get("/health", timeout=30).raise_for_status()

        long_datum, long_first_supervised = synthetic_ramp_datum(131_072)
        long_losses = [
            train_step(
                client,
                long_datum,
                first_supervised=long_first_supervised,
                learning_rate=1e-4,
                label=f"131k step {step + 1}/{args.long_steps}",
            )[0]
            for step in range(args.long_steps)
        ]
        status = client.get("/status", timeout=60).json()
        allocated = max((status.get("gpu_memory") or {"": 0}).values())
        print(
            f"131k summary: losses={long_losses} "
            f"rank0_alloc={allocated / 2**30:.2f}GiB",
            flush=True,
        )

        prompt = "What's your favourite food?"
        answer = "French foie gras"
        short_datum, short_first_supervised, answer_tokens = text_overfit_datum(
            args.model, prompt, answer
        )
        print(
            f"overfit datum: prompt={prompt!r} answer={answer!r} "
            f"seq_len={len(short_datum['model_input']['chunks'][0]['tokens'])} "
            f"answer_tokens={answer_tokens}",
            flush=True,
        )
        overfit_losses = [
            train_step(
                client,
                short_datum,
                first_supervised=short_first_supervised,
                learning_rate=1e-3,
                label=f"overfit step {step + 1}/{args.overfit_steps}",
            )[0]
            for step in range(args.overfit_steps)
        ]
        if overfit_losses[-1] >= overfit_losses[0]:
            raise AssertionError(f"overfit loss did not decrease: {overfit_losses}")
        print(f"overfit summary: losses={overfit_losses}", flush=True)

        state = submit_and_wait(
            client,
            "/save_state",
            {"name": "pr592-smoke-state", "run_id": args.run_id},
            timeout=3600,
        )
        print("save_state:", json.dumps(state, sort_keys=True), flush=True)

        sampler = submit_and_wait(
            client,
            "/save_weights_for_sampler",
            {
                "name": "pr592-smoke-sampler",
                "run_id": args.run_id,
                "bump_version": False,
            },
            timeout=3600,
        )
        print(
            "save_weights_for_sampler:",
            json.dumps(sampler, sort_keys=True),
            flush=True,
        )
        client.get("/health", timeout=30).raise_for_status()
        print("RESULT: PASS", flush=True)


if __name__ == "__main__":
    main()
