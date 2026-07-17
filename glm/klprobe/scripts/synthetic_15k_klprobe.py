#!/usr/bin/env python3
"""Measure live trainer/sampler KL on an exact-length synthetic token sequence."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
import uuid
from pathlib import Path

import httpx
import numpy as np
from baseten.loops import ModelInput, SamplingClient, SamplingParams
from baseten.loops.sampling_client import LocalDeployment
from transformers import AutoTokenizer


def make_synthetic_tokens(tokenizer, seq_len: int) -> list[int]:
    # Keep one normal sequence-start token, then repeat a deterministic text
    # pattern until the requested total sequence length is exact.
    prefix = tokenizer.encode("", add_special_tokens=True)
    pattern = tokenizer.encode(
        " The quick brown fox checks trainer sampler numerical parity.",
        add_special_tokens=False,
    )
    if not pattern:
        raise RuntimeError("tokenizer produced an empty synthetic pattern")
    tokens = list(prefix[:1])
    while len(tokens) < seq_len:
        tokens.extend(pattern[: seq_len - len(tokens)])
    return tokens


def submit_forward(
    trainer_url: str, tokens: list[int], timeout: float = 3600.0
) -> list[float | None]:
    datum = {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
        "loss_fn_inputs": {},
    }
    with httpx.Client(base_url=trainer_url, timeout=timeout) as client:
        response = client.post(
            "/forward",
            json={"data": [datum], "loss_fn": "cross_entropy"},
            headers={"Idempotency-Key": uuid.uuid4().hex},
        )
        response.raise_for_status()
        operation_id = response.json()["operation_id"]
        while True:
            result_response = client.get(f"/operations/{operation_id}", timeout=timeout)
            if result_response.status_code == 408:
                continue
            result_response.raise_for_status()
            payload = result_response.json()
            if payload["status"] == "error":
                raise RuntimeError(payload.get("error", "trainer forward failed"))
            if payload["status"] == "done":
                return payload["result"]["loss_fn_outputs"][0]["logprobs"]["data"]


async def sampler_prompt_logprobs(
    sampler_url: str, model: str, tokens: list[int]
) -> tuple[list[float | None], int | None]:
    client = SamplingClient(
        base_model=model,
        deployment=LocalDeployment(base_url=sampler_url),
    )
    result = await client.sample_async(
        prompt=ModelInput.from_ints(tokens),
        num_samples=1,
        sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=0,
    )
    if result.prompt_logprobs is None:
        raise RuntimeError("sampler omitted prompt logprobs")
    if result.prompt_token_ids is not None and result.prompt_token_ids != tokens:
        raise RuntimeError("sampler changed the synthetic prompt token ids")
    return result.prompt_logprobs, result.policy_version


def metrics(behavior: list[float], target: list[float]) -> dict:
    if len(behavior) != len(target):
        raise RuntimeError(
            f"unaligned scores: sampler={len(behavior)} trainer={len(target)}"
        )
    ratios = [
        target_lp - behavior_lp for behavior_lp, target_lp in zip(behavior, target)
    ]
    absolute = [abs(ratio) for ratio in ratios]
    weights = [math.exp(max(-50.0, min(50.0, ratio))) for ratio in ratios]
    weight_sum = sum(weights)
    squared_weight_sum = sum(weight * weight for weight in weights)
    count = len(ratios)
    return {
        "tokens": count,
        "k3": sum(math.exp(-ratio) + ratio - 1.0 for ratio in ratios) / count,
        "mean_abs": sum(absolute) / count,
        "rms": math.sqrt(sum(value * value for value in absolute) / count),
        "max_abs": max(absolute),
        "ess_over_n": ((weight_sum * weight_sum) / squared_weight_sum) / count,
        "clip_fraction": sum(1 for weight in weights if weight < 0.8 or weight > 1.2)
        / count,
        "mean_log_ratio": sum(ratios) / count,
        "tail_counts": {
            f"abs_r_gt_{threshold}": sum(
                1 for ratio in ratios if abs(ratio) > threshold
            )
            for threshold in (1, 2, 5, 10)
        },
        "positive_tail_counts": {
            f"r_gt_{threshold}": sum(1 for ratio in ratios if ratio > threshold)
            for threshold in (1, 2, 5, 10)
        },
    }


async def run(args) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokens = make_synthetic_tokens(tokenizer, args.seq_len)

    started = time.perf_counter()
    sampler_scores, policy_version = await sampler_prompt_logprobs(
        args.sampler_url, args.model, tokens
    )
    sampler_seconds = time.perf_counter() - started

    started = time.perf_counter()
    trainer_wire = await asyncio.to_thread(submit_forward, args.trainer_url, tokens)
    trainer_seconds = time.perf_counter() - started

    # Sampler prompt score i predicts token i. Trainer wire score i predicts
    # token i+1, so these slices refer to the same 14,999 target tokens.
    sampler_aligned = sampler_scores[1:]
    trainer_aligned = trainer_wire[: args.seq_len - 1]
    if any(value is None for value in sampler_aligned + trainer_aligned):
        raise RuntimeError("aligned score region contains missing logprobs")
    behavior = [float(value) for value in sampler_aligned]
    target = [float(value) for value in trainer_aligned]
    result = {
        "sequence_length": args.seq_len,
        "scored_tokens": args.seq_len - 1,
        "sampler_policy_version": policy_version,
        "sampler_seconds": sampler_seconds,
        "trainer_seconds": trainer_seconds,
        **metrics(behavior, target),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        output.with_suffix(".npz"),
        tokens=np.asarray(tokens, dtype=np.int64),
        behavior=np.asarray(behavior, dtype=np.float32),
        target=np.asarray(target, dtype=np.float32),
    )
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-url", default="http://127.0.0.1:8001")
    parser.add_argument("--sampler-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seq-len", type=int, default=15_000)
    parser.add_argument("--output", required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
