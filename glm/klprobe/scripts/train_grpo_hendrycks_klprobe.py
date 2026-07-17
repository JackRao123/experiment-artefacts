#!/usr/bin/env python3
"""Rich GLM-5.2 CP32 trainer/sampler parity probe.

Each step publishes the current rank-16 adapter, waits for the TP8 sampler to
serve that policy version, samples GSM8K or filtered MATH problems,
teacher-forces every completion through the trainer, captures aligned per-token
logprobs, and only then performs the importance-sampling update.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import time
from pathlib import Path

from baseten.loops import (
    AdamParams,
    Datum,
    ModelInput,
    SamplingClient,
    SamplingParams,
    TensorData,
    TrainingClient,
)
from baseten.loops.models import EncodedTextChunk
from baseten.loops.sampling_client import LocalDeployment

from glm_klprobe_config import (
    dataset_spec,
    k3_from_log_ratio,
    load_dataset_from_spec,
)

SYSTEM_MSG = (
    "You are a math tutor. Reason step by step and give the final answer "
    "in \\boxed{...}."
)
QUESTION_SUFFIX = "\n\nReason step by step and put your final answer in \\boxed{...}."
OUTLIER_ABS_R = 5.0
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


class Renderer:
    def __init__(self, tokenizer, *, enable_thinking: bool):
        self._tokenizer = tokenizer
        self._enable_thinking = enable_thinking

    def build_generation_prompt(self, conversation: list[dict]) -> ModelInput:
        kwargs = {"tokenize": True, "add_generation_prompt": True}
        try:
            ids = self._tokenizer.apply_chat_template(
                conversation, enable_thinking=self._enable_thinking, **kwargs
            )
        except TypeError:
            ids = self._tokenizer.apply_chat_template(conversation, **kwargs)
        if hasattr(ids, "get") and "input_ids" in ids:
            ids = ids["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
        return ModelInput.from_ints(list(ids))

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=True)


def make_conversation(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": question + QUESTION_SUFFIX},
    ]


def last_boxed(text: str) -> str | None:
    index = text.rfind("\\boxed")
    if index < 0:
        return None
    open_brace = text.find("{", index)
    if open_brace < 0:
        return None
    depth = 0
    for cursor in range(open_brace, len(text)):
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : cursor]
    return None


def normalize_math(value: str | None) -> str:
    if not value:
        return ""
    for old, new in (
        ("\\left", ""),
        ("\\right", ""),
        ("\\!", ""),
        ("\\,", ""),
        ("\\ ", ""),
        ("\\$", ""),
        ("$", ""),
        ("\\%", ""),
        ("%", ""),
        ("\\text{", ""),
        ("\\dfrac", "\\frac"),
        ("\\tfrac", "\\frac"),
        ("^{\\circ}", ""),
        ("^\\circ", ""),
        (" ", ""),
    ):
        value = value.replace(old, new)
    return value.strip().rstrip(".").rstrip("}")


def ground_truth(answer: str, dataset: str) -> str:
    if dataset == "math":
        return last_boxed(answer) or ""
    return answer.rpartition("####")[2].strip().replace(",", "")


def grade(response_text: str, answer: str, dataset: str) -> float:
    expected = ground_truth(answer, dataset)
    if dataset == "gsm8k":
        matches = _BOXED_RE.findall(response_text)
        if not matches:
            return 0.0
        return float(matches[-1].replace(",", "").strip() == expected)

    prediction = last_boxed(response_text)
    if prediction is None:
        matches = _BOXED_RE.findall(response_text)
        prediction = matches[-1] if matches else None
    if prediction is None:
        return 0.0
    try:
        from math_verify import parse, verify

        if verify(parse(expected), parse(prediction)):
            return 1.0
    except Exception:
        pass
    return float(normalize_math(prediction) == normalize_math(expected))


def kl_metrics(behavior: list[float], target: list[float]) -> dict:
    if len(behavior) != len(target):
        raise ValueError(
            f"unaligned logprob streams: behavior={len(behavior)}, target={len(target)}"
        )

    pairs = [
        (float(b), float(t))
        for b, t in zip(behavior, target)
        if b is not None
        and t is not None
        and math.isfinite(float(b))
        and math.isfinite(float(t))
    ]
    dropped = len(behavior) - len(pairs)
    if not pairs:
        return {"tokens": 0, "dropped": dropped}

    ratios = [t - b for b, t in pairs]
    importance_weights = [math.exp(max(-50.0, min(50.0, ratio))) for ratio in ratios]
    absolute_ratios = [abs(ratio) for ratio in ratios]
    weight_sum = sum(importance_weights)
    squared_weight_sum = sum(weight * weight for weight in importance_weights)
    n = len(ratios)
    return {
        "tokens": n,
        "dropped": dropped,
        "k3": sum(k3_from_log_ratio(ratio) for ratio in ratios) / n,
        "mean_abs": sum(absolute_ratios) / n,
        "rms": math.sqrt(sum(value * value for value in absolute_ratios) / n),
        "max_abs": max(absolute_ratios),
        "ess_over_n": (
            ((weight_sum * weight_sum) / squared_weight_sum) / n
            if squared_weight_sum
            else 0.0
        ),
        "clip_fraction": (
            sum(1 for weight in importance_weights if weight < 0.8 or weight > 1.2) / n
        ),
        "mean_log_ratio": sum(ratios) / n,
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


def merge_token_streams(
    pairs: list[tuple[list[float], list[float]]],
) -> dict:
    behavior = [value for behavior_values, _ in pairs for value in behavior_values]
    target = [value for _, target_values in pairs for value in target_values]
    return kl_metrics(behavior, target)


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round(fraction * (len(ordered) - 1)))),
    )
    return float(ordered[index])


def tensor(data: list, dtype: str) -> TensorData:
    import torch

    torch_dtype = torch.float32 if dtype == "float32" else torch.int64
    return TensorData.from_torch(torch.tensor(data, dtype=torch_dtype))


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as output:
        output.write(json.dumps(value) + "\n")
        output.flush()
        os.fsync(output.fileno())


async def kl_probe(
    training_client: TrainingClient,
    sequences: list[dict],
    *,
    chunk_size: int,
    capture_dir: Path | None = None,
    step: int | None = None,
    tokenizer=None,
) -> dict:
    """Compare sampler completion logprobs with aligned trainer forward output.

    The trainer wire row has ``wire[k] = log p(full[k+1] | full[:k+1])`` and
    a final sentinel, so completion scores are ``wire[plen-1:plen-1+clen]``.
    """
    pairs: list[tuple[list[float], list[float]]] = []
    for start in range(0, len(sequences), chunk_size):
        batch = sequences[start : start + chunk_size]
        datums = [
            Datum(
                model_input=ModelInput.from_ints(
                    sequence["prompt_ids"] + sequence["tokens"]
                ),
                loss_fn_inputs={},
            )
            for sequence in batch
        ]
        future = await training_client.forward_async(datums, loss_fn="cross_entropy")
        result = await future.result_async()
        if len(result.loss_fn_outputs) != len(batch):
            raise RuntimeError(
                "trainer forward result count does not match submitted sequences"
            )
        for sequence, output in zip(batch, result.loss_fn_outputs):
            wire = output["logprobs"].tolist()
            prompt_length = len(sequence["prompt_ids"])
            completion_length = len(sequence["tokens"])
            target = wire[prompt_length - 1 : prompt_length - 1 + completion_length]
            if len(target) != completion_length:
                raise RuntimeError(
                    "trainer logprob alignment failed: "
                    f"wire={len(wire)} prompt={prompt_length} "
                    f"completion={completion_length} aligned={len(target)}"
                )
            pairs.append(
                (
                    [float(value) for value in sequence["logprobs"]],
                    [
                        float("nan") if value is None else float(value)
                        for value in target
                    ],
                )
            )

    metrics = merge_token_streams(pairs)
    if capture_dir is None:
        return metrics
    if step is None or tokenizer is None:
        raise ValueError("capture requires step and tokenizer")

    import numpy as np

    capture_dir.mkdir(parents=True, exist_ok=True)
    behavior = np.asarray(
        [value for behavior_values, _ in pairs for value in behavior_values],
        dtype=np.float32,
    )
    target = np.asarray(
        [value for _, target_values in pairs for value in target_values],
        dtype=np.float32,
    )
    sequence_lengths = np.asarray(
        [len(behavior_values) for behavior_values, _ in pairs], dtype=np.int32
    )
    sequence_offsets = np.zeros(len(sequence_lengths) + 1, dtype=np.int64)
    sequence_offsets[1:] = np.cumsum(sequence_lengths, dtype=np.int64)
    finite = np.isfinite(behavior) & np.isfinite(target)
    ratios = np.full(behavior.shape, np.nan, dtype=np.float32)
    ratios[finite] = target[finite] - behavior[finite]
    max_sequence_length = int(sequence_lengths.max(initial=0))
    log_ratio_by_position = np.full(
        (len(sequence_lengths), max_sequence_length), np.nan, dtype=np.float32
    )
    k3_by_position = np.full(
        (len(sequence_lengths), max_sequence_length), np.nan, dtype=np.float64
    )
    for sequence_index, (start, end) in enumerate(
        zip(sequence_offsets[:-1], sequence_offsets[1:])
    ):
        sequence_ratios = ratios[start:end]
        log_ratio_by_position[sequence_index, : len(sequence_ratios)] = sequence_ratios
        with np.errstate(over="ignore", invalid="ignore"):
            k3_by_position[sequence_index, : len(sequence_ratios)] = (
                np.exp(sequence_ratios.astype(np.float64)) - sequence_ratios - 1.0
            )
    np.savez_compressed(
        capture_dir / f"step{step:02d}_logprobs.npz",
        behavior=behavior,
        target=target,
        seq_lens=sequence_lengths,
        seq_offsets=sequence_offsets,
        prompt_indices=np.asarray(
            [sequence.get("prompt_index", -1) for sequence in sequences],
            dtype=np.int32,
        ),
        sample_indices=np.asarray(
            [sequence.get("sample_index", -1) for sequence in sequences],
            dtype=np.int32,
        ),
        log_ratio_by_rollout_position=log_ratio_by_position,
        k3_by_rollout_position=k3_by_position,
    )

    absolute_ratios = np.abs(ratios)
    flat_index = 0
    outlier_count = 0
    for sequence_index, (behavior_values, _) in enumerate(pairs):
        sequence = sequences[sequence_index]
        for position in range(len(behavior_values)):
            global_index = flat_index + position
            if (
                not finite[global_index]
                or absolute_ratios[global_index] <= OUTLIER_ABS_R
            ):
                continue
            context_start = max(0, position - 30)
            append_jsonl(
                capture_dir / "outliers.jsonl",
                {
                    "step": step,
                    "seq": sequence_index,
                    "pos": position,
                    "completion_len": len(behavior_values),
                    "token_id": sequence["tokens"][position],
                    "token_text": tokenizer.decode([sequence["tokens"][position]]),
                    "behavior_lp": float(behavior_values[position]),
                    "target_lp": float(target[global_index]),
                    "r": float(ratios[global_index]),
                    "context_text": tokenizer.decode(
                        sequence["tokens"][context_start : position + 1]
                    ),
                },
            )
            outlier_count += 1
        flat_index += len(behavior_values)
    metrics["n_outlier_records"] = outlier_count
    return metrics


def observed_policy_versions(results) -> list[int]:
    return sorted(
        {
            int(result.policy_version)
            for result in results
            if result.policy_version is not None
        }
    )


def verify_sampler_reload(results, required_version: int) -> list[int]:
    versions = observed_policy_versions(results)
    if len(versions) != 1 or versions[0] < required_version:
        raise RuntimeError(
            "sampler did not serve the exported adapter: "
            f"required>={required_version}, observed={versions}"
        )
    if any(result.policy_version is None for result in results):
        raise RuntimeError("sampler response omitted policy_version")
    return versions


def make_probe_sequence(
    prompt: ModelInput,
    sequence,
    *,
    prompt_index: int = -1,
    sample_index: int = -1,
) -> dict:
    if sequence.logprobs is None:
        raise RuntimeError("sampler response omitted completion logprobs")
    if len(sequence.logprobs) != len(sequence.tokens):
        raise RuntimeError(
            "sampler token/logprob lengths differ: "
            f"tokens={len(sequence.tokens)}, logprobs={len(sequence.logprobs)}"
        )
    return {
        "prompt_ids": prompt.to_ints(),
        "tokens": sequence.tokens,
        "logprobs": sequence.logprobs,
        "prompt_index": prompt_index,
        "sample_index": sample_index,
    }


async def evaluate(
    sampling_client: SamplingClient,
    renderer: Renderer,
    rows: list[dict],
    *,
    max_tokens: int,
    dataset: str,
) -> float:
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0)
    results = await asyncio.gather(
        *[
            sampling_client.sample_async(
                prompt=renderer.build_generation_prompt(
                    make_conversation(row["question"])
                ),
                num_samples=1,
                sampling_params=params,
            )
            for row in rows
        ]
    )
    correct = sum(
        grade(
            renderer.decode(result.sequences[0].tokens),
            row["answer"],
            dataset,
        )
        for result, row in zip(results, rows)
    )
    return correct / len(rows)


def build_long_prefix(
    tokenizer,
    renderer: Renderer,
    rows,
    *,
    question_key: str,
    answer_key: str,
    budget: int,
) -> ModelInput:
    parts: list[str] = []
    total = 0
    for question, answer in zip(rows[question_key], rows[answer_key]):
        piece = f"Problem: {question}\n\nSolution: {answer}\n\n"
        total += len(tokenizer.encode(piece, add_special_tokens=False))
        parts.append(piece)
        if total >= budget - 512:
            break
    content = "".join(parts) + (
        "Study the worked examples above, then continue writing new problems "
        "with fully worked solutions in the same style. Do not stop."
    )
    return renderer.build_generation_prompt(
        [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": content},
        ]
    )


async def decode_long(
    sampling_client,
    tokenizer,
    prefix: ModelInput,
    target: int,
    *,
    required_version: int,
    chunk: int = 2048,
) -> tuple[list[int], list[float]]:
    sampled: list[int] = []
    logprobs: list[float] = []
    nudge = (tokenizer.encode("\n", add_special_tokens=False) or [0])[0]
    rounds = 0
    timeouts = 0
    while len(sampled) < target and rounds < 96:
        rounds += 1
        prompt = prefix.append(EncodedTextChunk(tokens=sampled)) if sampled else prefix
        try:
            result = await sampling_client.sample_async(
                prompt=prompt,
                num_samples=1,
                sampling_params=SamplingParams(
                    max_tokens=min(chunk, target - len(sampled)),
                    temperature=1.0,
                    top_p=1.0,
                ),
            )
        except Exception as exc:
            if "timeout" in type(exc).__name__.lower() and chunk > 256:
                timeouts += 1
                if timeouts > 8:
                    raise
                chunk = max(256, chunk // 2)
                print(f"long probe timeout; retrying with chunk={chunk}", flush=True)
                continue
            raise
        verify_sampler_reload([result], required_version)
        sequence = result.sequences[0]
        if sequence.tokens:
            if sequence.logprobs is None or len(sequence.logprobs) != len(
                sequence.tokens
            ):
                raise RuntimeError("long probe response omitted aligned logprobs")
            sampled.extend(sequence.tokens)
            logprobs.extend(float(value) for value in sequence.logprobs)
        if len(sampled) >= target:
            break
        if sequence.stop_reason != "length" or not sequence.tokens:
            sampled.append(nudge)
            # The forced continuation token was not sampled. NaN excludes it
            # from parity metrics while preserving the exact next-round prefix.
            logprobs.append(float("nan"))
    return sampled, logprobs


async def run_long_probe(
    *,
    training_client: TrainingClient,
    tokenizer,
    renderer: Renderer,
    dataset,
    spec,
    args,
    metrics_path: Path,
    capture_dir: Path,
) -> None:
    used = args.steps * args.batch_size
    rows = dataset.select(range(used, min(used + 4000, len(dataset))))
    prefix = build_long_prefix(
        tokenizer,
        renderer,
        rows,
        question_key=spec.question_key,
        answer_key=spec.answer_key,
        budget=args.long_probe_tokens - args.long_probe_decode_tokens,
    )
    sampling_client = await training_client.save_weights_and_get_sampling_client_async(
        name="long-probe"
    )
    required_version = training_client.policy_version
    decode_start = time.perf_counter()
    tokens, logprobs = await decode_long(
        sampling_client,
        tokenizer,
        prefix,
        args.long_probe_decode_tokens,
        required_version=required_version,
    )
    decode_seconds = time.perf_counter() - decode_start
    parity = await kl_probe(
        training_client,
        [
            {
                "prompt_ids": prefix.to_ints(),
                "tokens": tokens,
                "logprobs": logprobs,
                "prompt_index": 0,
                "sample_index": 0,
            }
        ],
        chunk_size=1,
        capture_dir=capture_dir,
        step=args.steps,
        tokenizer=tokenizer,
    )
    record = {
        "event": "long_probe",
        "prefix_tokens": prefix.length,
        "decode_tokens": len(tokens),
        "target_decode_tokens": args.long_probe_decode_tokens,
        "required_policy_version": required_version,
        "decode_s": decode_seconds,
        **parity,
    }
    append_jsonl(metrics_path, record)
    print(
        f"LONG_PROBE prefix={prefix.length} decode={len(tokens)} "
        f"k3={parity['k3']:.6f} tokens={parity['tokens']} "
        f"decode_s={decode_seconds:.0f}",
        flush=True,
    )


async def bootstrap_policy_version(
    *,
    training_client: TrainingClient,
    renderer: Renderer,
    tokenizer,
    row: dict,
) -> dict:
    """Advance version 0 to 1 without changing adapter or optimizer state.

    The sampler reserves public policy version 0 for the base model and ignores
    a version-0 adapter pointer. A single importance-sampling datum with exactly
    zero advantage materializes zero gradients; an Adam step at learning rate
    zero advances the trainer's version while leaving parameters and moments
    unchanged.
    """
    initial_version = training_client.policy_version
    if initial_version != 0:
        raise RuntimeError(
            f"policy bootstrap requires a fresh trainer at version 0, got {initial_version}"
        )

    prompt = renderer.build_generation_prompt(make_conversation(row["question"]))
    token_id = tokenizer.eos_token_id
    if token_id is None:
        token_id = prompt.to_ints()[-1]
    observation_length = prompt.length - 1
    datum = Datum(
        model_input=prompt,
        loss_fn_inputs={
            "target_tokens": tensor([0] * observation_length + [token_id], "int64"),
            "logprobs": tensor([0.0] * prompt.length, "float32"),
            "advantages": tensor([0.0] * prompt.length, "float32"),
        },
    )
    future = await training_client.forward_backward_async(
        [datum], loss_fn="importance_sampling"
    )
    result = await future.result_async()
    optim_future = await training_client.optim_step_async(AdamParams(learning_rate=0.0))
    optim_result = await optim_future.result_async()
    metrics = optim_result.metrics or {}
    final_version = training_client.policy_version
    grad_norm = float(metrics.get("grad_norm", float("nan")))
    if final_version != 1 or not math.isfinite(grad_norm) or grad_norm > 1e-12:
        raise RuntimeError(
            "zero-gradient policy bootstrap was not a no-op: "
            f"version={final_version}, grad_norm={grad_norm}"
        )
    return {
        "event": "policy_version_bootstrap",
        "initial_policy_version": initial_version,
        "final_policy_version": final_version,
        "loss": float(result.loss),
        "grad_norm": grad_norm,
        "learning_rate": 0.0,
    }


async def run_preflight(
    *,
    training_client: TrainingClient,
    renderer: Renderer,
    tokenizer,
    row: dict,
    sampling_params: SamplingParams,
) -> dict:
    sampling_client = await training_client.save_weights_and_get_sampling_client_async(
        name="preflight-step-0"
    )
    required_version = training_client.policy_version
    if required_version <= 0:
        raise RuntimeError(
            f"preflight requires a reloadable policy version, got {required_version}"
        )
    prompt = renderer.build_generation_prompt(make_conversation(row["question"]))
    preflight_params = SamplingParams(
        max_tokens=min(128, sampling_params.max_tokens),
        temperature=1.0,
        top_p=1.0,
        seed=sampling_params.seed,
    )
    result = await sampling_client.sample_async(
        prompt=prompt, num_samples=1, sampling_params=preflight_params
    )
    versions = verify_sampler_reload([result], required_version)
    sequence = result.sequences[0]
    if not sequence.tokens:
        raise RuntimeError("preflight sampler returned an empty completion")
    metrics = await kl_probe(
        training_client,
        [make_probe_sequence(prompt, sequence)],
        chunk_size=1,
    )
    if metrics.get("tokens") != len(sequence.tokens):
        raise RuntimeError(
            "preflight did not align every sampled completion token: "
            f"sampled={len(sequence.tokens)}, scored={metrics.get('tokens')}"
        )
    return {
        "event": "preflight",
        "required_policy_version": required_version,
        "observed_policy_versions": versions,
        "completion_tokens": len(sequence.tokens),
        **metrics,
    }


async def run(args) -> None:
    metrics_path = Path(args.metrics_out)
    capture_dir = Path(args.capture_dir)
    if metrics_path.exists() and not args.allow_existing_output:
        raise FileExistsError(
            f"{metrics_path} already exists; use a fresh run path or "
            "--allow-existing-output"
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    renderer = Renderer(tokenizer, enable_thinking=not args.disable_thinking)
    spec = dataset_spec(args.dataset, args.math_levels, data_seed=args.data_seed)
    dataset = load_dataset_from_spec(spec)
    training_count = args.steps * args.batch_size
    train_rows = [
        {
            "question": dataset[index][spec.question_key],
            "answer": dataset[index][spec.answer_key],
        }
        for index in range(training_count)
    ]
    eval_rows = [
        {
            "question": dataset[index][spec.question_key],
            "answer": dataset[index][spec.answer_key],
        }
        for index in range(training_count, training_count + args.eval_problems)
    ]

    deployment = LocalDeployment(base_url=args.sampler_url)
    training_client = TrainingClient(
        args.trainer_url,
        lora_rank=args.rank,
        base_model=args.model,
        timeout=1800.0,
        ready_timeout=3600.0,
        run_id=args.run_id,
        paired_sampling_deployment=deployment,
    )
    baseline_sampling_client = SamplingClient(
        base_model=args.model, deployment=deployment
    )
    sampling_kwargs = {
        "max_tokens": args.max_tokens,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    if args.sample_seed is not None:
        sampling_kwargs["seed"] = args.sample_seed
    sampling_params = SamplingParams(**sampling_kwargs)
    adam_params = AdamParams(learning_rate=args.learning_rate)

    try:
        append_jsonl(
            metrics_path,
            {
                "event": "run_metadata",
                "run_id": args.run_id,
                "model": args.model,
                "dataset": spec.key,
                "dataset_name": spec.dataset_name,
                "dataset_config": spec.dataset_config,
                "math_levels": list(spec.math_levels),
                "enable_thinking": not args.disable_thinking,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "group_size": args.group_size,
                "max_tokens": args.max_tokens,
                "rank": args.rank,
                "data_seed": args.data_seed,
                "sample_seed": args.sample_seed,
                "gate": args.gate,
                "warmup_steps": args.warmup_steps,
                "learning_rate": args.learning_rate,
            },
        )

        bootstrap = await bootstrap_policy_version(
            training_client=training_client,
            renderer=renderer,
            tokenizer=tokenizer,
            row=train_rows[0],
        )
        append_jsonl(metrics_path, bootstrap)
        print(
            "POLICY_BOOTSTRAP PASS | "
            f"version={bootstrap['initial_policy_version']}"
            f"->{bootstrap['final_policy_version']} "
            f"grad_norm={bootstrap['grad_norm']:.1e}",
            flush=True,
        )

        preflight = await run_preflight(
            training_client=training_client,
            renderer=renderer,
            tokenizer=tokenizer,
            row=train_rows[0],
            sampling_params=sampling_params,
        )
        append_jsonl(metrics_path, preflight)
        print(
            "PREFLIGHT PASS | "
            f"policy={preflight['observed_policy_versions']} "
            f"tokens={preflight['tokens']} k3={preflight['k3']:.6f}",
            flush=True,
        )
        if args.preflight_only:
            print("PREFLIGHT_COMPLETE", flush=True)
            return

        before = (
            await evaluate(
                baseline_sampling_client,
                renderer,
                eval_rows,
                max_tokens=args.max_tokens,
                dataset=spec.key,
            )
            if eval_rows
            else float("nan")
        )
        append_jsonl(metrics_path, {"event": "eval_before", "pass1": before})
        print(f"BEFORE pass@1={before:.3f} n={len(eval_rows)}", flush=True)

        last_sampling_client = None
        gate_passes = 0
        for step in range(args.steps):
            step_start = time.perf_counter()
            batch = train_rows[step * args.batch_size : (step + 1) * args.batch_size]

            phase_start = time.perf_counter()
            last_sampling_client = (
                await training_client.save_weights_and_get_sampling_client_async(
                    name=f"step-{step}"
                )
            )
            required_version = training_client.policy_version
            save_weights_seconds = time.perf_counter() - phase_start

            prompts = [
                renderer.build_generation_prompt(make_conversation(row["question"]))
                for row in batch
            ]
            phase_start = time.perf_counter()
            results = await asyncio.gather(
                *[
                    last_sampling_client.sample_async(
                        prompt=prompt,
                        num_samples=args.group_size,
                        sampling_params=sampling_params,
                    )
                    for prompt in prompts
                ]
            )
            sampling_seconds = time.perf_counter() - phase_start
            policy_versions = verify_sampler_reload(results, required_version)

            completion_lengths = [
                len(sequence.tokens)
                for result in results
                for sequence in result.sequences
            ]
            completion_tokens = sum(completion_lengths)
            probe_sequences = [
                make_probe_sequence(
                    prompt,
                    sequence,
                    prompt_index=prompt_index,
                    sample_index=sample_index,
                )
                for prompt_index, (prompt, result) in enumerate(zip(prompts, results))
                for sample_index, sequence in enumerate(result.sequences)
                if sequence.tokens
            ]

            phase_start = time.perf_counter()
            parity = await kl_probe(
                training_client,
                probe_sequences,
                chunk_size=args.probe_chunk,
                capture_dir=capture_dir,
                step=step,
                tokenizer=tokenizer,
            )
            probe_seconds = time.perf_counter() - phase_start
            gated = step >= args.warmup_steps
            gate_pass = gated and parity["k3"] < args.gate
            gate_passes += int(gate_pass)

            datums: list[Datum] = []
            group_rewards: list[float] = []
            degenerate_groups = 0
            for prompt, result, row in zip(prompts, results, batch):
                rewards = [
                    grade(
                        renderer.decode(sequence.tokens),
                        row["answer"],
                        spec.key,
                    )
                    for sequence in result.sequences
                ]
                mean_reward = sum(rewards) / len(rewards)
                group_rewards.append(mean_reward)
                advantages = [reward - mean_reward for reward in rewards]
                if all(advantage == 0.0 for advantage in advantages):
                    degenerate_groups += 1
                    continue
                observation_length = prompt.length - 1
                for sequence, advantage in zip(result.sequences, advantages):
                    if not sequence.tokens:
                        continue
                    model_input = prompt.append(
                        EncodedTextChunk(tokens=sequence.tokens[:-1])
                    )
                    datums.append(
                        Datum(
                            model_input=model_input,
                            loss_fn_inputs={
                                "target_tokens": tensor(
                                    [0] * observation_length + sequence.tokens,
                                    "int64",
                                ),
                                "logprobs": tensor(
                                    [0.0] * observation_length + sequence.logprobs,
                                    "float32",
                                ),
                                "advantages": tensor(
                                    [0.0] * observation_length
                                    + [advantage]
                                    * (model_input.length - observation_length),
                                    "float32",
                                ),
                            },
                        )
                    )

            losses: list[float] = []
            forward_backward_seconds = 0.0
            optim_seconds = 0.0
            grad_norm = None
            if datums:
                micro_batch_size = args.micro_batch_size or len(datums)
                for start in range(0, len(datums), micro_batch_size):
                    phase_start = time.perf_counter()
                    future = await training_client.forward_backward_async(
                        datums[start : start + micro_batch_size],
                        loss_fn="importance_sampling",
                    )
                    result = await future.result_async()
                    forward_backward_seconds += time.perf_counter() - phase_start
                    loss = getattr(result, "loss", None)
                    if loss is None:
                        loss = result.metrics.get("loss", float("nan"))
                    losses.append(float(loss))

                phase_start = time.perf_counter()
                optim_future = await training_client.optim_step_async(adam_params)
                optim_result = await optim_future.result_async()
                optim_seconds = time.perf_counter() - phase_start
                grad_norm = (optim_result.metrics or {}).get("grad_norm")

            mean_reward = sum(group_rewards) / len(group_rewards)
            fraction_degenerate = degenerate_groups / len(group_rewards)
            step_seconds = time.perf_counter() - step_start
            record = {
                "event": "step",
                "step": step,
                **parity,
                "gate_pass": gate_pass,
                "gated": gated,
                "required_policy_version": required_version,
                "observed_policy_versions": policy_versions,
                "sampler_reload_verified": True,
                "reward": mean_reward,
                "frac_degenerate": fraction_degenerate,
                "n_datums": len(datums),
                "loss": (sum(losses) / len(losses) if losses else float("nan")),
                "grad_norm": grad_norm,
                "comp_tok_mean": (completion_tokens / len(completion_lengths)),
                "comp_tok_p95": percentile(completion_lengths, 0.95),
                "timings": {
                    "save_weights_s": save_weights_seconds,
                    "sampling_s": sampling_seconds,
                    "probe_s": probe_seconds,
                    "fb_s": forward_backward_seconds,
                    "optim_s": optim_seconds,
                    "step_s": step_seconds,
                },
            }
            append_jsonl(metrics_path, record)
            tails = parity["tail_counts"]
            print(
                f"step {step:02d}: k3={parity['k3']:.6f} "
                f"mean_abs={parity['mean_abs']:.6f} "
                f"max_abs={parity['max_abs']:.4f} "
                f"ESS/N={parity['ess_over_n']:.4f} "
                f"clip={parity['clip_fraction']:.4f} "
                f"tokens={parity['tokens']} "
                "tails="
                f"{tails['abs_r_gt_1']}/{tails['abs_r_gt_2']}/"
                f"{tails['abs_r_gt_5']}/{tails['abs_r_gt_10']} "
                f"gate={('PASS' if gate_pass else 'FAIL') if gated else 'EXCLUDED'} "
                f"reload={policy_versions} step_s={step_seconds:.0f}",
                flush=True,
            )

        if last_sampling_client is None:
            raise RuntimeError("no training steps completed")
        if args.long_probe_tokens > 0:
            await run_long_probe(
                training_client=training_client,
                tokenizer=tokenizer,
                renderer=renderer,
                dataset=dataset,
                spec=spec,
                args=args,
                metrics_path=metrics_path,
                capture_dir=capture_dir,
            )
        if eval_rows:
            final_sampling_client = (
                await training_client.save_weights_and_get_sampling_client_async(
                    name="final-eval"
                )
            )
            after = await evaluate(
                final_sampling_client,
                renderer,
                eval_rows,
                max_tokens=args.max_tokens,
                dataset=spec.key,
            )
        else:
            after = float("nan")
        append_jsonl(metrics_path, {"event": "eval_after", "pass1": after})
        append_jsonl(
            metrics_path,
            {
                "event": "run_complete",
                "steps_below_gate": gate_passes,
                "steps": args.steps - args.warmup_steps,
                "overall_gate_pass": gate_passes == args.steps - args.warmup_steps,
            },
        )
        print(
            f"RUN_COMPLETE gate={gate_passes}/{args.steps - args.warmup_steps} "
            f"pass1={before:.3f}->{after:.3f}",
            flush=True,
        )
    finally:
        await training_client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-url", default="http://127.0.0.1:8001")
    parser.add_argument("--sampler-url", required=True)
    parser.add_argument(
        "--model",
        default=(
            "/root/.cache/team_artifacts/huggingface/hub/"
            "models--zai-org--GLM-5.2-FP8/snapshots/"
            "70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1"
        ),
    )
    parser.add_argument("--run-id", default="glm52-cp32-klprobe")
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "math"), default="math")
    parser.add_argument("--math-levels", default="Level 4,Level 5")
    parser.add_argument("--disable-thinking", action="store_true")
    # Slim probe: 4 prompts x 8 completions = 32 active sequences, matching the
    # GLM golden sampler's max_num_seqs=32 without scheduler-level queuing.
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--data-seed", type=int, default=999)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--eval-problems", type=int, default=0)
    parser.add_argument("--probe-chunk", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--gate", type=float, default=0.015)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--long-probe-tokens", type=int, default=0)
    parser.add_argument("--long-probe-decode-tokens", type=int, default=15_000)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-existing-output", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
