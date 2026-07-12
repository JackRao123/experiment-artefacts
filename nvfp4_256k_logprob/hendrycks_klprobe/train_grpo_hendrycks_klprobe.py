#!/usr/bin/env python3
"""20-step GRPO on Hendrycks MATH with a per-step trainer-vs-sampler KL probe.

The prime-rl-style gate table experiment (see profiling.md, "Hendrycks 20-step
GRPO KL benchmark"): two arms differing only in the trainer's base weights
(native BF16 vs NVFP4-dequant-bf16), same NVFP4 sampler. Each step, BEFORE the
optimizer update, every sampled completion is teacher-forced through the
trainer's forward (cross_entropy) and compared token-by-token against the
sampler's generation-time logprobs — k3 estimates KL(behavior‖target), same
conventions as experiment_client.py in this directory.

Step order (policy parity is load-bearing):
  save_weights(step-N) -> sampler reloads LoRA -> sample 64x8 -> KL probe
  (trainer /forward on ALL completions) -> build non-degenerate datums ->
  forward_backward chunks -> optim_step
so the trainer weights scored by the probe are exactly the weights the
sampler served. The probe scores all completions (not just surviving datums)
to avoid arm-dependent selection confounds.

Run (on the leader, examples venv):
  .venv/bin/python -u train_grpo_hendrycks_klprobe.py \
      --arm A --trainer-url http://127.0.0.1:8000 --sampler-url http://<node4>:8000 \
      --model nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
      --metrics-out /root/.cache/user_artifacts/rl_klprobe/logs/metrics_armA.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
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

# ── prompt ────────────────────────────────────────────────────────────

SYSTEM_MSG = (
    "You are a math competition expert. Solve the problem step by step and "
    "give the final answer in \\boxed{...}."
)
QUESTION_SUFFIX = (
    "\n\nReason step by step and put your final answer in \\boxed{...}."
)


class Renderer:
    """Chat-template renderer with thinking disabled (Nemotron supports the
    enable_thinking kwarg; fall back silently for tokenizers that don't)."""

    def __init__(self, tokenizer):
        self._tok = tokenizer

    def build_generation_prompt(self, convo: list[dict]) -> ModelInput:
        kwargs = dict(tokenize=True, add_generation_prompt=True)
        try:
            ids = self._tok.apply_chat_template(
                convo, enable_thinking=False, **kwargs
            )
        except TypeError:
            ids = self._tok.apply_chat_template(convo, **kwargs)
        if hasattr(ids, "get") and "input_ids" in ids:
            ids = ids["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
        return ModelInput.from_ints(list(ids))

    def decode(self, tokens: list[int]) -> str:
        return self._tok.decode(tokens, skip_special_tokens=True)


def make_convo(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": question + QUESTION_SUFFIX},
    ]


# ── grading (math_verify handles LaTeX equivalence) ──────────────────


def grade(response_text: str, gt: str) -> float:
    from math_verify import parse, verify

    try:
        gold = parse(gt)
        pred = parse(response_text)
        return 1.0 if verify(gold, pred) else 0.0
    except Exception:
        return 0.0


# ── KL metrics (same conventions as experiment_client.py) ────────────


def kl_metrics(
    behavior: list[float], target: list[float], clip_low=0.8, clip_high=1.2
) -> dict:
    xs: list[float] = []
    ys: list[float] = []
    dropped = 0
    for x, y in zip(behavior, target):
        if x is None or y is None:
            dropped += 1
            continue
        x, y = float(x), float(y)
        if math.isnan(x) or math.isnan(y):
            dropped += 1
            continue
        xs.append(x)
        ys.append(y)
    n = len(xs)
    if n == 0:
        return {"tokens": 0, "dropped": dropped}
    r = [t - b for b, t in zip(xs, ys)]  # log(target/behavior)
    w = [math.exp(max(-50.0, min(50.0, ri))) for ri in r]
    absd = [abs(ri) for ri in r]
    sw = sum(w)
    sw2 = sum(v * v for v in w)
    return {
        "tokens": n,
        "dropped": dropped,
        "k3": sum(math.exp(-ri) + ri - 1 for ri in r) / n,
        "mean_abs": sum(absd) / n,
        "rms": math.sqrt(sum(d * d for d in absd) / n),
        "max_abs": max(absd),
        "ess_over_n": ((sw * sw) / sw2) / n if sw2 else 0.0,
        "clip_fraction": sum(1 for v in w if v < clip_low or v > clip_high) / n,
        "mean_log_ratio": sum(r) / n,
    }


def merge_token_streams(pairs: list[tuple[list[float], list[float]]]) -> dict:
    """Pool (behavior, target) logprob lists from many sequences into one
    token-level metric dict."""
    behavior: list[float] = []
    target: list[float] = []
    for b, t in pairs:
        behavior.extend(b)
        target.extend(t)
    return kl_metrics(behavior, target)


def percentile(vals: list[int], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return float(s[k])


# ── helpers ───────────────────────────────────────────────────────────


def _tensor(data: list, dtype: str) -> TensorData:
    import torch

    torch_dtype = torch.float32 if dtype == "float32" else torch.int64
    return TensorData.from_torch(torch.tensor(data, dtype=torch_dtype))


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ── KL probe ─────────────────────────────────────────────────────────


OUTLIER_ABS_R = 5.0  # |log ratio| above this gets a full per-token record


async def kl_probe(
    training_client: TrainingClient,
    sequences: list[dict],
    *,
    chunk: int,
    capture: dict | None = None,
) -> dict:
    """Teacher-force every sampled sequence through the trainer's forward
    (cross_entropy, no backward) and compare against generation logprobs.

    sequences: [{prompt_ids, tokens, logprobs}]  (all completions, graded or not)
    Wire format: wire[k] = logprob(full[k+1] | full[0..k]) with a 0.0 sentinel
    at the last slot => completion region is wire[plen-1 : plen-1+clen].

    capture (optional): {"dir": Path, "step": int, "tokenizer": tok} — dumps
    the full per-token (behavior, target) arrays for the step as a compressed
    npz, writes an outliers JSONL (every token with |r| > OUTLIER_ABS_R,
    signed r, token id, decoded context), and adds tail counts to the
    returned metrics.
    """
    pairs: list[tuple[list[float], list[float]]] = []
    for i in range(0, len(sequences), chunk):
        batch = sequences[i : i + chunk]
        datums = []
        for s in batch:
            full = s["prompt_ids"] + s["tokens"]
            datums.append(
                Datum(
                    model_input=ModelInput.from_ints(full),
                    loss_fn_inputs={},
                )
            )
        fut = await training_client.forward_async(datums, loss_fn="cross_entropy")
        result = await fut.result_async()
        for s, out in zip(batch, result.loss_fn_outputs):
            wire = out["logprobs"].tolist()
            plen = len(s["prompt_ids"])
            clen = len(s["tokens"])
            target = wire[plen - 1 : plen - 1 + clen]
            target = [float("nan") if v is None else float(v) for v in target]
            pairs.append((s["logprobs"], target))
    metrics = merge_token_streams(pairs)

    if capture is not None:
        import numpy as np

        cap_dir = Path(capture["dir"])
        cap_dir.mkdir(parents=True, exist_ok=True)
        step = capture["step"]
        tok = capture["tokenizer"]

        behavior = np.array(
            [x for b, _ in pairs for x in b], dtype=np.float32
        )
        target = np.array(
            [np.nan if x is None else x for _, t in pairs for x in t],
            dtype=np.float32,
        )
        seq_lens = np.array([len(b) for b, _ in pairs], dtype=np.int32)
        np.savez_compressed(
            cap_dir / f"step{step:02d}_logprobs.npz",
            behavior=behavior,
            target=target,
            seq_lens=seq_lens,
        )

        finite = np.isfinite(behavior) & np.isfinite(target)
        r = np.where(finite, target - behavior, 0.0)
        absr = np.abs(r)
        metrics["tail_counts"] = {
            f"abs_r_gt_{th}": int((absr > th).sum()) for th in (1, 2, 5, 10)
        }
        metrics["finite_tokens"] = int(finite.sum())

        # Per-token outlier records with decoded context.
        outliers = []
        flat_idx = 0
        for seq_i, (b, t) in enumerate(pairs):
            s = sequences[seq_i]
            for pos in range(len(b)):
                gi = flat_idx + pos
                if not finite[gi] or absr[gi] <= OUTLIER_ABS_R:
                    continue
                lo = max(0, pos - 30)
                ctx_ids = s["tokens"][lo : pos + 1]
                outliers.append(
                    {
                        "step": step,
                        "seq": seq_i,
                        "pos": pos,
                        "completion_len": len(b),
                        "token_id": s["tokens"][pos],
                        "token_text": tok.decode([s["tokens"][pos]]),
                        "behavior_lp": float(b[pos]),
                        "target_lp": float(t[pos]),
                        "r": float(r[gi]),
                        "context_text": tok.decode(ctx_ids),
                    }
                )
            flat_idx += len(b)
        for rec in outliers:
            append_jsonl(cap_dir / "outliers.jsonl", rec)
        metrics["n_outlier_records"] = len(outliers)

    return metrics


# ── eval (held-out greedy pass@1) ────────────────────────────────────


async def eval_pass_rate(
    sampling_client, renderer: Renderer, rows: list[dict], *, max_tokens: int
) -> float:
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0)
    coros = [
        sampling_client.sample_async(
            prompt=renderer.build_generation_prompt(make_convo(r["question"])),
            num_samples=1,
            sampling_params=params,
        )
        for r in rows
    ]
    results = await asyncio.gather(*coros)
    correct = sum(
        1
        for res, r in zip(results, rows)
        if grade(renderer.decode(res.sequences[0].tokens), r["answer"]) > 0
    )
    return correct / len(rows)


# ── main loop ─────────────────────────────────────────────────────────


async def run(args) -> None:
    metrics_path = Path(args.metrics_out)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    renderer = Renderer(tokenizer)

    from datasets import load_dataset

    ds = load_dataset("PrimeIntellect/Hendrycks-Math", "default", split="train")
    ds = ds.shuffle(seed=args.data_seed)
    n_train = args.steps * args.batch_size
    train_rows = [
        {"question": ds[i]["question"], "answer": ds[i]["answer"]}
        for i in range(n_train)
    ]
    eval_rows = [
        {"question": ds[i]["question"], "answer": ds[i]["answer"]}
        for i in range(n_train, n_train + args.eval_problems)
    ]
    print(
        f"[{args.arm}] dataset ready: {len(train_rows)} train, "
        f"{len(eval_rows)} eval (shuffle seed {args.data_seed})",
        flush=True,
    )

    dep = LocalDeployment(base_url=args.sampler_url)
    training_client = TrainingClient(
        args.trainer_url,
        lora_rank=args.rank,
        base_model=args.model,
        timeout=1800.0,
        ready_timeout=3600.0,
        run_id=f"klprobe-arm{args.arm}",
        paired_sampling_deployment=dep,
    )
    baseline_sampling_client = SamplingClient(base_model=args.model, deployment=dep)

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens, temperature=1.0, top_p=1.0, seed=args.sample_seed
    )
    adam_params = AdamParams(learning_rate=args.learning_rate)

    if eval_rows:
        t0 = time.perf_counter()
        before = await eval_pass_rate(
            baseline_sampling_client, renderer, eval_rows, max_tokens=args.max_tokens
        )
        print(
            f"[{args.arm}] BEFORE pass@1: {before:.3f} "
            f"(n={len(eval_rows)}, {time.perf_counter() - t0:.0f}s)",
            flush=True,
        )
        append_jsonl(metrics_path, {"arm": args.arm, "event": "eval_before", "pass1": before})

    sampling_client = None
    for step in range(args.steps):
        step_t0 = time.perf_counter()
        batch = train_rows[step * args.batch_size : (step + 1) * args.batch_size]

        # 1. publish weights, sampler reloads -> target == behavior policy
        t0 = time.perf_counter()
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(
            name=f"step-{step}"
        )
        save_weights_s = time.perf_counter() - t0
        print(f"Step {step:2d} | phase=save_weights {save_weights_s:.1f}s", flush=True)

        # 2. sample 64 problems x group 8
        t0 = time.perf_counter()
        prompts = [renderer.build_generation_prompt(make_convo(r["question"])) for r in batch]
        results = await asyncio.gather(
            *[
                sampling_client.sample_async(
                    prompt=p, num_samples=args.group_size, sampling_params=sampling_params
                )
                for p in prompts
            ]
        )
        sampling_s = time.perf_counter() - t0
        comp_lens = [len(seq.tokens) for r in results for seq in r.sequences]
        n_comp_tokens = sum(comp_lens)
        print(
            f"Step {step:2d} | phase=sample {sampling_s:.1f}s | "
            f"{n_comp_tokens} tok = {n_comp_tokens / max(sampling_s, 1e-9):.0f} tok/s | "
            f"comp mean/p95: {n_comp_tokens / len(comp_lens):.0f}/{percentile(comp_lens, 0.95):.0f}",
            flush=True,
        )

        # 3. KL probe on ALL completions (before any grad/optim this step)
        t0 = time.perf_counter()
        probe_seqs = []
        for prompt, res in zip(prompts, results):
            p_ids = prompt.to_ints()
            for seq in res.sequences:
                if len(seq.tokens) == 0:
                    continue
                probe_seqs.append(
                    {"prompt_ids": p_ids, "tokens": seq.tokens, "logprobs": seq.logprobs}
                )
        capture = (
            {"dir": args.capture_dir, "step": step, "tokenizer": tokenizer}
            if args.capture_dir
            else None
        )
        klm = await kl_probe(
            training_client, probe_seqs, chunk=args.probe_chunk, capture=capture
        )
        probe_s = time.perf_counter() - t0
        gate = "PASS" if klm.get("k3", float("nan")) < 0.015 else "FAIL"
        print(
            f"KL step={step} k3={klm.get('k3', float('nan')):.6f} "
            f"mean_abs={klm.get('mean_abs', float('nan')):.6f} "
            f"max_abs={klm.get('max_abs', float('nan')):.4f} "
            f"ess={klm.get('ess_over_n', float('nan')):.4f} "
            f"clip={klm.get('clip_fraction', float('nan')):.4f} "
            f"tokens={klm.get('tokens', 0)} gate_0.015={gate} ({probe_s:.1f}s)",
            flush=True,
        )

        # 4. grade + advantages + datums (skip degenerate groups)
        datums: list[Datum] = []
        group_rewards: list[float] = []
        n_degenerate = 0
        for prompt, res, row in zip(prompts, results, batch):
            rewards = [
                grade(renderer.decode(seq.tokens), row["answer"]) for seq in res.sequences
            ]
            mean_r = sum(rewards) / len(rewards)
            group_rewards.append(mean_r)
            advantages = [r - mean_r for r in rewards]
            if all(a == 0.0 for a in advantages):
                n_degenerate += 1
                continue
            ob_len = prompt.length - 1
            for seq, adv in zip(res.sequences, advantages):
                if len(seq.tokens) == 0:
                    continue
                model_input = prompt.append(EncodedTextChunk(tokens=seq.tokens[:-1]))
                datums.append(
                    Datum(
                        model_input=model_input,
                        loss_fn_inputs={
                            "target_tokens": _tensor([0] * ob_len + seq.tokens, "int64"),
                            "logprobs": _tensor([0.0] * ob_len + seq.logprobs, "float32"),
                            "advantages": _tensor(
                                [0.0] * ob_len + [adv] * (model_input.length - ob_len),
                                "float32",
                            ),
                        },
                    )
                )

        # 5. fb + optim
        loss = float("nan")
        fb_s = 0.0
        optim_s = 0.0
        if datums:
            losses = []
            chunk = args.micro_batch_size or len(datums)
            for i in range(0, len(datums), chunk):
                t0 = time.perf_counter()
                fut = await training_client.forward_backward_async(
                    datums[i : i + chunk], loss_fn="importance_sampling"
                )
                fb_result = await fut.result_async()
                fb_s += time.perf_counter() - t0
                step_loss = getattr(fb_result, "loss", None)
                if step_loss is None:
                    step_loss = float(fb_result.metrics.get("loss", float("nan")))
                losses.append(float(step_loss))
            t0 = time.perf_counter()
            await training_client.optim_step_async(adam_params)
            optim_s = time.perf_counter() - t0
            loss = sum(losses) / len(losses)

        mean_reward = sum(group_rewards) / len(group_rewards)
        frac_degenerate = n_degenerate / len(group_rewards)
        step_s = time.perf_counter() - step_t0
        print(
            f"Step {step:2d} | reward: {mean_reward:.3f} | loss: {loss:.4f} | "
            f"degenerate: {frac_degenerate:.0%} | datums: {len(datums)} | "
            f"fb: {fb_s:.1f}s | optim: {optim_s:.2f}s | step: {step_s:.0f}s",
            flush=True,
        )
        append_jsonl(
            metrics_path,
            {
                "arm": args.arm,
                "event": "step",
                "step": step,
                **{k: klm.get(k) for k in (
                    "k3", "mean_abs", "rms", "max_abs", "ess_over_n",
                    "clip_fraction", "mean_log_ratio", "tokens", "dropped",
                    "tail_counts", "n_outlier_records",
                )},
                "gate_pass": gate == "PASS",
                "reward": mean_reward,
                "frac_degenerate": frac_degenerate,
                "n_datums": len(datums),
                "loss": loss,
                "comp_tok_mean": n_comp_tokens / len(comp_lens),
                "comp_tok_p95": percentile(comp_lens, 0.95),
                "timings": {
                    "save_weights_s": save_weights_s,
                    "sampling_s": sampling_s,
                    "probe_s": probe_s,
                    "fb_s": fb_s,
                    "optim_s": optim_s,
                    "step_s": step_s,
                },
            },
        )

    if eval_rows and sampling_client is not None:
        after = await eval_pass_rate(
            sampling_client, renderer, eval_rows, max_tokens=args.max_tokens
        )
        print(f"[{args.arm}] AFTER pass@1: {after:.3f}", flush=True)
        append_jsonl(metrics_path, {"arm": args.arm, "event": "eval_after", "pass1": after})

    print(f"[{args.arm}] RUN_COMPLETE", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=["A", "B"])
    p.add_argument("--trainer-url", required=True)
    p.add_argument("--sampler-url", required=True)
    p.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16")
    p.add_argument("--metrics-out", required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=4e-5)
    p.add_argument("--data-seed", type=int, default=999)
    p.add_argument("--sample-seed", type=int, default=1234)
    p.add_argument("--eval-problems", type=int, default=64)
    p.add_argument("--probe-chunk", type=int, default=64)
    p.add_argument("--micro-batch-size", type=int, default=32)
    p.add_argument(
        "--capture-dir",
        default=None,
        help="Directory for per-token logprob capture: per-step npz of "
        "(behavior, target) arrays + outliers.jsonl with decoded context "
        "for every token with |log ratio| > 5.",
    )
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
