#!/usr/bin/env python3
"""LPS-1063 forward-determinism driver, 4-node B300 (TP8/CP4/EP32).

Runs NO save/load (sidesteps the phase-4 load deadlock). Sequence:

  fwd_pre  (adapters at init)
  forward_backward -> optim_step(Adam 2e-4)     [unless --skip-train]
  fwd[0..N-1]   N back-to-back forwards on the IDENTICAL datum0

Every arm uses the same fixed datum construction as the nightly gate and
the phase-1..4 roundtrip driver (hendrycks_math shuffle(seed=16) rows 4..8,
587-token probe, 8-datum mixed group).

Verdict: bitwise-deterministic iff all N logprob vectors are identical.
Evidence JSON carries all vectors + pairwise delta stats + distinct-vector
grouping, tagged with --label (control / overlap_off / envtrio / ...).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time

from baseten.loops import (
    AdamParams,
    Datum,
    ModelInput,
    TensorData,
    TrainingClient,
)
from baseten.loops.models import EncodedTextChunk
from datasets import concatenate_datasets, load_dataset

DEFAULT_MODEL = (
    "/root/.cache/team_artifacts/huggingface/hub/"
    "models--baseten--NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4-dequant-to-BF16/"
    "snapshots/c1a0675dcd27b7f05cb0e280bdd425c51b65395a"
)
TRAINER_URL = "http://localhost:8001"
LORA_RANK = 16
MAX_SEQ_LEN = 262144
LR = 2e-4
DATUM0_TOKENS = 587
OTHER_TOKENS = 600

SYSTEM = (
    "You are a math tutor. Reason step by step and give the final answer "
    "in \\boxed{...}."
)
SUFFIX = "\n\nReason step by step and put your final answer in \\boxed{...}."

_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")

FILLER = [
    "We begin by reading the problem carefully and identifying what is asked.",
    "Let us introduce notation for the quantities involved and write down the givens.",
    "A natural first step is to simplify the expression using the standard rules.",
    "We check the edge cases before committing to the main computation.",
    "The key observation is that the structure repeats with a fixed period.",
    "Substituting the known values reduces the problem to a routine calculation.",
    "We verify each algebraic manipulation before moving on to the next one.",
    "An alternative approach confirms the same intermediate result.",
    "The constraints narrow the candidate answers down to a small set.",
    "Direct computation now settles the remaining cases one by one.",
    "We keep track of the signs carefully to avoid an off-by-one error.",
    "Putting the pieces together gives a single consistent value.",
]


def last_boxed(text: str) -> str | None:
    m = _BOXED_RE.findall(text)
    return m[-1].strip() if m else None


def wrong_answer(gt: str) -> str:
    if re.fullmatch(r"-?\d+", gt):
        return str(int(gt) + 1)
    if re.fullmatch(r"-?\d+/\d+", gt):
        n, d = gt.split("/")
        return f"{int(n) + 1}/{d}"
    return gt + "1"


def _tensor(data: list, dtype: str) -> TensorData:
    import torch

    torch_dtype = torch.float32 if dtype == "float32" else torch.int64
    return TensorData.from_torch(torch.tensor(data, dtype=torch_dtype))


def _render_chat_ids(tokenizer, convo: list[dict]) -> list[int]:
    ids = tokenizer.apply_chat_template(
        convo,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(ids, "get") and "input_ids" in ids:
        ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
    return list(ids)


def build_prompt(tokenizer, question: str) -> ModelInput:
    convo = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question + SUFFIX},
    ]
    return ModelInput.from_ints(_render_chat_ids(tokenizer, convo))


def _completion_datum(
    prompt: ModelInput,
    tokens: list[int],
    logprobs: list[float],
    advantages: list[float],
) -> Datum:
    ob_len = prompt.length - 1
    return Datum(
        model_input=prompt.append(EncodedTextChunk(tokens=list(tokens[:-1]))),
        loss_fn_inputs={
            "target_tokens": _tensor([0] * ob_len + list(tokens), "int64"),
            "logprobs": _tensor([0.0] * ob_len + list(logprobs), "float32"),
            "advantages": _tensor([0.0] * ob_len + list(advantages), "float32"),
        },
    )


def completion_ids(tokenizer, answer: str, target_len: int, seed_idx: int) -> list[int]:
    suffix = f"\nThe final answer is $\\boxed{{{answer}}}$."
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    body_budget = target_len - len(suffix_ids)
    assert body_budget > 0, "target length too small for answer suffix"
    ids: list[int] = []
    i = seed_idx
    while len(ids) < body_budget:
        sentence = FILLER[i % len(FILLER)] + " "
        ids.extend(tokenizer(sentence, add_special_tokens=False)["input_ids"])
        i += 1
    ids = ids[:body_budget]
    return ids + suffix_ids


def load_step1_batch():
    subjects = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    parts = [
        load_dataset("EleutherAI/hendrycks_math", s, split="train") for s in subjects
    ]
    data = concatenate_datasets(parts)
    data = data.filter(lambda r: r["level"] in {"Level 4", "Level 5"})
    data = data.shuffle(seed=16)
    return data.select(range(4, 8))


def vec_delta(a: list[float], b: list[float]) -> dict:
    assert len(a) == len(b), f"length mismatch {len(a)} vs {len(b)}"
    diffs = [abs(x - y) for x, y in zip(a, b)]
    n_diff = sum(1 for d in diffs if d > 0)
    return {
        "max": max(diffs),
        "mean": sum(diffs) / len(diffs),
        "n_nonzero": n_diff,
        "n": len(diffs),
        "argmax": max(range(len(diffs)), key=lambda i: diffs[i]),
    }


async def fwd(tc, datum, label: str) -> list[float]:
    t0 = time.time()
    r = await tc.forward([datum], loss_fn="importance_sampling")
    assert r.loss_fn_outputs, f"{label}: forward returned no loss_fn_outputs"
    lp = list(r.loss_fn_outputs[0]["logprobs"].data)
    print(f"{label}: {len(lp)} logprobs in {time.time()-t0:.0f}s, head={lp[:3]}", flush=True)
    return lp


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer-url", default=TRAINER_URL)
    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--n-fwd", type=int, default=10)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--label", required=True,
                    help="arm tag: control / overlap_off / envtrio / ...")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    rows = load_step1_batch()
    row = rows[0]
    gt = last_boxed(row["solution"])
    assert gt, f"no boxed ground truth in solution: {row['solution'][:200]}"
    print(f"mixed group: {row['level']} problem: {row['problem'][:90]!r}")

    prompt = build_prompt(tokenizer, row["problem"])
    print(f"prompt tokens: {prompt.length}")

    datums: list[Datum] = []
    for j in range(8):
        correct = j < 7
        target = DATUM0_TOKENS if j == 0 else OTHER_TOKENS
        answer = gt if correct else wrong_answer(gt)
        toks = completion_ids(tokenizer, answer, target, seed_idx=j)
        adv = 0.125 if correct else -0.875
        datums.append(
            _completion_datum(prompt, toks, [-0.3] * len(toks), [adv] * len(toks))
        )
    datum0 = datums[0]

    tc = TrainingClient(
        args.trainer_url,
        lora_rank=LORA_RANK,
        base_model=args.model_path,
        max_seq_len=args.max_seq_len,
        timeout=1800.0,
        ready_timeout=3600.0,
        run_id=f"nemotron-fwd-determinism-{args.label}",
    )
    print(f"policy_version at start: {tc.policy_version}", flush=True)

    results: dict = {"config": vars(args), "label": args.label, "phases": {}}

    lp_pre = await fwd(tc, datum0, "fwd_pre")

    if not args.skip_train:
        t0 = time.time()
        fb = await tc.forward_backward(datums, loss_fn="importance_sampling")
        print(f"forward_backward {time.time()-t0:.0f}s; metrics: {getattr(fb, 'metrics', None)}", flush=True)
        opt = await tc.optim_step(AdamParams(learning_rate=LR))
        print(f"optim_step; metrics: {getattr(opt, 'metrics', None)}", flush=True)

    vecs: list[list[float]] = []
    for i in range(args.n_fwd):
        vecs.append(await fwd(tc, datum0, f"fwd{i}"))

    d_train = vec_delta(vecs[0], lp_pre)
    results["phases"]["training_effect_fwd0_vs_fwd_pre"] = d_train
    print(f"TRAINING EFFECT fwd0 vs fwd_pre: {d_train}", flush=True)

    # vs-first deltas + full pairwise max
    vs_first = [vec_delta(vecs[i], vecs[0]) for i in range(1, len(vecs))]
    results["phases"]["vs_first"] = vs_first
    for i, d in enumerate(vs_first, start=1):
        print(f"fwd{i} vs fwd0: max={d['max']:.6g} mean={d['mean']:.6g} "
              f"nonzero={d['n_nonzero']}/{d['n']}", flush=True)

    pair_max = 0.0
    worst_pair = None
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            m = max(abs(x - y) for x, y in zip(vecs[i], vecs[j]))
            if m > pair_max:
                pair_max, worst_pair = m, (i, j)
    results["phases"]["pairwise_max"] = {"max": pair_max, "pair": worst_pair}

    # distinct-vector grouping (bitwise)
    groups: dict[tuple, list[int]] = {}
    for i, v in enumerate(vecs):
        groups.setdefault(tuple(v), []).append(i)
    group_list = sorted(groups.values(), key=len, reverse=True)
    results["phases"]["distinct_groups"] = group_list
    print(f"distinct bitwise outcomes: {len(group_list)} groups: {group_list}", flush=True)

    results["logprobs"] = {"fwd_pre": lp_pre,
                           **{f"fwd{i}": v for i, v in enumerate(vecs)}}

    deterministic = len(group_list) == 1
    verdict = ("BITWISE_DETERMINISTIC" if deterministic
               else f"NONDETERMINISTIC max|Δ|={pair_max:.6g} over {len(vecs)} fwds")
    results["verdict"] = verdict
    print(f"\nVERDICT [{args.label}]: {verdict}", flush=True)

    out = os.path.join(args.out_dir, f"fwd_determinism_{args.label}.json")
    with open(out, "w") as f:
        json.dump(results, f)
    print(f"evidence written: {out}", flush=True)
    return 0 if deterministic else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
