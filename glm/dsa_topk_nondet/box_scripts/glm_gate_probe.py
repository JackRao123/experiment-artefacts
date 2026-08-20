#!/usr/bin/env python3
"""Replay the nightly SFT gate's trainer-determinism probe against a local trainer.

Arms (each: N repeated /forward calls over the same six CE datums):
  padded-tiled    gate-like: tiled pangram tokens, doc lens NOT 8-aligned
  aligned-tiled   same content recipe, doc lens 8-aligned (padding-free pack)
  padded-random   random token ids, doc lens NOT 8-aligned

Reports per-datum max per-position logprob spread across repeats, per arm.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid

import httpx

PATTERN_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
)

PADDED_LENS = [2999, 1499, 1249, 2499, 1999, 499]
ALIGNED_LENS = [3000, 1496, 1248, 2496, 2000, 496]
# Big arms: lens ≡ 1 (mod 8) → 7 pad rows per doc at CP4; all ≫ index_topk=2048.
PADDED_LENS_BIG = [11993, 5993, 4993, 9993, 7993, 1993]
ALIGNED_LENS_BIG = [12000, 6000, 5000, 10000, 8000, 2000]


def submit_and_wait(client: httpx.Client, op_path: str, body: dict, timeout=900.0):
    r = client.post(op_path, json=body, headers={"Idempotency-Key": uuid.uuid4().hex})
    assert r.status_code == 202, f"{op_path}: {r.status_code} {r.text[:500]}"
    op = r.json()["operation_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rr = client.get(f"/operations/{op}", timeout=35.0)
        if rr.status_code == 408:
            continue
        rr.raise_for_status()
        b = rr.json()
        if b.get("status") == "done":
            return b["result"]
        if b.get("status") == "error":
            raise RuntimeError(f"{op_path} failed: {b.get('error')}")
        time.sleep(0.5)
    raise TimeoutError(op_path)


def ce_datum(tokens: list[int]) -> dict:
    n = len(tokens)
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
        "loss_fn_inputs": {
            "target_tokens": {
                "data": tokens[1:] + [-100],
                "dtype": "int64",
                "shape": [n],
            },
            "weights": {
                "data": [1.0] * (n - 1) + [0.0],
                "dtype": "float32",
                "shape": [n],
            },
        },
    }


def cycle(pattern: list[int], n: int) -> list[int]:
    reps = n // len(pattern) + 1
    return (pattern * reps)[:n]


def logprobs_of(result: dict) -> list[list[float]]:
    outs = result["loss_fn_outputs"]
    return [
        [float("nan") if v is None else float(v) for v in o["logprobs"]["data"]]
        for o in outs
    ]


def spread(runs: list[list[list[float]]]) -> list[float]:
    per_datum = []
    for d in range(len(runs[0])):
        worst = 0.0
        cols = zip(*(r[d] for r in runs))
        for col in cols:
            worst = max(worst, max(col) - min(col))
        per_datum.append(worst)
    return per_datum


def run_arm(client, name, datums, repeats):
    runs = []
    for i in range(repeats):
        res = submit_and_wait(
            client, "/forward", {"data": datums, "loss_fn": "cross_entropy"}
        )
        runs.append(logprobs_of(res))
        print(f"  {name} forward {i + 1}/{repeats}: {[len(x) for x in runs[-1]]}", flush=True)
    sp = spread(runs)
    bitwise = all(runs[i] == runs[0] for i in range(1, repeats))
    print(
        json.dumps(
            {
                "arm": name,
                "per_datum_max_spread": [round(s, 6) for s in sp],
                "max_spread": max(sp),
                "bitwise_identical": bitwise,
            }
        ),
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--snapshot", default="/root/glm52_smoke")
    ap.add_argument(
        "--arms", default="padded-tiled,aligned-tiled,padded-random"
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.snapshot, trust_remote_code=True)
    pattern = tok.encode(PATTERN_TEXT, add_special_tokens=False)
    print(f"pattern tokens: {len(pattern)}", flush=True)
    vocab = tok.vocab_size

    import random

    rng = random.Random(20260819)

    client = httpx.Client(base_url=args.url, timeout=60.0)
    st = client.get("/status").json()
    print(f"status: {json.dumps({k: st.get(k) for k in ('max_seq_len', 'world_size', 'data_parallel_size')})}", flush=True)

    arms = args.arms.split(",")
    for arm in arms:
        if arm == "padded-tiled":
            datums = [ce_datum(cycle(pattern, n)) for n in PADDED_LENS]
        elif arm == "aligned-tiled":
            datums = [ce_datum(cycle(pattern, n)) for n in ALIGNED_LENS]
        elif arm == "padded-random":
            datums = [
                ce_datum([rng.randrange(100, vocab - 100) for _ in range(n)])
                for n in PADDED_LENS
            ]
        elif arm == "padded-tiled-big":
            datums = [ce_datum(cycle(pattern, n)) for n in PADDED_LENS_BIG]
        elif arm == "aligned-tiled-big":
            datums = [ce_datum(cycle(pattern, n)) for n in ALIGNED_LENS_BIG]
        elif arm == "padded-random-big":
            datums = [
                ce_datum([rng.randrange(100, vocab - 100) for _ in range(n)])
                for n in PADDED_LENS_BIG
            ]
        else:
            raise ValueError(arm)
        run_arm(client, arm, datums, args.repeats)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
