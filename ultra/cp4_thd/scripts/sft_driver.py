"""Deterministic SFT/RL driver for a running dp_worker trainer.

Builds a seeded multi-document payload, runs forward_backward (+ optional
optim_step) cycles over the trainer HTTP API, and writes one JSON result with
loss, token counts, grad norm, per-datum logprobs, timings, and all-rank
allocator stats. Used for the tiny hybrid THD+CP probe, the Ultra CP1-vs-CP4
parity comparison, and the CP4/EP32 sequence-length profile.

    uv run --no-sync python sft_driver.py \
        --url http://127.0.0.1:8001 \
        --doc-lens 1500,900,2000 \
        --steps 2 --lr 0.0 \
        --out /root/.cache/user_artifacts/ultra_cp4/out/probe_cp2.json

The payload depends only on (--seed, --doc-lens, --vocab, --loss-fn), so two
trainers driven with identical flags see byte-identical requests.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import httpx


def build_payload(
    *, seed: int, doc_lens: list[int], vocab: int, loss_fn: str
) -> dict[str, Any]:
    rng = random.Random(seed)
    data = []
    for length in doc_lens:
        tokens = [rng.randrange(10, vocab) for _ in range(length)]
        targets = tokens[1:] + [-100]
        # Mask the first ~10% of positions like a prompt/completion split.
        prompt = max(1, length // 10)
        weights = [0.0] * prompt + [1.0] * (length - prompt)
        loss_fn_inputs = {
            "target_tokens": {"data": targets, "dtype": "int64", "shape": [length]},
            "weights": {"data": weights, "dtype": "float32", "shape": [length]},
        }
        if loss_fn != "cross_entropy":
            # Token-level RL fields: plausible logprobs, +/-1 advantages, T=1.
            loss_fn_inputs["logprobs"] = {
                "data": [-rng.random() * 2.0 for _ in range(length)],
                "dtype": "float32",
                "shape": [length],
            }
            loss_fn_inputs["advantages"] = {
                "data": [float(rng.choice((-1, 0, 1))) for _ in range(length)],
                "dtype": "float32",
                "shape": [length],
            }
            loss_fn_inputs["temperatures"] = {
                "data": [1.0] * length,
                "dtype": "float32",
                "shape": [length],
            }
        data.append(
            {
                "model_input": {
                    "chunks": [{"type": "encoded_text", "tokens": tokens}]
                },
                "loss_fn_inputs": loss_fn_inputs,
            }
        )
    return {"data": data, "loss_fn": loss_fn}


def submit_and_wait(
    client: httpx.Client, url: str, path: str, body: dict, timeout_s: float
) -> dict:
    r = client.post(f"{url}{path}", json=body)
    r.raise_for_status()
    oid = r.json()["operation_id"]
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        r = client.get(f"{url}/operations/{oid}")
        if r.status_code == 408:
            continue
        r.raise_for_status()
        payload = r.json()
        if payload.get("status") == "done":
            return payload["result"]
        if payload.get("status") == "error":
            raise RuntimeError(f"{path} failed: {json.dumps(payload)[:2000]}")
    raise TimeoutError(f"{path} did not finish within {timeout_s}s")


def memory_stats(client: httpx.Client, url: str, *, reset_peaks: bool) -> dict | None:
    try:
        r = client.post(f"{url}/memory_stats", json={"reset_peaks": reset_peaks})
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as exc:  # older images without the endpoint
        return {"error": str(exc)}


def summarize_memory(stats: dict | None) -> dict | None:
    if not stats or "ranks" not in stats or not stats["ranks"]:
        return None
    ranks = stats["ranks"]
    hot = max(ranks, key=lambda r: r["max_allocated_bytes"])
    hot_reserved = max(ranks, key=lambda r: r["max_reserved_bytes"])
    return {
        "num_ranks": len(ranks),
        "max_allocated_gib": hot["max_allocated_bytes"] / 2**30,
        "max_allocated_rank": hot["rank"],
        "max_reserved_gib": hot_reserved["max_reserved_bytes"] / 2**30,
        "max_reserved_rank": hot_reserved["rank"],
        "min_max_allocated_gib": min(r["max_allocated_bytes"] for r in ranks) / 2**30,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--doc-lens", required=True, help="comma-separated doc lengths")
    parser.add_argument("--steps", type=int, default=1, help="fb+optim cycles")
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--loss-fn", default="cross_entropy")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--vocab", type=int, default=1000)
    parser.add_argument("--skip-optim", action="store_true")
    parser.add_argument("--logprob-prefix", type=int, default=8)
    parser.add_argument("--op-timeout", type=float, default=3600.0)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    doc_lens = [int(x) for x in args.doc_lens.split(",") if x]
    payload = build_payload(
        seed=args.seed, doc_lens=doc_lens, vocab=args.vocab, loss_fn=args.loss_fn
    )

    result: dict[str, Any] = {
        "label": args.label,
        "url": args.url,
        "doc_lens": doc_lens,
        "loss_fn": args.loss_fn,
        "seed": args.seed,
        "vocab": args.vocab,
        "lr": args.lr,
        "steps": [],
    }

    with httpx.Client(timeout=httpx.Timeout(60.0, read=120.0)) as client:
        status = client.get(f"{args.url}/status")
        status.raise_for_status()
        result["trainer_status"] = status.json()
        memory_stats(client, args.url, reset_peaks=True)

        for step in range(args.steps):
            wall_start = time.time()
            t0 = time.monotonic()
            fb = submit_and_wait(
                client, args.url, "/forward_backward", payload, args.op_timeout
            )
            fb_seconds = time.monotonic() - t0

            step_row: dict[str, Any] = {
                "step": step,
                "loss": fb["loss"],
                "metrics": fb.get("metrics", {}),
                "fb_seconds": fb_seconds,
            }
            outputs = fb.get("loss_fn_outputs") or []
            step_row["logprob_rows"] = len(outputs)
            step_row["logprob_lens"] = [
                len(o["logprobs"]["data"]) for o in outputs if "logprobs" in o
            ]
            # Compact numerical fingerprint per datum: prefix values + sums.
            step_row["logprob_prefix"] = [
                [round(v, 6) for v in o["logprobs"]["data"][: args.logprob_prefix]]
                for o in outputs
                if "logprobs" in o
            ]
            step_row["logprob_sums"] = [
                round(sum(o["logprobs"]["data"]), 4) for o in outputs if "logprobs" in o
            ]

            if not args.skip_optim:
                t1 = time.monotonic()
                optim = submit_and_wait(
                    client,
                    args.url,
                    "/optim_step",
                    {"adam_params": {"learning_rate": args.lr}},
                    args.op_timeout,
                )
                step_row["optim_seconds"] = time.monotonic() - t1
                step_row["optim_metrics"] = optim.get("metrics", {})

            mem = memory_stats(client, args.url, reset_peaks=False)
            step_row["memory"] = summarize_memory(mem)
            step_row["memory_raw"] = mem
            step_row["wall_window"] = [wall_start, time.time()]
            result["steps"].append(step_row)
            print(
                f"step {step}: loss={step_row['loss']:.6f} "
                f"fb={fb_seconds:.1f}s "
                f"grad_norm={step_row.get('optim_metrics', {}).get('grad_norm')}"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
