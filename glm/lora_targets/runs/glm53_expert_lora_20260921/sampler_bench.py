"""Measure completion throughput after loading a local exported adapter."""

import argparse
import json
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--name", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    windows = []
    with httpx.Client(base_url=args.url, timeout=1800) as client:
        response = client.get("/v1/models")
        response.raise_for_status()
        base = response.json()["data"][0]["id"]
        response = client.post(
            "/v1/load_lora_adapter",
            json={"lora_name": args.name, "lora_path": args.adapter},
        )
        response.raise_for_status()
        for model in (base, args.name):
            for repeat in range(args.warmups + args.repeats):
                started = time.perf_counter()
                response = client.post(
                    "/v1/completions",
                    json={
                        "model": model,
                        "prompt": ["The capital of France is"] * args.batch_size,
                        "max_tokens": args.output_tokens,
                        "ignore_eos": True,
                        "temperature": 0,
                        "seed": 17,
                        "logprobs": 1,
                    },
                )
                elapsed = time.perf_counter() - started
                response.raise_for_status()
                body = response.json()
                tokens = body["usage"]["completion_tokens"]
                assert tokens == args.batch_size * args.output_tokens
                record = {
                    "model": model,
                    "phase": "warmup" if repeat < args.warmups else "control",
                    "elapsed_s": elapsed,
                    "completion_tokens": tokens,
                    "output_tps_per_gpu": tokens / elapsed / args.num_gpus,
                    "first_choice": body["choices"][0],
                }
                windows.append(record)
                print(model, repeat, record["output_tps_per_gpu"], flush=True)
    aggregates = {}
    for model in (base, args.name):
        controls = [
            w for w in windows if w["model"] == model and w["phase"] == "control"
        ]
        aggregates[model] = sum(w["completion_tokens"] for w in controls) / (
            sum(w["elapsed_s"] for w in controls) * args.num_gpus
        )
    args.output.write_text(
        json.dumps(
            {
                "arguments": {**vars(args), "output": str(args.output)},
                "windows": windows,
                "aggregate_output_tps_per_gpu": aggregates,
            },
            indent=2,
        )
    )
    print(json.dumps(aggregates), flush=True)


if __name__ == "__main__":
    main()
