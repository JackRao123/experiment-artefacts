"""Check that served routed-expert LoRA weights affect fixed-token scores."""

import argparse
import json
import shutil
from pathlib import Path

import httpx
import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter", type=Path)
    parser.add_argument("control", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--expert-scale", type=float, default=1.0)
    args = parser.parse_args()
    tensors = load_file(str(args.adapter / "adapter_model.safetensors"))
    expert_b = [k for k in tensors if ".mlp.experts." in k and ".lora_B." in k]
    assert expert_b
    nonzero = sum(bool(torch.count_nonzero(tensors[k])) for k in expert_b)
    assert nonzero > 0
    trained_path = args.adapter
    if args.expert_scale != 1.0:
        trained_path = args.control.with_name(args.control.name + "-scaled")
        trained_path.mkdir(parents=True, exist_ok=False)
        for key in expert_b:
            tensors[key] = tensors[key] * args.expert_scale
        save_file(tensors, str(trained_path / "adapter_model.safetensors"))
        shutil.copyfile(
            args.adapter / "adapter_config.json", trained_path / "adapter_config.json"
        )
    for key in expert_b:
        tensors[key] = torch.zeros_like(tensors[key])
    args.control.mkdir(parents=True, exist_ok=False)
    save_file(tensors, str(args.control / "adapter_model.safetensors"))
    del tensors
    shutil.copyfile(
        args.adapter / "adapter_config.json", args.control / "adapter_config.json"
    )
    prompt = "The capital of France is Paris. The capital of Germany is Berlin. Explain why water freezes when the temperature falls below zero degrees Celsius."
    scores = {}
    with httpx.Client(base_url=args.url, timeout=1800) as client:
        for name, path in (
            (args.control.name + "-trained", trained_path),
            (args.control.name + "-zero", args.control),
        ):
            response = client.post(
                "/v1/load_lora_adapter",
                json={"lora_name": name, "lora_path": str(path)},
            )
            response.raise_for_status()
            scores[name] = []
            for _ in range(2):
                response = client.post(
                    "/v1/completions",
                    json={
                        "model": name,
                        "prompt": prompt,
                        "max_tokens": 1,
                        "echo": True,
                        "logprobs": 1,
                        "temperature": 0,
                        "seed": 17,
                    },
                )
                response.raise_for_status()
                # Exclude the generated token so both adapters score identical input.
                values = response.json()["choices"][0]["logprobs"]["token_logprobs"][
                    1:-1
                ]
                assert values and all(v is not None for v in values)
                scores[name].append(values)
    trained, zero = scores.values()
    assert len(trained[0]) == len(zero[0])
    delta = max(abs(a - b) for a, b in zip(trained[0], zero[0], strict=True))
    noise = max(
        abs(a - b) for repeats in scores.values() for a, b in zip(*repeats, strict=True)
    )
    result = {
        "expert_scale": args.expert_scale,
        "expert_b_tensors": len(expert_b),
        "nonzero_expert_b_tensors": nonzero,
        "max_logprob_delta": delta,
        "max_repeat_noise": noise,
        "scores": scores,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "scores"}), flush=True)
    assert delta > max(10 * noise, 1e-6), (
        "expert effect is not distinguishable from repeat noise"
    )


if __name__ == "__main__":
    main()
