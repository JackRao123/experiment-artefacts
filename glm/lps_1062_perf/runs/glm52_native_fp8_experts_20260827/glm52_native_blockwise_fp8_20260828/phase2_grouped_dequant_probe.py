"""Probe compiled native-FP8 materialization before BF16 grouped_mm."""

from __future__ import annotations

import argparse
import json
import statistics
from functools import partial
from pathlib import Path

import torch
from safetensors import safe_open

BLOCK = 128


def load(checkpoint: Path, names: list[str]) -> dict[str, torch.Tensor]:
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    result = {}
    for name in names:
        with safe_open(
            checkpoint / weight_map[name], framework="pt", device="cpu"
        ) as shard:
            result[name] = shard.get_tensor(name).cuda()
    return result


def dequantize_one(qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    block_rows, block_columns = scale.shape
    blocked = qweight.float().view(block_rows, BLOCK, block_columns, BLOCK)
    return (blocked * scale[:, None, :, None]).reshape(qweight.shape).to(torch.bfloat16)


def dequantize_group(
    qweights: tuple[torch.Tensor, ...], scales: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    return torch.stack(
        [dequantize_one(qweight, scale) for qweight, scale in zip(qweights, scales)]
    )


def time_operation(operation, *, iterations: int, repeats: int) -> dict:
    for _ in range(10):
        operation()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            operation()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) / iterations)
    return {"median_ms": statistics.median(values), "repeat_ms": values}


def grouped_mm(
    activation: torch.Tensor, offsets: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return torch.nn.functional.grouped_mm(
        activation, weight.transpose(1, 2), offs=offsets
    )


def compiled_dequant_and_grouped_mm(
    compiled,
    qweights: tuple[torch.Tensor, ...],
    scales: tuple[torch.Tensor, ...],
    activation: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    return grouped_mm(activation, offsets, compiled(qweights, scales))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--m", type=int, default=4096)
    args = parser.parse_args()

    prefix = "model.layers.6.mlp.experts"
    names = [
        f"{prefix}.{expert}.{projection}.{suffix}"
        for expert in range(args.experts)
        for projection in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("weight", "weight_scale_inv")
    ]
    tensors = load(args.checkpoint, names)
    projections = {}
    for projection in ("fc1", "fc2"):
        qweights = []
        scales = []
        for expert in range(args.experts):
            expert_prefix = f"{prefix}.{expert}"
            if projection == "fc1":
                qweights.append(
                    torch.cat(
                        (
                            tensors[f"{expert_prefix}.gate_proj.weight"],
                            tensors[f"{expert_prefix}.up_proj.weight"],
                        )
                    )
                )
                scales.append(
                    torch.cat(
                        (
                            tensors[f"{expert_prefix}.gate_proj.weight_scale_inv"],
                            tensors[f"{expert_prefix}.up_proj.weight_scale_inv"],
                        )
                    )
                )
            else:
                qweights.append(tensors[f"{expert_prefix}.down_proj.weight"])
                scales.append(tensors[f"{expert_prefix}.down_proj.weight_scale_inv"])
        projections[projection] = (tuple(qweights), tuple(scales))

    compiled = torch.compile(dequantize_group, fullgraph=True, dynamic=False)
    results = {}
    for name, (qweights, scales) in projections.items():
        reference_weights = dequantize_group(qweights, scales)
        candidate_weights = compiled(qweights, scales)
        torch.cuda.synchronize()
        assert torch.equal(reference_weights, candidate_weights)
        activation = torch.randn(
            args.experts * args.m,
            qweights[0].shape[1],
            device="cuda",
            dtype=torch.bfloat16,
        )
        offsets = torch.full(
            (args.experts,), args.m, device="cuda", dtype=torch.int32
        ).cumsum(0, dtype=torch.int32)

        old_operation = partial(grouped_mm, activation, offsets, reference_weights)
        candidate_operation = partial(
            compiled_dequant_and_grouped_mm,
            compiled,
            qweights,
            scales,
            activation,
            offsets,
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        output = candidate_operation()
        torch.cuda.synchronize()
        memory = {
            "incremental_peak_allocated_bytes": torch.cuda.max_memory_allocated()
            - baseline
        }
        results[name] = {
            "weight_shape": list(reference_weights.shape),
            "old": time_operation(old_operation, iterations=20, repeats=7),
            "compiled_dequant_plus_grouped_mm": time_operation(
                candidate_operation, iterations=20, repeats=7
            ),
            "memory": memory,
        }
        results[name]["ratio"] = (
            results[name]["compiled_dequant_plus_grouped_mm"]["median_ms"]
            / results[name]["old"]["median_ms"]
        )
        del output, activation, reference_weights, candidate_weights

    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
