"""Optimize full-weight dequantization while preserving the Phase 1A algorithm."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch
from phase1a_benchmark import (
    BLOCK_SIZE,
    benchmark,
    correctness,
    dequantize_blockwise,
    load_tensors,
    measure_peak,
)

Dequantize = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--m-values", type=int, nargs="+", default=[4096])
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("repeat_interleave", "broadcast", "compiled_broadcast"),
        default=["repeat_interleave", "broadcast", "compiled_broadcast"],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1062)
    return parser.parse_args()


def dequantize_broadcast(
    qweight: torch.Tensor, scale_inv: torch.Tensor
) -> torch.Tensor:
    row_blocks, column_blocks = scale_inv.shape
    expected_shape = (
        row_blocks * BLOCK_SIZE,
        column_blocks * BLOCK_SIZE,
    )
    if tuple(qweight.shape) != expected_shape:
        raise ValueError(
            f"broadcast path requires aligned blocks: {qweight.shape} != "
            f"{expected_shape}"
        )
    blocked_weight = qweight.float().view(
        row_blocks,
        BLOCK_SIZE,
        column_blocks,
        BLOCK_SIZE,
    )
    scaled_weight = blocked_weight * scale_inv[:, None, :, None]
    return scaled_weight.reshape(qweight.shape).to(torch.bfloat16)


def dequantize_and_mm(
    dequantize: Dequantize,
    activation: torch.Tensor,
    qweight: torch.Tensor,
    scale_inv: torch.Tensor,
) -> torch.Tensor:
    temporary_weight = dequantize(qweight, scale_inv)
    return torch.mm(activation, temporary_weight.t())


def projection_measurements(
    *,
    name: str,
    qweight: torch.Tensor,
    scale_inv: torch.Tensor,
    m_values: list[int],
    arms: dict[str, Dequantize],
    warmup: int,
    iterations: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    reference_weight = dequantize_blockwise(qweight, scale_inv)
    result: dict[str, Any] = {
        "qweight_shape": list(qweight.shape),
        "scale_shape": list(scale_inv.shape),
        "m_sweep": [],
    }
    for m in m_values:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed + m + (0 if name == "fc1_gate_up" else 1))
        activation = torch.randn(
            (m, qweight.shape[1]),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        old_operation = partial(torch.mm, activation, reference_weight.t())
        old_output = old_operation()
        old_timing = benchmark(
            old_operation,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
        row: dict[str, Any] = {
            "m": m,
            "old_reference": old_timing,
            "arms": {},
        }
        for arm_name, dequantize in arms.items():
            candidate_weight = dequantize(qweight, scale_inv)
            weight_correctness = correctness(reference_weight, candidate_weight)
            candidate_operation = partial(
                dequantize_and_mm,
                dequantize,
                activation,
                qweight,
                scale_inv,
            )
            candidate_output = candidate_operation()
            output_correctness = correctness(old_output, candidate_output)
            dequant_timing = benchmark(
                partial(dequantize, qweight, scale_inv),
                warmup=warmup,
                iterations=iterations,
                repeats=repeats,
            )
            operation_timing = benchmark(
                candidate_operation,
                warmup=warmup,
                iterations=iterations,
                repeats=repeats,
            )
            operation_ms = operation_timing["gpu_operation_duration_ms"]
            old_ms = old_timing["gpu_operation_duration_ms"]
            row["arms"][arm_name] = {
                "weight_correctness": weight_correctness,
                "output_correctness": output_correctness,
                "dequantization": dequant_timing,
                "dequant_plus_gemm": operation_timing,
                "ratio_vs_persistent_bf16": operation_ms / old_ms,
                "memory": measure_peak(candidate_operation),
            }
        result["m_sweep"].append(row)
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_grad_enabled(False)

    prefix = f"model.layers.{args.layer}.mlp.experts.{args.expert}"
    tensor_names = [
        f"{prefix}.{projection}.{suffix}"
        for projection in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("weight", "weight_scale_inv")
    ]
    cpu_tensors = load_tensors(args.checkpoint, tensor_names)
    tensors = {name: tensor.cuda() for name, tensor in cpu_tensors.items()}
    gate = tensors[f"{prefix}.gate_proj.weight"]
    up = tensors[f"{prefix}.up_proj.weight"]
    gate_scale = tensors[f"{prefix}.gate_proj.weight_scale_inv"]
    up_scale = tensors[f"{prefix}.up_proj.weight_scale_inv"]
    projections = {
        "fc1_gate_up": (
            torch.cat((gate, up), dim=0),
            torch.cat((gate_scale, up_scale), dim=0),
        ),
        "fc2_down": (
            tensors[f"{prefix}.down_proj.weight"],
            tensors[f"{prefix}.down_proj.weight_scale_inv"],
        ),
    }

    arm_functions: dict[str, Dequantize] = {
        "repeat_interleave": dequantize_blockwise,
        "broadcast": dequantize_broadcast,
    }
    if "compiled_broadcast" in args.arms:
        arm_functions["compiled_broadcast"] = torch.compile(
            dequantize_broadcast,
            fullgraph=True,
            dynamic=False,
        )
    selected_arms = {name: arm_functions[name] for name in args.arms}

    # Compile every requested shape before any timing window.
    if "compiled_broadcast" in selected_arms:
        for qweight, scale_inv in projections.values():
            selected_arms["compiled_broadcast"](qweight, scale_inv)
        torch.cuda.synchronize()

    result: dict[str, Any] = {
        "scope": "full BF16 weight materialization followed by BF16 torch.mm",
        "checkpoint": str(args.checkpoint),
        "layer": args.layer,
        "expert": args.expert,
        "m_values": args.m_values,
        "arms": args.arms,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "projections": {},
    }
    for name, (qweight, scale_inv) in projections.items():
        result["projections"][name] = projection_measurements(
            name=name,
            qweight=qweight,
            scale_inv=scale_inv,
            m_values=args.m_values,
            arms=selected_arms,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            seed=args.seed,
        )

    combined: dict[str, Any] = {}
    for m_index, m in enumerate(args.m_values):
        old_ms = sum(
            projection["m_sweep"][m_index]["old_reference"][
                "gpu_operation_duration_ms"
            ]
            for projection in result["projections"].values()
        )
        combined[str(m)] = {"old_reference_ms": old_ms, "arms": {}}
        for arm_name in args.arms:
            candidate_ms = sum(
                projection["m_sweep"][m_index]["arms"][arm_name][
                    "dequant_plus_gemm"
                ]["gpu_operation_duration_ms"]
                for projection in result["projections"].values()
            )
            combined[str(m)]["arms"][arm_name] = {
                "gpu_operation_duration_ms": candidate_ms,
                "ratio_vs_persistent_bf16": candidate_ms / old_ms,
            }
    result["combined_fc1_fc2"] = combined

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
