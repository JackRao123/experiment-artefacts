"""Benchmark compiled dequantization while cycling across real experts."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch
from phase1a_benchmark import (
    correctness,
    dequantize_blockwise,
    load_tensors,
    measure_peak,
)
from phase1a_optimize_dequant import (
    dequantize_and_mm,
    dequantize_broadcast,
)

Operation = Callable[[], torch.Tensor]
M_VALUES = (256, 512, 1024, 2048, 4096, 8192)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--first-expert", type=int, default=0)
    parser.add_argument("--expert-count", type=int, default=8)
    parser.add_argument("--m-values", type=int, nargs="+", default=M_VALUES)
    parser.add_argument("--warmup-operations", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1062)
    return parser.parse_args()


def benchmark_cycle(
    operations: list[Operation],
    *,
    warmup_operations: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    for index in range(warmup_operations):
        output = operations[index % len(operations)]()
    torch.cuda.synchronize()
    del output

    repeat_ms = [time_cycle(operations, iterations) for _ in range(repeats)]
    return summarize(repeat_ms)


def time_cycle(operations: list[Operation], iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for index in range(iterations):
        output = operations[index % len(operations)]()
    end.record()
    end.synchronize()
    del output
    return start.elapsed_time(end) / iterations


def summarize(repeat_ms: list[float]) -> dict[str, Any]:
    return {
        "gpu_operation_duration_ms": statistics.median(repeat_ms),
        "repeat_mean_ms": statistics.mean(repeat_ms),
        "repeat_min_ms": min(repeat_ms),
        "repeat_max_ms": max(repeat_ms),
        "repeat_ms": repeat_ms,
    }


def benchmark_paired_cycles(
    old_operations: list[Operation],
    candidate_operations: list[Operation],
    *,
    warmup_operations: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    for index in range(warmup_operations):
        old_operations[index % len(old_operations)]()
        candidate_operations[index % len(candidate_operations)]()
    torch.cuda.synchronize()

    old_repeat_ms = []
    candidate_repeat_ms = []
    order = []
    for repeat in range(repeats):
        if repeat % 2 == 0:
            old_ms = time_cycle(old_operations, iterations)
            candidate_ms = time_cycle(candidate_operations, iterations)
            order.append("old_then_candidate")
        else:
            candidate_ms = time_cycle(candidate_operations, iterations)
            old_ms = time_cycle(old_operations, iterations)
            order.append("candidate_then_old")
        old_repeat_ms.append(old_ms)
        candidate_repeat_ms.append(candidate_ms)
    paired_ratios = [
        candidate_ms / old_ms
        for old_ms, candidate_ms in zip(
            old_repeat_ms, candidate_repeat_ms, strict=True
        )
    ]
    return {
        "old_reference": summarize(old_repeat_ms),
        "compiled_dequant_plus_gemm": summarize(candidate_repeat_ms),
        "paired_ratio_vs_persistent_bf16": statistics.median(paired_ratios),
        "paired_ratios": paired_ratios,
        "order": order,
    }


def independent_block_check(
    qweight: torch.Tensor,
    scale_inv: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, Any]:
    row_blocks, column_blocks = scale_inv.shape
    indices = (
        (0, 0),
        (0, column_blocks - 1),
        (row_blocks - 1, 0),
        (row_blocks - 1, column_blocks - 1),
        (row_blocks // 3, column_blocks // 3),
        (row_blocks // 2, column_blocks // 2),
        (2 * row_blocks // 3, 2 * column_blocks // 3),
        (row_blocks // 4, 3 * column_blocks // 4),
    )
    qblocks = torch.stack(
        [
            qweight[
                row * 128 : (row + 1) * 128,
                column * 128 : (column + 1) * 128,
            ]
            for row, column in indices
        ]
    )
    scales = torch.stack([scale_inv[row, column] for row, column in indices])
    expected = (qblocks.float() * scales[:, None, None]).to(torch.bfloat16)
    actual = torch.stack(
        [
            candidate[
                row * 128 : (row + 1) * 128,
                column * 128 : (column + 1) * 128,
            ]
            for row, column in indices
        ]
    )
    return {
        "sampled_blocks": [list(index) for index in indices],
        **correctness(expected, actual),
    }


def load_experts(
    checkpoint: Path,
    *,
    layer: int,
    expert_ids: list[int],
) -> dict[str, list[tuple[torch.Tensor, torch.Tensor]]]:
    names = [
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}.{suffix}"
        for expert in expert_ids
        for projection in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("weight", "weight_scale_inv")
    ]
    cpu_tensors = load_tensors(checkpoint, names)
    tensors = {name: tensor.cuda() for name, tensor in cpu_tensors.items()}
    projections: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {
        "fc1_gate_up": [],
        "fc2_down": [],
    }
    for expert in expert_ids:
        prefix = f"model.layers.{layer}.mlp.experts.{expert}"
        projections["fc1_gate_up"].append(
            (
                torch.cat(
                    (
                        tensors[f"{prefix}.gate_proj.weight"],
                        tensors[f"{prefix}.up_proj.weight"],
                    ),
                    dim=0,
                ),
                torch.cat(
                    (
                        tensors[f"{prefix}.gate_proj.weight_scale_inv"],
                        tensors[f"{prefix}.up_proj.weight_scale_inv"],
                    ),
                    dim=0,
                ),
            )
        )
        projections["fc2_down"].append(
            (
                tensors[f"{prefix}.down_proj.weight"],
                tensors[f"{prefix}.down_proj.weight_scale_inv"],
            )
        )
    return projections


def main() -> None:
    args = parse_args()
    if args.iterations % args.expert_count != 0:
        raise ValueError("iterations must be divisible by expert-count")
    torch.set_grad_enabled(False)
    expert_ids = list(
        range(args.first_expert, args.first_expert + args.expert_count)
    )
    projections = load_experts(
        args.checkpoint,
        layer=args.layer,
        expert_ids=expert_ids,
    )
    compiled_dequantize = torch.compile(
        dequantize_broadcast,
        fullgraph=True,
        dynamic=False,
    )
    for qweight, scale_inv in projections["fc1_gate_up"][:1]:
        compiled_dequantize(qweight, scale_inv)
    for qweight, scale_inv in projections["fc2_down"][:1]:
        compiled_dequantize(qweight, scale_inv)
    torch.cuda.synchronize()

    result: dict[str, Any] = {
        "scope": "compiled full BF16 materialization plus BF16 torch.mm",
        "cache_model": "round-robin across real expert weights",
        "checkpoint": str(args.checkpoint),
        "layer": args.layer,
        "expert_ids": expert_ids,
        "m_values": args.m_values,
        "warmup_operations": args.warmup_operations,
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
    for projection_name, native_weights in projections.items():
        reference_weights = [
            dequantize_blockwise(qweight, scale_inv)
            for qweight, scale_inv in native_weights
        ]
        first_qweight, first_scale = native_weights[0]
        first_candidate = compiled_dequantize(first_qweight, first_scale)
        projection_result: dict[str, Any] = {
            "qweight_shape": list(first_qweight.shape),
            "scale_shape": list(first_scale.shape),
            "persistent_bf16_working_set_bytes": sum(
                weight.numel() * weight.element_size()
                for weight in reference_weights
            ),
            "native_fp8_working_set_bytes": sum(
                qweight.numel() * qweight.element_size()
                + scale_inv.numel() * scale_inv.element_size()
                for qweight, scale_inv in native_weights
            ),
            "compiled_weight_correctness": correctness(
                reference_weights[0], first_candidate
            ),
            "independent_block_check": independent_block_check(
                first_qweight,
                first_scale,
                first_candidate,
            ),
            "m_sweep": [],
        }
        dequant_operations = [
            partial(compiled_dequantize, qweight, scale_inv)
            for qweight, scale_inv in native_weights
        ]
        dequant_timing = benchmark_cycle(
            dequant_operations,
            warmup_operations=args.warmup_operations,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        for m in args.m_values:
            generator = torch.Generator(device="cuda")
            generator.manual_seed(
                args.seed + m + (0 if projection_name == "fc1_gate_up" else 1)
            )
            activation = torch.randn(
                (m, first_qweight.shape[1]),
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            old_operations = [
                partial(torch.mm, activation, weight.t())
                for weight in reference_weights
            ]
            candidate_operations = [
                partial(
                    dequantize_and_mm,
                    compiled_dequantize,
                    activation,
                    qweight,
                    scale_inv,
                )
                for qweight, scale_inv in native_weights
            ]
            old_output = old_operations[0]()
            candidate_output = candidate_operations[0]()
            paired_timing = benchmark_paired_cycles(
                old_operations,
                candidate_operations,
                warmup_operations=args.warmup_operations,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            old_timing = paired_timing["old_reference"]
            candidate_timing = paired_timing["compiled_dequant_plus_gemm"]
            projection_result["m_sweep"].append(
                {
                    "m": m,
                    "output_correctness": correctness(
                        old_output, candidate_output
                    ),
                    "old_reference": old_timing,
                    "compiled_dequantization": dequant_timing,
                    "compiled_dequant_plus_gemm": candidate_timing,
                    "paired_ratio_vs_persistent_bf16": paired_timing[
                        "paired_ratio_vs_persistent_bf16"
                    ],
                    "paired_ratios": paired_timing["paired_ratios"],
                    "measurement_order": paired_timing["order"],
                    "ratio_of_median_durations": (
                        candidate_timing["gpu_operation_duration_ms"]
                        / old_timing["gpu_operation_duration_ms"]
                    ),
                    "memory": measure_peak(candidate_operations[0]),
                }
            )
        result["projections"][projection_name] = projection_result

    combined: dict[str, Any] = {}
    for index, m in enumerate(args.m_values):
        projection_rows = [
            projection["m_sweep"][index]
            for projection in result["projections"].values()
        ]
        old_ms = sum(
            row["old_reference"]["gpu_operation_duration_ms"]
            for row in projection_rows
        )
        candidate_ms = sum(
            row["compiled_dequant_plus_gemm"]["gpu_operation_duration_ms"]
            for row in projection_rows
        )
        old_repeat_ms = [
            sum(row["old_reference"]["repeat_ms"][repeat] for row in projection_rows)
            for repeat in range(args.repeats)
        ]
        candidate_repeat_ms = [
            sum(
                row["compiled_dequant_plus_gemm"]["repeat_ms"][repeat]
                for row in projection_rows
            )
            for repeat in range(args.repeats)
        ]
        paired_ratios = [
            candidate / old
            for old, candidate in zip(
                old_repeat_ms, candidate_repeat_ms, strict=True
            )
        ]
        combined[str(m)] = {
            "old_reference_ms": old_ms,
            "compiled_dequant_plus_gemm_ms": candidate_ms,
            "paired_ratio_vs_persistent_bf16": statistics.median(
                paired_ratios
            ),
            "paired_ratios": paired_ratios,
            "ratio_of_median_durations": candidate_ms / old_ms,
        }
    result["combined_fc1_fc2"] = combined

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
