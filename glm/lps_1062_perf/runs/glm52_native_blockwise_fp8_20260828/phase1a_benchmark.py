"""Benchmark GLM-5.2 native FP8 expert weights without a custom kernel."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

BLOCK_SIZE = 128
DEFAULT_M_VALUES = (256, 512, 1024, 2048, 4096, 8192)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--peak-tflops", type=float, default=2250.0)
    parser.add_argument("--seed", type=int, default=1062)
    return parser.parse_args()


def load_tensors(checkpoint: Path, names: list[str]) -> dict[str, torch.Tensor]:
    weight_map = json.loads(
        (checkpoint / "model.safetensors.index.json").read_text()
    )["weight_map"]
    tensors: dict[str, torch.Tensor] = {}
    for name in names:
        with safe_open(
            checkpoint / weight_map[name], framework="pt", device="cpu"
        ) as shard:
            tensors[name] = shard.get_tensor(name)
    return tensors


def dequantize_blockwise(
    qweight: torch.Tensor, scale_inv: torch.Tensor
) -> torch.Tensor:
    rows, columns = qweight.shape
    scales = scale_inv.repeat_interleave(BLOCK_SIZE, dim=0)[:rows]
    scales = scales.repeat_interleave(BLOCK_SIZE, dim=1)[:, :columns]
    return (qweight.float() * scales).to(torch.bfloat16)


def naive_temporary_mm(
    activation: torch.Tensor,
    qweight: torch.Tensor,
    scale_inv: torch.Tensor,
) -> torch.Tensor:
    temporary_weight = dequantize_blockwise(qweight, scale_inv)
    return torch.mm(activation, temporary_weight.t())


def ordered_bfloat16_bits(tensor: torch.Tensor) -> torch.Tensor:
    bits = tensor.view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where(
        (bits & 0x8000) != 0,
        (~bits) & 0xFFFF,
        bits | 0x8000,
    )


def correctness(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    difference = candidate.float() - reference.float()
    ulp = (
        ordered_bfloat16_bits(candidate) - ordered_bfloat16_bits(reference)
    ).abs()
    values, counts = torch.unique(ulp, return_counts=True)
    histogram = {
        str(int(value)): int(count)
        for value, count in zip(values.cpu(), counts.cpu(), strict=True)
    }
    return {
        "bitwise_equal": bool(torch.equal(candidate, reference)),
        "max_absolute_error": float(difference.abs().max()),
        "mean_absolute_error": float(difference.abs().mean()),
        "ulp_histogram": histogram,
    }


def benchmark(
    operation: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        output = operation()
    torch.cuda.synchronize()
    del output

    repeat_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            output = operation()
        end.record()
        end.synchronize()
        repeat_ms.append(start.elapsed_time(end) / iterations)
        del output

    return {
        "gpu_operation_duration_ms": statistics.median(repeat_ms),
        "repeat_mean_ms": statistics.mean(repeat_ms),
        "repeat_min_ms": min(repeat_ms),
        "repeat_max_ms": max(repeat_ms),
        "repeat_ms": repeat_ms,
    }


def measure_peak(operation: Callable[[], torch.Tensor]) -> dict[str, int]:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    output = operation()
    torch.cuda.synchronize()
    measurement = {
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "incremental_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() - baseline_allocated
        ),
        "incremental_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved() - baseline_reserved
        ),
    }
    del output
    return measurement


def projection_result(
    *,
    name: str,
    qweight: torch.Tensor,
    scale_inv: torch.Tensor,
    m_values: tuple[int, ...],
    warmup: int,
    iterations: int,
    repeats: int,
    peak_tflops: float,
    seed: int,
) -> dict[str, Any]:
    reference_weight = dequantize_blockwise(qweight, scale_inv)
    weight_elements = qweight.numel()
    old_persistent_bytes = weight_elements * torch.bfloat16.itemsize
    native_persistent_bytes = (
        weight_elements * qweight.element_size()
        + scale_inv.numel() * scale_inv.element_size()
    )
    result: dict[str, Any] = {
        "qweight_shape": list(qweight.shape),
        "scale_shape": list(scale_inv.shape),
        "qweight_dtype": str(qweight.dtype),
        "scale_dtype": str(scale_inv.dtype),
        "reference_weight_dtype": str(reference_weight.dtype),
        "persistent_storage": {
            "old_reference_bytes": old_persistent_bytes,
            "native_fp8_bytes": native_persistent_bytes,
            "native_saving_bytes": old_persistent_bytes - native_persistent_bytes,
            "native_saving_percent": (
                100.0 * (old_persistent_bytes - native_persistent_bytes)
                / old_persistent_bytes
            ),
        },
        "m_sweep": [],
    }

    k = qweight.shape[1]
    n = qweight.shape[0]
    for m in m_values:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed + m + (0 if name == "fc1_gate_up" else 1))
        activation = torch.randn(
            (m, k), device="cuda", dtype=torch.bfloat16, generator=generator
        )

        old_reference = partial(torch.mm, activation, reference_weight.t())
        naive_temporary = partial(
            naive_temporary_mm, activation, qweight, scale_inv
        )

        reference_output = old_reference()
        candidate_output = naive_temporary()
        numerical = correctness(reference_output, candidate_output)
        if reference_output.dtype != torch.bfloat16:
            raise AssertionError(f"reference output is {reference_output.dtype}")
        if candidate_output.dtype != torch.bfloat16:
            raise AssertionError(f"candidate output is {candidate_output.dtype}")
        del reference_output, candidate_output

        old_timing = benchmark(
            old_reference,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
        naive_timing = benchmark(
            naive_temporary,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
        old_memory = measure_peak(old_reference)
        naive_memory = measure_peak(naive_temporary)
        useful_flops = 2 * m * n * k
        for arm in (old_timing, naive_timing):
            arm["useful_flops"] = useful_flops
            arm["useful_tflops"] = useful_flops / (
                arm["gpu_operation_duration_ms"] * 1e9
            )
            arm["useful_mfu_percent"] = (
                100.0 * arm["useful_tflops"] / peak_tflops
            )

        result["m_sweep"].append(
            {
                "m": m,
                "activation_shape": [m, k],
                "output_shape": [m, n],
                "output_dtype": str(torch.bfloat16),
                "correctness": numerical,
                "old_reference": {**old_timing, "memory": old_memory},
                "naive_temporary": {**naive_timing, "memory": naive_memory},
                "naive_overhead_percent": (
                    100.0
                    * (
                        naive_timing["gpu_operation_duration_ms"]
                        / old_timing["gpu_operation_duration_ms"]
                        - 1.0
                    )
                ),
            }
        )

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
    cuda_tensors = {name: tensor.cuda() for name, tensor in cpu_tensors.items()}
    del cpu_tensors

    gate = cuda_tensors[f"{prefix}.gate_proj.weight"]
    up = cuda_tensors[f"{prefix}.up_proj.weight"]
    gate_scale = cuda_tensors[f"{prefix}.gate_proj.weight_scale_inv"]
    up_scale = cuda_tensors[f"{prefix}.up_proj.weight_scale_inv"]
    fc1_weight = torch.cat((gate, up), dim=0)
    fc1_scale = torch.cat((gate_scale, up_scale), dim=0)
    fc2_weight = cuda_tensors[f"{prefix}.down_proj.weight"]
    fc2_scale = cuda_tensors[f"{prefix}.down_proj.weight_scale_inv"]
    del cuda_tensors, gate, up, gate_scale, up_scale

    expected = {
        "fc1_weight": (4096, 6144),
        "fc1_scale": (32, 48),
        "fc2_weight": (6144, 2048),
        "fc2_scale": (48, 16),
    }
    actual = {
        "fc1_weight": tuple(fc1_weight.shape),
        "fc1_scale": tuple(fc1_scale.shape),
        "fc2_weight": tuple(fc2_weight.shape),
        "fc2_scale": tuple(fc2_scale.shape),
    }
    if actual != expected:
        raise AssertionError(f"unexpected expert shapes: {actual}")

    device = torch.cuda.get_device_properties(0)
    result = {
        "phase": "1A",
        "scope": "old persistent BF16 and naive temporary dequantization only",
        "checkpoint": str(args.checkpoint),
        "layer": args.layer,
        "expert": args.expert,
        "seed": args.seed,
        "block_size": BLOCK_SIZE,
        "m_values": list(DEFAULT_M_VALUES),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "assumed_peak_bf16_tflops": args.peak_tflops,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": device.name,
            "compute_capability": list(torch.cuda.get_device_capability()),
            "total_memory_bytes": device.total_memory,
        },
        "projections": {},
    }
    result["projections"]["fc1_gate_up"] = projection_result(
        name="fc1_gate_up",
        qweight=fc1_weight,
        scale_inv=fc1_scale,
        m_values=DEFAULT_M_VALUES,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
        peak_tflops=args.peak_tflops,
        seed=args.seed,
    )
    result["projections"]["fc2_down"] = projection_result(
        name="fc2_down",
        qweight=fc2_weight,
        scale_inv=fc2_scale,
        m_values=DEFAULT_M_VALUES,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
        peak_tflops=args.peak_tflops,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
