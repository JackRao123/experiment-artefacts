#!/usr/bin/env python3
"""Compare the old and split-BF16 LM-head backward implementations on CUDA."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open


def _load_tensor(checkpoint: Path, key: str) -> torch.Tensor:
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shard = index["weight_map"][key]
        weight_path = checkpoint / shard
    else:
        weight_path = checkpoint / "model.safetensors"

    if not weight_path.exists():
        raise FileNotFoundError(f"no safetensors file found for {key!r} in {checkpoint}")

    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        if key not in handle.keys():
            raise KeyError(f"{key!r} is not present in {weight_path}")
        return handle.get_tensor(key)


def _rms_norm_epsilon(checkpoint: Path, override: float | None) -> float:
    if override is not None:
        return override
    config_path = checkpoint / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} does not exist; pass --rms-norm-eps explicitly"
        )
    config = json.loads(config_path.read_text())
    for key in ("rms_norm_eps", "layer_norm_epsilon"):
        value = config.get(key)
        if value is not None:
            return float(value)
    raise KeyError(
        "config.json has neither rms_norm_eps nor layer_norm_epsilon; "
        "pass --rms-norm-eps explicitly"
    )


def _rms_norm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Match the common FP32-statistics, BF16-output RMSNorm formulation."""
    normalized = hidden.float()
    variance = normalized.square().mean(dim=-1, keepdim=True)
    normalized.mul_(torch.rsqrt(variance + epsilon))
    return normalized.to(hidden.dtype).mul_(weight)


def _old_backward(
    grad_logits: torch.Tensor,
    weight_fp32: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Previous autograd behavior: FP32 GEMM followed by the cast boundary."""
    return torch.mm(grad_logits, weight_fp32).to(output_dtype)


def _new_backward(
    grad_logits: torch.Tensor,
    weight: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Current PR behavior: two BF16 tensor-core GEMMs with FP32 outputs."""
    hi = grad_logits.to(weight.dtype)
    lo = (grad_logits - hi.to(grad_logits.dtype)).to(weight.dtype)
    grad_hidden = torch.mm(hi, weight, out_dtype=torch.float32)
    grad_hidden.add_(torch.mm(lo, weight, out_dtype=torch.float32))
    return grad_hidden.to(output_dtype)


def _old_backward_fp32(
    grad_logits: torch.Tensor,
    weight_fp32: torch.Tensor,
) -> torch.Tensor:
    return torch.mm(grad_logits, weight_fp32)


def _new_backward_fp32(
    grad_logits: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    hi = grad_logits.to(weight.dtype)
    lo = (grad_logits - hi.to(grad_logits.dtype)).to(weight.dtype)
    result = torch.mm(hi, weight, out_dtype=torch.float32)
    result.add_(torch.mm(lo, weight, out_dtype=torch.float32))
    return result


def _time_cuda(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        timings.append((start, end))
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in timings]


def _timing_summary(milliseconds: list[float]) -> str:
    return (
        f"median={statistics.median(milliseconds):.3f} ms, "
        f"mean={statistics.mean(milliseconds):.3f} ms, "
        f"min={min(milliseconds):.3f} ms"
    )


def _gib(tensor: torch.Tensor) -> float:
    return tensor.numel() * tensor.element_size() / 1024**3


def _cross_entropy_gradient(
    hidden: torch.Tensor,
    weight_fp32: torch.Tensor,
    *,
    lora_rank: int,
    lora_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate d(mean cross entropy)/d(logits) from a fixed forward pass."""
    logits = torch.mm(hidden.float(), weight_fp32.t())
    if lora_rank > 0:
        lora_a = (
            torch.randn(
                lora_rank,
                hidden.shape[1],
                dtype=torch.bfloat16,
                device=hidden.device,
            )
            * 0.01
        )
        lora_b = (
            torch.randn(
                weight_fp32.shape[0],
                lora_rank,
                dtype=torch.bfloat16,
                device=hidden.device,
            )
            * 0.01
        )
        lora_delta = torch.mm(torch.mm(hidden, lora_a.t()), lora_b.t())
        lora_delta.mul_(lora_scale)
        logits.add_(lora_delta.float())
        del lora_a, lora_b, lora_delta

    labels = torch.randint(
        0,
        weight_fp32.shape[0],
        (hidden.shape[0],),
        dtype=torch.int64,
        device=hidden.device,
    )
    grad_logits = torch.softmax(logits, dim=-1)
    grad_logits[torch.arange(hidden.shape[0], device=hidden.device), labels] -= 1.0
    grad_logits.div_(hidden.shape[0])
    return grad_logits, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Hugging Face checkpoint directory containing safetensors",
    )
    parser.add_argument("--weight-key", default="lm_head.weight")
    parser.add_argument("--norm-weight-key", default="model.norm.weight")
    parser.add_argument(
        "--rms-norm-eps",
        type=float,
        default=None,
        help="override config.json rms_norm_eps/layer_norm_epsilon",
    )
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="synthetic BF16 LM-head LoRA rank; use 0 for the frozen base only",
    )
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1208)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    weight_cpu = _load_tensor(args.checkpoint, args.weight_key)
    if weight_cpu.dtype != torch.bfloat16:
        raise TypeError(
            f"expected a BF16 LM-head weight, got {weight_cpu.dtype}; "
            "this benchmark targets the GLM BF16 path"
        )
    if weight_cpu.ndim != 2:
        raise ValueError(f"expected a 2D LM-head weight, got {weight_cpu.shape}")

    weight = weight_cpu.to(device=device)
    del weight_cpu
    norm_weight_cpu = _load_tensor(args.checkpoint, args.norm_weight_key)
    if norm_weight_cpu.dtype != torch.bfloat16:
        raise TypeError(
            f"expected a BF16 RMSNorm weight, got {norm_weight_cpu.dtype}"
        )
    if norm_weight_cpu.shape != (weight.shape[1],):
        raise ValueError(
            f"RMSNorm weight shape {tuple(norm_weight_cpu.shape)} does not match "
            f"LM-head hidden size {weight.shape[1]}"
        )
    norm_weight = norm_weight_cpu.to(device=device)
    del norm_weight_cpu
    rms_norm_eps = _rms_norm_epsilon(args.checkpoint, args.rms_norm_eps)

    weight_fp32 = weight.float()
    vocab_size, hidden_size = weight.shape
    pre_norm_hidden = torch.randn(
        args.tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    hidden = _rms_norm(pre_norm_hidden, norm_weight, rms_norm_eps)
    del pre_norm_hidden
    grad_logits, labels = _cross_entropy_gradient(
        hidden,
        weight_fp32,
        lora_rank=args.lora_rank,
        lora_scale=args.lora_scale,
    )

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"weight: {tuple(weight.shape)} {weight.dtype}, {_gib(weight):.3f} GiB")
    print(f"FP32 weight copy: {_gib(weight_fp32):.3f} GiB")
    print(
        f"RMSNorm weight: {tuple(norm_weight.shape)} {norm_weight.dtype}, "
        f"epsilon={rms_norm_eps:g}"
    )
    print(f"hidden: {tuple(hidden.shape)} {hidden.dtype}, {_gib(hidden):.3f} GiB")
    print(f"synthetic LoRA rank: {args.lora_rank}")
    print(
        f"grad_logits: {tuple(grad_logits.shape)} {grad_logits.dtype}, "
        f"{_gib(grad_logits):.3f} GiB"
    )
    print("gradient source: calculated from mean cross entropy")
    print(f"grad_logits max abs: {grad_logits.abs().max().item():.9g}")
    print(f"grad_logits mean abs: {grad_logits.abs().mean().item():.9g}")
    print(f"first five labels: {labels[:5].tolist()}")

    with torch.inference_mode():
        old_fp32 = _old_backward_fp32(grad_logits, weight_fp32)
        new_fp32 = _new_backward_fp32(grad_logits, weight)
        fp32_difference = new_fp32 - old_fp32
        fp32_relative_l2 = (
            fp32_difference.norm(dtype=torch.float64)
            / old_fp32.norm(dtype=torch.float64)
        ).item()

        old_bf16 = old_fp32.to(torch.bfloat16)
        new_bf16 = new_fp32.to(torch.bfloat16)
        bf16_difference = new_bf16.float() - old_bf16.float()
        identical = (new_bf16 == old_bf16).float().mean().item()

        print("\nPrecision versus old FP32 GEMM")
        print(f"FP32 intermediate max abs error: {fp32_difference.abs().max().item():.9g}")
        print(f"FP32 intermediate mean abs error: {fp32_difference.abs().mean().item():.9g}")
        print(f"FP32 intermediate relative L2 error: {fp32_relative_l2:.9g}")
        print(f"Final BF16 bitwise-identical fraction: {identical:.6%}")
        print(f"Final BF16 max abs error: {bf16_difference.abs().max().item():.9g}")
        print(f"Final BF16 mean abs error: {bf16_difference.abs().mean().item():.9g}")

        del old_fp32, new_fp32, fp32_difference
        del old_bf16, new_bf16, bf16_difference
        torch.cuda.empty_cache()

        old_times = _time_cuda(
            lambda: _old_backward(grad_logits, weight_fp32, torch.bfloat16),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        new_times = _time_cuda(
            lambda: _new_backward(grad_logits, weight, torch.bfloat16),
            warmup=args.warmup,
            repeats=args.repeats,
        )

    old_median = statistics.median(old_times)
    new_median = statistics.median(new_times)
    print("\nEnd-to-end backward timing (including final BF16 cast)")
    print(f"Old FP32 GEMM: {_timing_summary(old_times)}")
    print(f"New split-BF16 GEMMs: {_timing_summary(new_times)}")
    print(f"Median speedup: {old_median / new_median:.3f}x")


if __name__ == "__main__":
    main()
