"""Benchmark GLM-5.2 expert weight runtimes on one B300 GPU."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import time
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch.utils.checkpoint import checkpoint

BLOCK = 128
HIDDEN_SIZE = 6144
EXPERT_SIZE = 2048
TOP_K = 8
GLOBAL_SEQUENCE_LENGTH = 131072
EXPERT_PARALLEL_SIZE = 8


def _dequantize_group(
    payloads: tuple[torch.Tensor, ...], scales: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    weights = []
    for payload, scale in zip(payloads, scales, strict=True):
        block_rows, block_columns = scale.shape
        blocked = payload.float().view(block_rows, BLOCK, block_columns, BLOCK)
        weights.append(
            (blocked * scale[:, None, :, None])
            .reshape(payload.shape)
            .to(torch.bfloat16)
        )
    return torch.stack(weights)


def _build_grouped_linear(
    *, num_experts: int, in_features: int, out_features: int, native: bool
) -> Any:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Float8BlockScaling, Format, QParams

    if native:
        qparams = QParams(power_2_scale=False)
        recipe = Float8BlockScaling(
            fp8_format=Format.E4M3,
            fp8_quant_fwd_inp=qparams,
            fp8_quant_fwd_weight=qparams,
            fp8_quant_bwd_grad=qparams,
        )
        with (
            torch.no_grad(),
            te.quantized_model_init(
                enabled=True,
                recipe=recipe,
                preserve_high_precision_init_val=False,
            ),
        ):
            module = te.GroupedLinear(
                num_gemms=num_experts,
                in_features=in_features,
                out_features=out_features,
                bias=False,
                params_dtype=torch.bfloat16,
                device="cuda",
            )
    else:
        module = te.GroupedLinear(
            num_gemms=num_experts,
            in_features=in_features,
            out_features=out_features,
            bias=False,
            params_dtype=torch.bfloat16,
            device="cuda",
        )
    module.requires_grad_(False)
    return module


def _projection_tensors(
    checkpoint: Path, *, projection: str, num_experts: int
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    payloads = []
    scales = []
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as shard:
        for expert in range(num_experts):
            prefix = f"model.layers.0.mlp.experts.{expert}"
            if projection == "fc1":
                payload = torch.cat(
                    (
                        shard.get_tensor(f"{prefix}.gate_proj.weight"),
                        shard.get_tensor(f"{prefix}.up_proj.weight"),
                    )
                )
                scale = torch.cat(
                    (
                        shard.get_tensor(f"{prefix}.gate_proj.weight_scale_inv"),
                        shard.get_tensor(f"{prefix}.up_proj.weight_scale_inv"),
                    )
                )
            else:
                payload = shard.get_tensor(f"{prefix}.down_proj.weight")
                scale = shard.get_tensor(f"{prefix}.down_proj.weight_scale_inv")
            payloads.append(payload.cuda())
            scales.append(scale.cuda())
    return tuple(payloads), tuple(scales)


def _load_projection(
    module: Any,
    checkpoint: Path,
    *,
    projection: str,
    num_experts: int,
    native: bool,
) -> None:
    payloads, scales = _projection_tensors(
        checkpoint, projection=projection, num_experts=num_experts
    )
    if native:
        for index, (payload, scale) in enumerate(
            zip(payloads, scales, strict=True)
        ):
            weight = getattr(module, f"weight{index}")
            if weight._rowwise_data is None or weight._rowwise_scale_inv is None:
                raise RuntimeError("native parameter lacks rowwise storage")
            if weight._columnwise_data is not None:
                raise RuntimeError("native parameter allocated columnwise storage")
            with torch.no_grad():
                weight._rowwise_data.copy_(payload.view(torch.uint8))
                weight._rowwise_scale_inv.zero_()
                weight._rowwise_scale_inv[:, : scale.shape[1]].copy_(scale)
    else:
        weights = _dequantize_group(payloads, scales)
        with torch.no_grad():
            for index, weight in enumerate(weights.unbind(dim=0)):
                getattr(module, f"weight{index}").copy_(weight)
        del weights
    del payloads, scales
    gc.collect()
    torch.cuda.empty_cache()


def _native_storage(module: Any) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    payloads = []
    scales = []
    for index in range(module.num_gemms):
        weight = getattr(module, f"weight{index}")
        payload = weight._rowwise_data
        scale = weight._rowwise_scale_inv
        if payload is None or payload.dtype is not torch.uint8:
            raise RuntimeError(f"weight{index} lacks uint8 rowwise payload")
        if scale is None or scale.dtype is not torch.float32:
            raise RuntimeError(f"weight{index} lacks FP32 rowwise scales")
        block_rows = weight.shape[0] // BLOCK
        block_columns = weight.shape[1] // BLOCK
        payloads.append(payload.view(torch.float8_e4m3fn))
        scales.append(scale[:block_rows, :block_columns])
    return tuple(payloads), tuple(scales)


def _external_grouped_bf16(
    module: Any,
    inp: torch.Tensor,
    m_splits: list[int],
    grouped_weights: torch.Tensor,
) -> torch.Tensor:
    from transformer_engine.pytorch.cpu_offload import is_cpu_offload_enabled
    from transformer_engine.pytorch.module.grouped_linear import _GroupedLinear

    inp = module.prepare_forward(inp, num_gemms=module.num_gemms)
    try:
        (
            input_quantizers,
            weight_quantizers,
            output_quantizers,
            grad_input_quantizers,
            grad_weight_quantizers,
            grad_output_quantizers,
        ) = module._get_quantizers()
        non_tensor_args = (
            m_splits,
            False,
            None,
            False,
            module.fp8_calibration,
            module.wgrad_store,
            input_quantizers,
            weight_quantizers,
            output_quantizers,
            grad_input_quantizers,
            grad_weight_quantizers,
            grad_output_quantizers,
            module.fuse_wgrad_accumulation,
            is_cpu_offload_enabled(),
            module.sequence_parallel,
            module.activation_dtype,
            torch.is_grad_enabled(),
            [None] * module.num_gemms,
            False,
            None,
            module.save_original_input,
            False,
        )
        weights = list(grouped_weights.unbind(dim=0))
        biases = [inp.new_empty(0) for _ in weights]
        result = _GroupedLinear.apply(inp, non_tensor_args, *weights, *biases)
        if isinstance(result, tuple):
            return result[0]
        return result
    finally:
        module.end_forward()


class ExpertBlock:
    def __init__(
        self,
        *,
        arm: str,
        checkpoint: Path,
        num_experts: int,
        m_splits: list[int],
    ) -> None:
        native = arm != "bf16"
        self.arm = arm
        self.m_splits = m_splits
        self.forward_calls = 0
        self.fc1 = _build_grouped_linear(
            num_experts=num_experts,
            in_features=HIDDEN_SIZE,
            out_features=2 * EXPERT_SIZE,
            native=native,
        )
        self.fc2 = _build_grouped_linear(
            num_experts=num_experts,
            in_features=EXPERT_SIZE,
            out_features=HIDDEN_SIZE,
            native=native,
        )
        _load_projection(
            self.fc1,
            checkpoint,
            projection="fc1",
            num_experts=num_experts,
            native=native,
        )
        _load_projection(
            self.fc2,
            checkpoint,
            projection="fc2",
            num_experts=num_experts,
            native=native,
        )
        self.compiled_dequantize = None
        self.native_projections = None
        if arm == "custom":
            self.compiled_dequantize = torch.compile(
                _dequantize_group, fullgraph=True, dynamic=False
            )
            self.native_projections = (
                _native_storage(self.fc1),
                _native_storage(self.fc2),
            )

    def compile(self, inp: torch.Tensor) -> None:
        if self.arm != "custom":
            return
        assert self.compiled_dequantize is not None
        assert self.native_projections is not None
        with torch.no_grad():
            for payloads, scales in self.native_projections:
                materialized = self.compiled_dequantize(payloads, scales)
                del materialized
        torch.cuda.synchronize()

    def __call__(self, inp: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        if self.arm == "custom":
            assert self.compiled_dequantize is not None
            assert self.native_projections is not None
            fc1_weights = self.compiled_dequantize(*self.native_projections[0])
            fc1_output = _external_grouped_bf16(
                self.fc1, inp, self.m_splits, fc1_weights
            )
        else:
            fc1_output = self.fc1(inp, self.m_splits)
        gate, up = fc1_output.chunk(2, dim=-1)
        hidden = torch.nn.functional.silu(gate) * up
        if self.arm == "custom":
            fc2_weights = self.compiled_dequantize(*self.native_projections[1])
            return _external_grouped_bf16(
                self.fc2, hidden, self.m_splits, fc2_weights
            )
        return self.fc2(hidden, self.m_splits)


def _storage_state(block: ExpertBlock) -> dict[str, Any]:
    result = {}
    for name, module in (("fc1", block.fc1), ("fc2", block.fc2)):
        weights = [getattr(module, f"weight{i}") for i in range(module.num_gemms)]
        if block.arm == "bf16":
            result[name] = {
                "dtype": str(weights[0].dtype),
                "parameter_bytes": sum(
                    weight.numel() * weight.element_size() for weight in weights
                ),
                "all_grad_none": all(weight.grad is None for weight in weights),
            }
            continue
        result[name] = {
            "logical_dtype": str(weights[0].dtype),
            "rowwise_dtype": str(weights[0]._rowwise_data.dtype),
            "scale_dtype": str(weights[0]._rowwise_scale_inv.dtype),
            "rowwise_bytes": sum(
                weight._rowwise_data.numel() * weight._rowwise_data.element_size()
                for weight in weights
            ),
            "scale_bytes": sum(
                weight._rowwise_scale_inv.numel()
                * weight._rowwise_scale_inv.element_size()
                for weight in weights
            ),
            "all_columnwise_none": all(
                weight._columnwise_data is None
                and weight._columnwise_scale_inv is None
                for weight in weights
            ),
            "all_grad_none": all(weight.grad is None for weight in weights),
            "rowwise_versions": [weight._rowwise_data._version for weight in weights],
            "scale_versions": [weight._rowwise_scale_inv._version for weight in weights],
        }
    return result


def _sample(tensor: torch.Tensor) -> dict[str, Any]:
    flat = tensor.detach().reshape(-1)
    count = min(flat.numel(), 4096)
    sample = flat[:count].float()
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sample_count": count,
        "sample_sum": float(sample.sum()),
        "sample_abs_sum": float(sample.abs().sum()),
        "sample_square_sum": float(sample.square().sum()),
        "sample_min": float(sample.min()),
        "sample_max": float(sample.max()),
        "sample_values": sample[:32].cpu().tolist(),
    }


def _run_step(
    block: ExpertBlock, inp: torch.Tensor, grad_output: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    inp.grad = None
    calls_before = block.forward_calls
    output = checkpoint(block, inp, use_reentrant=False)
    output.backward(grad_output)
    if block.forward_calls - calls_before != 2:
        raise RuntimeError(
            "full activation recompute must execute the expert block twice; "
            f"observed {block.forward_calls - calls_before} calls"
        )
    if inp.grad is None:
        raise RuntimeError("expert backward produced no activation gradient")
    return output, inp.grad


def _time_step(
    block: ExpertBlock,
    inp: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    retain_outputs: bool = False,
) -> tuple[float, int, torch.Tensor | None, torch.Tensor | None]:
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output, dgrad = _run_step(block, inp, grad_output)
    end.record()
    end.synchronize()
    duration_ms = start.elapsed_time(end)
    peak = torch.cuda.max_memory_allocated()
    if retain_outputs:
        return duration_ms, peak, output, dgrad
    del output, dgrad
    return duration_ms, peak, None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=("bf16", "custom", "generic"))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--controls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1062)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    gc.collect()
    torch.cuda.empty_cache()
    m_splits = [args.m] * args.experts
    routed_rows = sum(m_splits)
    original_tokens = routed_rows // TOP_K

    construction_start = time.perf_counter()
    block = ExpertBlock(
        arm=args.arm,
        checkpoint=args.checkpoint,
        num_experts=args.experts,
        m_splits=m_splits,
    )
    block.compile(
        torch.empty(1, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    )
    construction_seconds = time.perf_counter() - construction_start
    model_allocated_bytes = torch.cuda.memory_allocated()

    torch.manual_seed(args.seed + 1)
    inp = torch.randn(
        routed_rows,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    torch.manual_seed(args.seed + 2)
    grad_output = torch.randn_like(inp)
    initial_allocated_bytes = torch.cuda.memory_allocated()

    warnings.filterwarnings(
        "ignore", message=".*quantized weights without quantized compute.*"
    )
    storage_before = _storage_state(block)
    warmup_ms, warmup_peak, warm_output, warm_dgrad = _time_step(
        block, inp, grad_output, retain_outputs=True
    )
    assert warm_output is not None and warm_dgrad is not None
    correctness = {
        "output": _sample(warm_output),
        "dgrad": _sample(warm_dgrad),
        "loss_proxy_mean": float(warm_output.detach().float().mean()),
        "dgrad_norm": float(torch.linalg.vector_norm(warm_dgrad.float())),
        "output_all_finite": bool(torch.isfinite(warm_output).all()),
        "dgrad_all_finite": bool(torch.isfinite(warm_dgrad).all()),
    }
    del warm_output, warm_dgrad
    inp.grad = None

    controls_ms = []
    control_peaks = []
    for index in range(args.controls):
        duration_ms, peak, _, _ = _time_step(block, inp, grad_output)
        controls_ms.append(duration_ms)
        control_peaks.append(peak)
        print(
            f"{args.arm} control {index + 1}/{args.controls}: {duration_ms:.3f} ms",
            flush=True,
        )
    storage_after = _storage_state(block)
    if args.arm != "bf16":
        for projection in ("fc1", "fc2"):
            if storage_before[projection]["rowwise_versions"] != storage_after[projection][
                "rowwise_versions"
            ]:
                raise RuntimeError(f"{projection} rowwise payload mutated")
            if storage_before[projection]["scale_versions"] != storage_after[projection][
                "scale_versions"
            ]:
                raise RuntimeError(f"{projection} scales mutated")
            if not storage_after[projection]["all_columnwise_none"]:
                raise RuntimeError(f"{projection} allocated columnwise FP8 storage")

    mean_ms = statistics.mean(controls_ms)
    median_ms = statistics.median(controls_ms)
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = {
        "arm": args.arm,
        "design": {
            "global_sequence_length": GLOBAL_SEQUENCE_LENGTH,
            "expert_parallel_size": EXPERT_PARALLEL_SIZE,
            "top_k": TOP_K,
            "num_local_experts": args.experts,
            "rows_per_expert": args.m,
            "routed_rows_per_rank": routed_rows,
            "original_tokens_per_rank": original_tokens,
            "hidden_size": HIDDEN_SIZE,
            "expert_size": EXPERT_SIZE,
            "timed_path": (
                "full-recompute grouped FC1 + SwiGLU + grouped FC2 + "
                "activation backward"
            ),
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformer_engine": version("transformer_engine"),
            "device": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "checkpoint": str(args.checkpoint),
            "script_sha256": script_sha256,
            "seed": args.seed,
        },
        "construction_seconds": construction_seconds,
        "warmup_ms": warmup_ms,
        "controls_ms": controls_ms,
        "statistics": {
            "mean_ms": mean_ms,
            "median_ms": median_ms,
            "min_ms": min(controls_ms),
            "max_ms": max(controls_ms),
            "spread_ms": max(controls_ms) - min(controls_ms),
            "mean_effective_original_tokens_per_second_per_gpu": original_tokens
            / (mean_ms / 1000),
            "median_effective_original_tokens_per_second_per_gpu": original_tokens
            / (median_ms / 1000),
            "mean_routed_rows_per_second": routed_rows / (mean_ms / 1000),
        },
        "memory": {
            "model_allocated_bytes": model_allocated_bytes,
            "initial_with_input_and_grad_bytes": initial_allocated_bytes,
            "warmup_peak_allocated_bytes": warmup_peak,
            "control_peak_allocated_bytes": control_peaks,
            "control_peak_max_allocated_bytes": max(control_peaks),
        },
        "correctness": correctness,
        "storage_before": storage_before,
        "storage_after": storage_after,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["statistics"], indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
