"""Compare TE W8A8 with compiled BF16 materialization on real GLM experts."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

BLOCK = 128
DEFAULT_M_VALUES = (256, 512, 1024, 2048, 4096, 8192)


def load_tensors(checkpoint: Path, names: list[str]) -> dict[str, torch.Tensor]:
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


def dequantize_group(
    payloads: tuple[torch.Tensor, ...], scales: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    weights = []
    for index in range(len(payloads)):
        payload = payloads[index]
        scale = scales[index]
        block_rows, block_columns = scale.shape
        blocked = payload.float().view(block_rows, BLOCK, block_columns, BLOCK)
        weight = (blocked * scale[:, None, :, None]).reshape(payload.shape)
        weights.append(weight.to(torch.bfloat16))
    return torch.stack(weights)


def build_te_module(
    *,
    payloads: tuple[torch.Tensor, ...],
    scales: tuple[torch.Tensor, ...],
    recipe: Any,
) -> Any:
    import transformer_engine.pytorch as te

    with (
        torch.no_grad(),
        te.quantized_model_init(
            enabled=True,
            recipe=recipe,
            preserve_high_precision_init_val=False,
        ),
    ):
        module = te.GroupedLinear(
            num_gemms=len(payloads),
            in_features=payloads[0].shape[1],
            out_features=payloads[0].shape[0],
            bias=False,
            params_dtype=torch.bfloat16,
            device="cuda",
        )
    module.requires_grad_(False)
    weights = list(module.parameters())
    if len(weights) != len(payloads):
        raise RuntimeError(f"Expected {len(payloads)} TE weights, got {len(weights)}")
    for weight, payload, scale in zip(weights, payloads, scales, strict=True):
        if weight._rowwise_data is None or weight._rowwise_scale_inv is None:
            raise RuntimeError("TE blockwise weight is missing rowwise storage")
        if weight._columnwise_data is not None:
            raise RuntimeError("Frozen TE weight unexpectedly has columnwise storage")
        with torch.no_grad():
            weight._rowwise_data.copy_(payload.view(torch.uint8))
            weight._rowwise_scale_inv.zero_()
            weight._rowwise_scale_inv[:, : scale.shape[1]].copy_(scale)
    return module


def te_external_bf16(
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
            False,
            [None] * module.num_gemms,
            False,
            None,
            module.save_original_input,
            False,
        )
        weights = list(grouped_weights.unbind(dim=0))
        biases = [inp.new_empty(0) for _ in weights]
        output, _ = _GroupedLinear.forward(
            None, inp, non_tensor_args, *weights, *biases
        )
        return output
    finally:
        module.end_forward()


def compiled_bf16_operation(
    compiled,
    payloads: tuple[torch.Tensor, ...],
    scales: tuple[torch.Tensor, ...],
    module: Any,
    inp: torch.Tensor,
    m_splits: list[int],
) -> torch.Tensor:
    weights = compiled(payloads, scales)
    return te_external_bf16(module, inp, m_splits, weights)


def te_w8a8_operation(
    te: Any,
    recipe: Any,
    module: Any,
    inp: torch.Tensor,
    m_splits: list[int],
) -> torch.Tensor:
    with te.autocast(enabled=True, recipe=recipe):
        return module(inp, m_splits)


def time_operation(operation, *, warmup: int, iterations: int, repeats: int) -> dict:
    for _ in range(warmup):
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
    return {
        "gpu_operation_duration_ms": statistics.median(values),
        "repeat_ms": values,
    }


def memory_operation(operation) -> dict:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    output = operation()
    torch.cuda.synchronize()
    result = {
        "incremental_peak_allocated_bytes": torch.cuda.max_memory_allocated() - baseline
    }
    del output
    return result


def ordered_bf16(tensor: torch.Tensor) -> torch.Tensor:
    bits = tensor.view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where(
        (bits & 0x8000) != 0,
        (~bits) & 0xFFFF,
        bits | 0x8000,
    )


def numerical(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    difference = candidate.float() - reference.float()
    sample_count = min(reference.numel(), 1_048_576)
    reference_sample = reference.reshape(-1)[:sample_count]
    candidate_sample = candidate.reshape(-1)[:sample_count]
    ulp = (ordered_bf16(candidate_sample) - ordered_bf16(reference_sample)).abs()
    ulp_histogram = {
        "0": int((ulp == 0).sum()),
        "1": int((ulp == 1).sum()),
        "2": int((ulp == 2).sum()),
        "3-4": int(((ulp >= 3) & (ulp <= 4)).sum()),
        "5-8": int(((ulp >= 5) & (ulp <= 8)).sum()),
        "9-16": int(((ulp >= 9) & (ulp <= 16)).sum()),
        "17-32": int(((ulp >= 17) & (ulp <= 32)).sum()),
        "33-64": int(((ulp >= 33) & (ulp <= 64)).sum()),
        "65-128": int(((ulp >= 65) & (ulp <= 128)).sum()),
        "129-256": int(((ulp >= 129) & (ulp <= 256)).sum()),
        "257-512": int(((ulp >= 257) & (ulp <= 512)).sum()),
        "513-1024": int(((ulp >= 513) & (ulp <= 1024)).sum()),
        ">1024": int((ulp > 1024).sum()),
    }
    return {
        "bitwise_equal": bool(torch.equal(reference, candidate)),
        "max_absolute_error": float(difference.abs().max()),
        "mean_absolute_error": float(difference.abs().mean()),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                reference_sample.float(), candidate_sample.float(), dim=0
            )
        ),
        "ulp_sample_count": sample_count,
        "ulp_histogram": ulp_histogram,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--m-values", type=int, nargs="+", default=DEFAULT_M_VALUES)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Float8BlockScaling, Format, QParams

    qparams = QParams(power_2_scale=False)
    recipe = Float8BlockScaling(
        fp8_format=Format.E4M3,
        fp8_quant_fwd_inp=qparams,
        fp8_quant_fwd_weight=qparams,
        fp8_quant_bwd_grad=qparams,
    )
    prefix = "model.layers.6.mlp.experts"
    names = [
        f"{prefix}.{expert}.{projection}.{suffix}"
        for expert in range(args.experts)
        for projection in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("weight", "weight_scale_inv")
    ]
    tensors = load_tensors(args.checkpoint, names)
    compiled = torch.compile(dequantize_group, fullgraph=True, dynamic=False)
    result = {
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformer_engine": version("transformer_engine"),
            "device": torch.cuda.get_device_name(),
        },
        "experts": args.experts,
        "m_values": args.m_values,
        "projections": {},
    }

    for projection in ("fc1", "fc2"):
        payloads = []
        scales = []
        for expert in range(args.experts):
            expert_prefix = f"{prefix}.{expert}"
            if projection == "fc1":
                payloads.append(
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
                payloads.append(tensors[f"{expert_prefix}.down_proj.weight"])
                scales.append(tensors[f"{expert_prefix}.down_proj.weight_scale_inv"])
        payload_tuple = tuple(payloads)
        scale_tuple = tuple(scales)
        module = build_te_module(
            payloads=payload_tuple,
            scales=scale_tuple,
            recipe=recipe,
        )
        reference_weights = dequantize_group(payload_tuple, scale_tuple)
        compiled(payload_tuple, scale_tuple)
        torch.cuda.synchronize()
        projection_result = {
            "weight_shape": list(reference_weights.shape),
            "persistent_native_bytes": sum(
                payload.numel() * payload.element_size()
                + scale.numel() * scale.element_size()
                for payload, scale in zip(payload_tuple, scale_tuple, strict=True)
            ),
            "persistent_bf16_bytes": reference_weights.numel()
            * reference_weights.element_size(),
            "m_sweep": [],
        }
        for m in args.m_values:
            torch.manual_seed(1062 + m + (0 if projection == "fc1" else 1))
            inp = torch.randn(
                args.experts * m,
                payload_tuple[0].shape[1],
                device="cuda",
                dtype=torch.bfloat16,
            )
            m_splits = [m] * args.experts

            reference_operation = partial(
                te_external_bf16, module, inp, m_splits, reference_weights
            )
            current_operation = partial(
                compiled_bf16_operation,
                compiled,
                payload_tuple,
                scale_tuple,
                module,
                inp,
                m_splits,
            )
            w8a8_operation = partial(
                te_w8a8_operation, te, recipe, module, inp, m_splits
            )

            reference_output = reference_operation()
            current_output = current_operation()
            w8a8_output = w8a8_operation()
            torch.cuda.synchronize()
            reference_timing = time_operation(
                reference_operation,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            current_timing = time_operation(
                current_operation,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            w8a8_timing = time_operation(
                w8a8_operation,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            projection_result["m_sweep"].append(
                {
                    "m": m,
                    "reference": {
                        **reference_timing,
                        "memory": memory_operation(reference_operation),
                    },
                    "compiled_bf16": {
                        **current_timing,
                        "ratio_vs_reference": current_timing[
                            "gpu_operation_duration_ms"
                        ]
                        / reference_timing["gpu_operation_duration_ms"],
                        "numerical": numerical(reference_output, current_output),
                        "memory": memory_operation(current_operation),
                    },
                    "te_w8a8": {
                        **w8a8_timing,
                        "ratio_vs_reference": w8a8_timing["gpu_operation_duration_ms"]
                        / reference_timing["gpu_operation_duration_ms"],
                        "numerical": numerical(reference_output, w8a8_output),
                        "memory": memory_operation(w8a8_operation),
                    },
                }
            )
            del inp, reference_output, current_output, w8a8_output
        result["projections"][projection] = projection_result
        del module, reference_weights, payloads, scales, payload_tuple, scale_tuple
        gc.collect()
        torch.cuda.empty_cache()

    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for projection, projection_result in result["projections"].items():
        for row in projection_result["m_sweep"]:
            print(
                f"{projection} M={row['m']}: "
                f"compiled={row['compiled_bf16']['ratio_vs_reference']:.3f}x "
                f"W8A8={row['te_w8a8']['ratio_vs_reference']:.3f}x "
                f"W8A8 max_err={row['te_w8a8']['numerical']['max_absolute_error']:.6g} "
                f"mean_err={row['te_w8a8']['numerical']['mean_absolute_error']:.6g}",
                flush=True,
            )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
