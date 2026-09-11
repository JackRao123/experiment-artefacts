from __future__ import annotations

from typing import Any, cast

import torch
from megatron.bridge.peft.lora_layers import LoRALinear
from megatron.core.tensor_parallel import copy_to_tensor_model_parallel_region
from megatron.core.tensor_parallel.layers import ColumnParallelLinear


class _FrozenLMHeadWithFP32Output(torch.autograd.Function):
    """Projects a frozen BF16/FP16 head to FP32 and returns input-dtype hidden gradients."""

    @staticmethod
    def forward(ctx: Any, hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(weight)
        ctx.hidden_dtype = hidden.dtype
        return torch.mm(hidden, weight.t(), out_dtype=torch.float32)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        (grad_logits,) = grad_outputs
        (weight,) = ctx.saved_tensors
        grad_high = grad_logits.to(weight.dtype)
        grad_residual = (grad_logits - grad_high.to(grad_logits.dtype)).to(weight.dtype)
        grad_hidden = torch.mm(grad_high, weight, out_dtype=torch.float32)
        grad_hidden.add_(torch.mm(grad_residual, weight, out_dtype=torch.float32))
        return grad_hidden.to(ctx.hidden_dtype), None


class FP32LMHead:
    """Computes FP32 base-head logits and adds the enabled LoRA logits."""

    def __init__(
        self,
        lm_head_layer: LoRALinear,
        output_weight_override: torch.Tensor | None,
    ) -> None:
        self._lora_head = lm_head_layer
        base_head = cast(ColumnParallelLinear, lm_head_layer.to_wrap)
        if base_head.sequence_parallel:
            raise RuntimeError(
                "sequence-parallel hidden states must be gathered before FP32 projection"
            )

        weight = (
            output_weight_override
            if output_weight_override is not None
            else base_head.weight
        )
        if weight is None:
            raise RuntimeError("FP32 output projection requires an output weight")

        self._base_head = base_head
        self._output_weight_override = output_weight_override
        self._uses_tensor_core_gemm = weight.dtype in (torch.bfloat16, torch.float16)
        self._projection_weight = (
            weight if self._uses_tensor_core_gemm else weight.float()
        )

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        if self._uses_tensor_core_gemm:
            logits = self._project_with_tensor_cores(hidden)
        else:
            logits, _ = self._base_head(
                hidden.float(),
                weight=self._projection_weight,
                runtime_gather_output=False,
            )

        if self._lora_head._adapter_enabled:
            lora_logits = self._lora_head.adapter_forward(
                self._lora_head.adapter,
                hidden.contiguous(),
                weight=self._output_weight_override,
                runtime_gather_output=False,
            )
            logits = logits + lora_logits.reshape(logits.shape).float()
        return logits

    def _project_with_tensor_cores(self, hidden: torch.Tensor) -> torch.Tensor:
        tensor_parallel_group = cast(
            torch.distributed.ProcessGroup, self._base_head.tp_group
        )
        if tensor_parallel_group.size() > 1:
            hidden = copy_to_tensor_model_parallel_region(
                hidden, group=tensor_parallel_group
            )
        logits_shape = (*hidden.shape[:-1], self._projection_weight.shape[0])
        hidden = hidden.reshape(-1, hidden.shape[-1])
        return _FrozenLMHeadWithFP32Output.apply(
            hidden, self._projection_weight
        ).reshape(logits_shape)


def requires_fp32_logits(experimental_attention_variant: str | None) -> bool:
    return experimental_attention_variant == "dsa"
