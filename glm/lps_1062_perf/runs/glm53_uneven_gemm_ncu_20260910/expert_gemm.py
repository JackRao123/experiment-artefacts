"""Standalone frozen-BF16 expert GEMMs: PyTorch + Transformer Engine only."""
import torch
from transformer_engine.pytorch import GroupedLinear


class _FrozenGroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, offsets):
        ctx.save_for_backward(weights, offsets)
        return torch.nn.functional.grouped_mm(
            x, weights.transpose(1, 2), offs=offsets
        )

    @staticmethod
    def backward(ctx, grad_output):
        weights, offsets = ctx.saved_tensors
        grad_input = torch.nn.functional.grouped_mm(
            grad_output.contiguous(), weights, offs=offsets
        )
        return grad_input, None, None  # Base expert weights are frozen in LoRA SFT.


class ExpertGEMMPair:
    """Two implementations using identical weights and per-expert row counts.

    Weight packing is setup work, never inside the measured forward/backward.
    Production FSDP's optional implementation instead views contiguous weights
    without a copy. This standalone version does not need FSDP or Megatron.
    """

    def __init__(self, rows, in_features, out_features):
        self.rows = list(rows)
        self.te = GroupedLinear(
            len(rows), in_features, out_features,
            bias=False, params_dtype=torch.bfloat16,
        )
        self.te.requires_grad_(False)
        self.weights = torch.stack(
            [getattr(self.te, f"weight{i}").detach() for i in range(len(rows))]
        )
        self.offsets = torch.tensor(rows, device="cuda", dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )

    def te_forward(self, x):
        return self.te(x, self.rows)

    def grouped_forward(self, x):
        return _FrozenGroupedMM.apply(x, self.weights, self.offsets)
