"""Benchmark the production chunked LM-head path without transformer layers.

Run with torchrun so Megatron and Transformer Engine receive a real process
group, but use world size one for the TP1/CP1 measurements in LPS-1208.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state
from safetensors import safe_open
from trainers_server_megatron_bridge.chunked_lm_head import (
    CHUNKED_LM_HEAD_SEQ_CHUNK,
    chunked_lm_head_loss_from_hidden,
)


class FrozenBase(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)


class MinimalLoRAHead(torch.nn.Module):
    """The LoRALinear surface used by the production GLM FP32-head path."""

    def __init__(
        self,
        weight: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        scale: float,
    ) -> None:
        super().__init__()
        self.to_wrap = FrozenBase(weight)
        self.adapter = object()
        self.lora_a = torch.nn.Parameter(lora_a)
        self.lora_b = torch.nn.Parameter(lora_b)
        self.scale = scale
        self._adapter_enabled = True

    def disable_adapter_layers(self) -> None:
        self._adapter_enabled = False

    def enable_adapter_layers(self) -> None:
        self._adapter_enabled = True

    def _delta(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(hidden, self.lora_a), self.lora_b) * self.scale

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        weight: torch.Tensor | None,
        runtime_gather_output: bool,
    ) -> tuple[torch.Tensor, None]:
        del runtime_gather_output
        base_weight = weight if weight is not None else self.to_wrap.weight
        output = F.linear(hidden, base_weight)
        if self._adapter_enabled:
            output = output + self._delta(hidden)
        return output, None

    def adapter_forward(
        self,
        adapter: object,
        hidden: torch.Tensor,
        *,
        weight: torch.Tensor | None,
        runtime_gather_output: bool,
    ) -> torch.Tensor:
        del weight, runtime_gather_output
        assert adapter is self.adapter
        return self._delta(hidden)


class MinimalLanguageModel:
    def __init__(self, output_layer: MinimalLoRAHead, vocab_size: int) -> None:
        self.output_layer = output_layer
        self.config = SimpleNamespace(
            experimental_attention_variant="dsa",
            mtp_num_layers=None,
            vocab_size=vocab_size,
        )
        self.post_process = True
        self.share_embeddings_and_output_weights = False

    def _scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits


def load_head_weight(checkpoint: Path, device: torch.device) -> torch.Tensor:
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as f:
        weight = f.get_tensor("lm_head.weight")
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16 lm_head.weight, got {weight.dtype}")
    return weight.to(device=device)


def tensor_summary(tensor: torch.Tensor | None) -> dict[str, float | str | None]:
    if tensor is None:
        return {"dtype": None, "sum": None, "abs_sum": None, "square_sum": None}
    value = tensor.detach().float()
    return {
        "dtype": str(tensor.dtype),
        "sum": value.sum().item(),
        "abs_sum": value.abs().sum().item(),
        "square_sum": value.square().sum().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=16_384)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--memory-snapshot", action="store_true")
    args = parser.parse_args()

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "this benchmark currently measures TP1/CP1 with world size one"
        )

    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )
    device = torch.device("cuda")

    torch.manual_seed(1208)
    weight = load_head_weight(args.checkpoint, device)
    vocab_size, hidden_size = weight.shape
    lora_a = (
        torch.randn(args.lora_rank, hidden_size, device=device, dtype=torch.bfloat16)
        * 0.01
    )
    lora_b = (
        torch.randn(vocab_size, args.lora_rank, device=device, dtype=torch.bfloat16)
        * 0.01
    )
    output_layer = MinimalLoRAHead(weight, lora_a, lora_b, scale=1.0)
    language_model = MinimalLanguageModel(output_layer, vocab_size)
    hidden = torch.randn(
        args.seq_len,
        1,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    labels = torch.randint(
        0, vocab_size, (1, args.seq_len), device=device, dtype=torch.long
    )
    active_mask = torch.ones_like(labels, dtype=torch.bool)

    def run_once() -> tuple[float, int, int, torch.Tensor]:
        hidden.grad = None
        output_layer.lora_a.grad = None
        output_layer.lora_b.grad = None
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = chunked_lm_head_loss_from_hidden(
            language_model,
            hidden,
            labels=labels,
            active_mask=active_mask,
            temperatures=None,
            chunk_loss_fn=lambda logprobs, *_: -logprobs / args.seq_len,
        )
        output.loss_sum.backward()
        end.record()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        return start.elapsed_time(end), baseline, peak, output.loss_sum.detach()

    for _ in range(args.warmups):
        run_once()

    controls = [run_once() for _ in range(args.repeats)]
    times_ms = [item[0] for item in controls]
    baselines = [item[1] for item in controls]
    peaks = [item[2] for item in controls]
    losses = [item[3].item() for item in controls]

    snapshot_path = None
    snapshot_peak = None
    if args.memory_snapshot:
        torch.cuda.memory._record_memory_history(
            max_entries=500_000,
            stacks="python",
            context="alloc",
        )
        try:
            _, snapshot_baseline, snapshot_peak_abs, _ = run_once()
            snapshot_peak = snapshot_peak_abs - snapshot_baseline
            snapshot_path = args.output.with_suffix(".memory.pickle")
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)

    result = {
        "label": args.label,
        "shape": {
            "seq_len": args.seq_len,
            "chunk_size": CHUNKED_LM_HEAD_SEQ_CHUNK,
            "chunks": (args.seq_len + CHUNKED_LM_HEAD_SEQ_CHUNK - 1)
            // CHUNKED_LM_HEAD_SEQ_CHUNK,
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "lora_rank": args.lora_rank,
            "tp": 1,
            "cp": 1,
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
        },
        "runtime": {
            "controls_ms": times_ms,
            "mean_ms": statistics.mean(times_ms),
            "median_ms": statistics.median(times_ms),
            "min_ms": min(times_ms),
            "max_ms": max(times_ms),
            "tokens_per_second": args.seq_len / (statistics.mean(times_ms) / 1000),
        },
        "memory": {
            "baseline_allocated_bytes": max(baselines),
            "peak_allocated_bytes": max(peaks),
            "incremental_peak_bytes": max(
                peak - baseline for peak, baseline in zip(peaks, baselines)
            ),
            "snapshot_incremental_peak_bytes": snapshot_peak,
            "snapshot_path": str(snapshot_path) if snapshot_path is not None else None,
        },
        "correctness": {
            "losses": losses,
            "hidden_grad": tensor_summary(hidden.grad),
            "lora_a_grad": tensor_summary(output_layer.lora_a.grad),
            "lora_b_grad": tensor_summary(output_layer.lora_b.grad),
        },
        "completed_at": time.strftime("%F %T"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
