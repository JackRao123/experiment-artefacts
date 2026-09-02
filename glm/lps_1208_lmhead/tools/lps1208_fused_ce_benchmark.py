"""Compare exact large-vocabulary linear-cross-entropy implementations."""

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
from cut_cross_entropy import linear_cross_entropy
from liger_kernel.transformers.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyLoss,
)
from megatron.core import parallel_state
from safetensors import safe_open
from trainers_server_megatron_bridge.chunked_lm_head import (
    CHUNKED_LM_HEAD_SEQ_CHUNK,
    chunked_lm_head_loss_from_hidden,
)


class BareOutputHead(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        weight: torch.Tensor | None,
        runtime_gather_output: bool,
    ) -> tuple[torch.Tensor, None]:
        del runtime_gather_output
        return F.linear(hidden, weight if weight is not None else self.weight), None


class BareLanguageModel:
    def __init__(self, output_layer: BareOutputHead, vocab_size: int) -> None:
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
    parser.add_argument(
        "--method",
        choices=(
            "reference",
            "checkpoint-te",
            "cce",
            "cce-exact",
            "cce-exact-weighted",
            "liger",
            "liger-bf16",
        ),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=16_384)
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
    hidden = torch.randn(
        args.seq_len,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    labels = torch.randint(
        0, vocab_size, (args.seq_len,), device=device, dtype=torch.long
    )
    token_weights = torch.linspace(0.5, 1.5, args.seq_len, device=device)
    language_model = BareLanguageModel(BareOutputHead(weight), vocab_size)
    liger_loss = LigerFusedLinearCrossEntropyLoss(
        reduction="mean", accum_dtype=torch.float32
    )
    liger_loss_none = LigerFusedLinearCrossEntropyLoss(
        reduction="none", accum_dtype=torch.float32
    )

    def compute_loss() -> torch.Tensor:
        if args.method == "checkpoint-te":
            output = chunked_lm_head_loss_from_hidden(
                language_model,
                hidden.unsqueeze(1),
                labels=labels.unsqueeze(0),
                active_mask=torch.ones(
                    1, args.seq_len, device=device, dtype=torch.bool
                ),
                temperatures=None,
                chunk_loss_fn=lambda logprobs, *_: -logprobs / args.seq_len,
            )
            return output.loss_sum

        if args.method in ("cce", "cce-exact", "cce-exact-weighted"):
            nll = linear_cross_entropy(
                hidden,
                weight,
                labels,
                reduction="none" if args.method == "cce-exact-weighted" else "mean",
                shift=0,
                impl="cce" if args.method == "cce" else "cce_exact",
            )
            return (
                (nll * token_weights).mean()
                if args.method == "cce-exact-weighted"
                else nll
            )
        if args.method == "liger-bf16":
            return liger_loss(weight, hidden, labels)

        hidden_fp32 = hidden.float()
        weight_fp32 = weight.float()
        if args.method == "reference":
            return F.cross_entropy(F.linear(hidden_fp32, weight_fp32), labels)
        if args.method == "liger":
            return liger_loss(weight_fp32, hidden_fp32, labels)
        raise AssertionError(args.method)

    def run_once() -> tuple[float, int, int, torch.Tensor]:
        hidden.grad = None
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss = compute_loss()
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        return start.elapsed_time(end), baseline, peak, loss.detach()

    for _ in range(args.warmups):
        run_once()

    controls = [run_once() for _ in range(args.repeats)]
    times_ms = [item[0] for item in controls]
    baselines = [item[1] for item in controls]
    peaks = [item[2] for item in controls]
    losses = [item[3].item() for item in controls]

    precision_probe = None
    if args.method != "checkpoint-te":
        probe_tokens = min(256, args.seq_len)
        probe_hidden = hidden[:probe_tokens].detach()
        probe_labels = labels[:probe_tokens]

        def candidate_nll(probe: torch.Tensor) -> torch.Tensor:
            if args.method == "reference":
                return F.cross_entropy(
                    F.linear(probe.float(), weight.float()),
                    probe_labels,
                    reduction="none",
                )
            if args.method in ("cce", "cce-exact", "cce-exact-weighted"):
                return linear_cross_entropy(
                    probe,
                    weight,
                    probe_labels,
                    reduction="none",
                    shift=0,
                    impl="cce" if args.method == "cce" else "cce_exact",
                )
            if args.method == "liger-bf16":
                return liger_loss_none(weight, probe, probe_labels)
            if args.method == "liger":
                return liger_loss_none(weight.float(), probe.float(), probe_labels)
            raise AssertionError(args.method)

        with torch.no_grad():
            reference_nll = F.cross_entropy(
                F.linear(probe_hidden.float(), weight.float()),
                probe_labels,
                reduction="none",
            )
            measured_nll = candidate_nll(probe_hidden)
            nll_diff = measured_nll.float() - reference_nll

        torch.manual_seed(1209)
        upstream = torch.randn(probe_tokens, device=device, dtype=torch.float32)
        reference_hidden = probe_hidden.clone().requires_grad_(True)
        reference_weighted = (
            F.cross_entropy(
                F.linear(reference_hidden.float(), weight.float()),
                probe_labels,
                reduction="none",
            )
            * upstream
        ).sum()
        reference_weighted.backward()
        measured_hidden = probe_hidden.clone().requires_grad_(True)
        measured_weighted = (candidate_nll(measured_hidden) * upstream).sum()
        measured_weighted.backward()
        grad_diff = measured_hidden.grad.float() - reference_hidden.grad.float()
        precision_probe = {
            "tokens": probe_tokens,
            "nll_max_abs_error": nll_diff.abs().max().item(),
            "nll_mean_abs_error": nll_diff.abs().mean().item(),
            "nll_p99_abs_error": torch.quantile(nll_diff.abs(), 0.99).item(),
            "hidden_grad_max_abs_error": grad_diff.abs().max().item(),
            "hidden_grad_mean_abs_error": grad_diff.abs().mean().item(),
            "hidden_grad_p99_abs_error": torch.quantile(
                grad_diff.abs().flatten(), 0.99
            ).item(),
        }

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

    mean_ms = statistics.mean(times_ms)
    result = {
        "method": args.method,
        "shape": {
            "seq_len": args.seq_len,
            "chunk_size": CHUNKED_LM_HEAD_SEQ_CHUNK,
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "tp": 1,
            "cp": 1,
            "lora_rank": 0,
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
        },
        "runtime": {
            "controls_ms": times_ms,
            "mean_ms": mean_ms,
            "median_ms": statistics.median(times_ms),
            "min_ms": min(times_ms),
            "max_ms": max(times_ms),
            "tokens_per_second": args.seq_len / (mean_ms / 1000),
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
            "precision_probe": precision_probe,
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
