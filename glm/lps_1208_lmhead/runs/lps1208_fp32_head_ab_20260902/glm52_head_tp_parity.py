#!/usr/bin/env python3
"""Compare TP1 trainer-style and TP8 sampler-style GLM-5.2 LM heads."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open


def load_lm_head(checkpoint: Path) -> torch.Tensor:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    shard = index["weight_map"]["lm_head.weight"]
    with safe_open(checkpoint / shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor("lm_head.weight")
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16 lm_head.weight, got {weight.dtype}")
    return weight


def comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | int]:
    left_f32 = left.float()
    right_f32 = right.float()
    difference = left_f32 - right_f32
    left_logprobs = F.log_softmax(left_f32, dim=-1)
    right_logprobs = F.log_softmax(right_f32, dim=-1)
    kl = (left_logprobs.exp() * (left_logprobs - right_logprobs)).sum(dim=-1)
    top1_mismatch = (left.argmax(dim=-1) != right.argmax(dim=-1)).sum()
    return {
        "different_logits": int((left != right).sum().item()),
        "total_logits": left.numel(),
        "different_fraction": float((left != right).float().mean().item()),
        "logit_max_abs_error": float(difference.abs().max().item()),
        "logit_mean_abs_error": float(difference.abs().mean().item()),
        "kl_mean": float(kl.mean().item()),
        "kl_max": float(kl.max().item()),
        "top1_mismatches": int(top1_mismatch.item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 32, 256, 4096])
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 8:
        raise ValueError(f"expected eight ranks, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False

    if rank == 0:
        cpu_weight = load_lm_head(args.checkpoint)
        shape = torch.tensor(cpu_weight.shape, dtype=torch.int64, device=device)
    else:
        cpu_weight = None
        shape = torch.empty(2, dtype=torch.int64, device=device)
    dist.broadcast(shape, src=0)
    vocab_size, hidden_size = map(int, shape.tolist())
    if vocab_size % world_size:
        raise ValueError(f"vocab size {vocab_size} is not divisible by {world_size}")

    if rank == 0:
        full_weight = cpu_weight.to(device=device)
        del cpu_weight
    else:
        full_weight = torch.empty(
            vocab_size, hidden_size, dtype=torch.bfloat16, device=device
        )
    dist.broadcast(full_weight, src=0)
    vocab_per_rank = vocab_size // world_size
    local_weight = full_weight[
        rank * vocab_per_rank : (rank + 1) * vocab_per_rank
    ].clone()
    if rank != 0:
        del full_weight
    torch.cuda.empty_cache()

    local_weight_fp32 = local_weight.float()
    full_weight_fp32 = full_weight.float() if rank == 0 else None
    results: dict[str, object] = {
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "world_size": world_size,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "shape": {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "vocab_per_rank": vocab_per_rank,
        },
        "cases": [],
    }

    for tokens in args.tokens:
        torch.manual_seed(722 + tokens)
        hidden = torch.randn(
            tokens, hidden_size, dtype=torch.bfloat16, device=device
        )
        dist.broadcast(hidden, src=0)

        case: dict[str, object] = {"tokens": tokens}
        for dtype_name in ("bf16", "fp32"):
            dist.barrier()
            start = time.perf_counter()
            if dtype_name == "bf16":
                local_logits = F.linear(hidden, local_weight)
            else:
                local_logits = F.linear(hidden.float(), local_weight_fp32)
            local_transposed = local_logits.transpose(0, 1).contiguous()
            gathered_transposed = torch.empty(
                vocab_size,
                tokens,
                dtype=local_logits.dtype,
                device=device,
            )
            dist.all_gather_into_tensor(gathered_transposed, local_transposed)
            tp8_logits = gathered_transposed.transpose(0, 1)

            if rank == 0:
                if dtype_name == "bf16":
                    tp1_logits = torch.matmul(hidden, full_weight.transpose(0, 1))
                else:
                    assert full_weight_fp32 is not None
                    tp1_logits = torch.matmul(
                        hidden.float(), full_weight_fp32.transpose(0, 1)
                    )
                torch.cuda.synchronize()
                metrics = comparison(tp1_logits, tp8_logits)
                metrics["elapsed_seconds"] = time.perf_counter() - start
                case[f"tp1_vs_tp8_{dtype_name}"] = metrics
                del tp1_logits

            del local_logits, local_transposed, gathered_transposed, tp8_logits
            torch.cuda.empty_cache()
            dist.barrier()

        if rank == 0:
            results["cases"].append(case)
        del hidden
        torch.cuda.empty_cache()

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
