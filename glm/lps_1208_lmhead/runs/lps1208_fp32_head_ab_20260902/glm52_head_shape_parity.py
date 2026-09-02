#!/usr/bin/env python3
"""Compare trainer-sized and sampler-sized GLM-5.2 LM-head projections."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open


def load_lm_head(checkpoint: Path) -> torch.Tensor:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    shard = index["weight_map"]["lm_head.weight"]
    with safe_open(checkpoint / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor("lm_head.weight")


def comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | int]:
    left_f64 = left.double()
    right_f64 = right.double()
    difference = left_f64 - right_f64
    left_logprobs = F.log_softmax(left_f64, dim=-1)
    right_logprobs = F.log_softmax(right_f64, dim=-1)
    kl = (left_logprobs.exp() * (left_logprobs - right_logprobs)).sum(dim=-1)
    return {
        "different_logits": int((left != right).sum().item()),
        "total_logits": left.numel(),
        "different_fraction": float((left != right).float().mean().item()),
        "logit_max_abs_error": float(difference.abs().max().item()),
        "logit_mean_abs_error": float(difference.abs().mean().item()),
        "kl_mean": float(kl.mean().item()),
        "kl_max": float(kl.max().item()),
        "top1_mismatches": int(
            (left.argmax(dim=-1) != right.argmax(dim=-1)).sum().item()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trainer-tokens", type=int, default=4096)
    parser.add_argument("--probe-tokens", type=int, default=256)
    parser.add_argument("--sampler-batches", type=int, nargs="+", default=[1, 8, 32, 256])
    parser.add_argument("--lora-rank", type=int, default=32)
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
    vocab_per_rank = vocab_size // world_size

    if rank == 0:
        full_weight = cpu_weight.to(device=device)
        del cpu_weight
    else:
        full_weight = torch.empty(
            vocab_size, hidden_size, dtype=torch.bfloat16, device=device
        )
    dist.broadcast(full_weight, src=0)
    local_weight = full_weight[
        rank * vocab_per_rank : (rank + 1) * vocab_per_rank
    ].clone()
    if rank != 0:
        del full_weight
    local_weight_fp32 = local_weight.float()
    full_weight_fp32 = full_weight.float() if rank == 0 else None
    torch.cuda.empty_cache()

    torch.manual_seed(1265)
    hidden = torch.randn(
        args.trainer_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    dist.broadcast(hidden, src=0)
    torch.manual_seed(1266)
    lora_a = torch.randn(
        args.lora_rank, hidden_size, dtype=torch.bfloat16, device=device
    ) * 0.01
    full_lora_b = torch.randn(
        vocab_size, args.lora_rank, dtype=torch.bfloat16, device=device
    ) * 0.01
    dist.broadcast(lora_a, src=0)
    dist.broadcast(full_lora_b, src=0)
    local_lora_b = full_lora_b[
        rank * vocab_per_rank : (rank + 1) * vocab_per_rank
    ].clone()

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
            "trainer_tokens": args.trainer_tokens,
            "probe_tokens": args.probe_tokens,
        },
        "cases": [],
    }

    for dtype_name in ("bf16", "fp32"):
        for mode in ("base", "lora"):
            if rank == 0:
                if dtype_name == "bf16":
                    trainer_logits = torch.matmul(hidden, full_weight.transpose(0, 1))
                else:
                    assert full_weight_fp32 is not None
                    trainer_logits = torch.matmul(
                        hidden.float(), full_weight_fp32.transpose(0, 1)
                    )
                if mode == "lora":
                    trainer_delta = F.linear(F.linear(hidden, lora_a), full_lora_b)
                    trainer_logits = (
                        trainer_logits + trainer_delta
                        if dtype_name == "bf16"
                        else trainer_logits + trainer_delta.float()
                    )
                trainer_probe = trainer_logits[: args.probe_tokens]
            else:
                trainer_logits = None
                trainer_probe = None

            for sampler_batch in args.sampler_batches:
                parts = []
                for start in range(0, args.probe_tokens, sampler_batch):
                    batch_hidden = hidden[
                        start : min(start + sampler_batch, args.probe_tokens)
                    ]
                    if dtype_name == "bf16":
                        batch_logits = F.linear(batch_hidden, local_weight)
                    else:
                        batch_logits = F.linear(
                            batch_hidden.float(), local_weight_fp32
                        )
                    if mode == "lora":
                        batch_delta = F.linear(
                            F.linear(batch_hidden, lora_a), local_lora_b
                        )
                        batch_logits = (
                            batch_logits + batch_delta
                            if dtype_name == "bf16"
                            else batch_logits + batch_delta.float()
                        )
                    parts.append(batch_logits)
                local_logits = torch.cat(parts, dim=0)
                local_transposed = local_logits.transpose(0, 1).contiguous()
                gathered_transposed = torch.empty(
                    vocab_size,
                    args.probe_tokens,
                    dtype=local_logits.dtype,
                    device=device,
                )
                dist.all_gather_into_tensor(gathered_transposed, local_transposed)
                sampler_logits = gathered_transposed.transpose(0, 1)

                if rank == 0:
                    assert trainer_probe is not None
                    results["cases"].append(
                        {
                            "dtype": dtype_name,
                            "mode": mode,
                            "sampler_batch": sampler_batch,
                            **comparison(trainer_probe, sampler_logits),
                        }
                    )
                del parts, local_logits, local_transposed, gathered_transposed
                del sampler_logits
                torch.cuda.empty_cache()
                dist.barrier()

            if rank == 0:
                del trainer_logits, trainer_probe
                if mode == "lora":
                    del trainer_delta
            torch.cuda.empty_cache()
            dist.barrier()

    if rank == 0:
        args.output.write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
