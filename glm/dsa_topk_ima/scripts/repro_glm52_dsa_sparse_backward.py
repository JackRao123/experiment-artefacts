#!/usr/bin/env python3
"""Reproduce the GLM-5.2 sparse-attention kernel shape from incident e3m916q.

This is a one-GPU kernel reproducer, not a full 32-GPU trainer replay. It mirrors
the tensors seen by one context-parallel rank after trainers has:

1. padded every datum to a multiple of ``2 * cp_size``;
2. selected that rank's front and mirrored-back THD chunks;
3. all-gathered the compressed KV tensor; and
4. computed compact causal top-k metadata.

Run this inside the trainers worker image on a B200 or B300:

    python server/scripts/repro_glm52_dsa_sparse_backward.py \
        --doc-lengths 55111 --cp-rank 4 --repeats 100

Pass all datum lengths from the failed request to exercise the multi-datum
packed path. The incident log retained only its maximum datum length.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import math
from dataclasses import dataclass

import torch
from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels

_GLM_NUM_HEADS = 64
_GLM_QK_DIM = 576
_GLM_V_DIM = 512
_GLM_TOPK = 2048
_GLM_INDEXER_HEADS = 64
_GLM_INDEXER_DIM = 128
_TRAINER_MAX_SEQUENCE_LENGTH = 262_144


@dataclass(frozen=True)
class IncidentLayout:
    padded_doc_lengths: tuple[int, ...]
    padded_cu_seqlens: tuple[int, ...]
    query_positions: torch.Tensor
    query_doc_starts: torch.Tensor
    query_valid: torch.Tensor
    global_tokens: int


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _build_incident_layout(
    doc_lengths: list[int], *, cp_size: int, cp_rank: int
) -> IncidentLayout:
    if not doc_lengths or any(length <= 0 for length in doc_lengths):
        raise ValueError(f"doc lengths must all be positive, got {doc_lengths}")
    if cp_size <= 1 or not 0 <= cp_rank < cp_size:
        raise ValueError(
            f"expected cp_size > 1 and 0 <= cp_rank < cp_size, got {cp_size=}, {cp_rank=}"
        )

    pad_multiple = 2 * cp_size
    padded_doc_lengths = tuple(
        _round_up(length, pad_multiple) for length in doc_lengths
    )
    global_tokens = sum(padded_doc_lengths)
    if global_tokens > _TRAINER_MAX_SEQUENCE_LENGTH:
        raise ValueError(
            "the production packer would split this list into multiple THD partitions: "
            f"padded total {global_tokens} > {_TRAINER_MAX_SEQUENCE_LENGTH}"
        )

    position_chunks: list[torch.Tensor] = []
    doc_start_chunks: list[torch.Tensor] = []
    valid_chunks: list[torch.Tensor] = []
    padded_cu_seqlens = [0]
    doc_start = 0
    for doc_length, padded_length in zip(doc_lengths, padded_doc_lengths):
        chunk_length = padded_length // pad_multiple
        front_start = doc_start + cp_rank * chunk_length
        back_start = doc_start + (pad_multiple - cp_rank - 1) * chunk_length
        front_positions = torch.arange(
            front_start, front_start + chunk_length, dtype=torch.int64
        )
        back_positions = torch.arange(
            back_start, back_start + chunk_length, dtype=torch.int64
        )
        position_chunks.extend((front_positions, back_positions))
        doc_start_chunks.extend(
            (
                torch.full((chunk_length,), doc_start, dtype=torch.int64),
                torch.full((chunk_length,), doc_start, dtype=torch.int64),
            )
        )
        doc_end = doc_start + doc_length
        valid_chunks.extend((front_positions < doc_end, back_positions < doc_end))
        doc_start += padded_length
        padded_cu_seqlens.append(doc_start)

    return IncidentLayout(
        padded_doc_lengths=padded_doc_lengths,
        padded_cu_seqlens=tuple(padded_cu_seqlens),
        query_positions=torch.cat(position_chunks),
        query_doc_starts=torch.cat(doc_start_chunks),
        query_valid=torch.cat(valid_chunks),
        global_tokens=global_tokens,
    )


def _build_causal_topk(
    layout: IncidentLayout, *, index_pattern: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build sorted, compact top-k indices with the same causal bounds as DSA."""
    causal_lengths = layout.query_positions - layout.query_doc_starts + 1
    topk_lengths = causal_lengths.clamp(max=_GLM_TOPK).to(torch.int32)
    topk_lengths *= layout.query_valid.to(torch.int32)
    safe_topk_lengths = topk_lengths.clamp_min(1).to(torch.int64)
    slots = torch.arange(_GLM_TOPK, dtype=torch.int64).view(1, -1)
    if index_pattern == "contiguous":
        first_indices = layout.query_positions - topk_lengths.to(torch.int64) + 1
        indices = first_indices.view(-1, 1) + slots
    elif index_pattern == "spread":
        # The learned indexer returns sorted, non-contiguous keys. Select evenly
        # spread causal keys without materializing its enormous dense score tensor.
        relative_indices = (
            (2 * slots + 1) * causal_lengths.to(torch.int64).view(-1, 1)
        ) // (2 * safe_topk_lengths.view(-1, 1))
        indices = layout.query_doc_starts.view(-1, 1) + relative_indices
    else:
        raise ValueError(f"unknown index pattern: {index_pattern}")
    indices.masked_fill_(slots >= topk_lengths.to(torch.int64).view(-1, 1), -1)
    return indices.to(torch.int32).unsqueeze(0), topk_lengths.unsqueeze(0)


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--doc-lengths",
        type=int,
        nargs="+",
        default=[55_111],
        help="Unpadded datum lengths in one failed forward_backward request partition.",
    )
    parser.add_argument("--cp-size", type=int, default=32)
    parser.add_argument("--cp-rank", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--std",
        type=float,
        default=0.02,
        help="Std of the random tensor init; sweep upward to probe data-dependent kernel paths.",
    )
    parser.add_argument(
        "--index-pattern",
        choices=("contiguous", "spread"),
        default="spread",
        help="Use recent contiguous keys or sorted non-contiguous keys like the learned indexer.",
    )
    parser.add_argument(
        "--mode",
        choices=("full-indexer", "precomputed-topk"),
        default="full-indexer",
        help="Run the production fused indexer too, or enter at sparse attention with synthetic top-k.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required; run this in the trainers worker image on B200/B300"
        )
    if args.repeats <= 0:
        raise ValueError(f"repeats must be positive, got {args.repeats}")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    layout = _build_incident_layout(
        args.doc_lengths, cp_size=args.cp_size, cp_rank=args.cp_rank
    )
    topk_indices_cpu, topk_length_cpu = _build_causal_topk(
        layout, index_pattern=args.index_pattern
    )
    local_tokens = layout.query_positions.numel()

    print(
        {
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": torch.cuda.get_device_capability(device),
            "torch": torch.__version__,
            "cudnn_backend": torch.backends.cudnn.version(),
            "cudnn_frontend": _package_version("nvidia-cudnn-frontend"),
            "flash_mla": _package_version("flash-mla"),
            "doc_lengths": args.doc_lengths,
            "padded_doc_lengths": layout.padded_doc_lengths,
            "cp_size": args.cp_size,
            "cp_rank": args.cp_rank,
            "global_tokens": layout.global_tokens,
            "local_tokens": local_tokens,
            "topk_shape": tuple(topk_indices_cpu.shape),
            "topk_length_min": int(topk_length_cpu.min()),
            "topk_length_max": int(topk_length_cpu.max()),
            "index_pattern": args.index_pattern,
            "mode": args.mode,
        }
    )

    topk_indices = topk_indices_cpu.to(device=device, non_blocking=True)
    topk_length = topk_length_cpu.to(device=device, non_blocking=True)
    # Production fidelity: at this pin the trainer's PackedSeqParams never sets
    # real_token_mask_q, so FusedIndexerSparseAttnFunc receives
    # query_valid_rows=None and takes the all-rows-nonempty cuDNN sparse
    # backward over every local row, padding rows included. Passing a mask
    # here would silently switch the repro to the compacting backward path
    # that production never runs. layout.query_valid is still used above to
    # build the synthetic precomputed-topk metadata.
    query_valid_rows = None
    varlen_starts = layout.query_doc_starts.to(device=device, non_blocking=True)
    varlen_ends = (layout.query_positions + 1).to(device=device, non_blocking=True)
    packed_cu_seqlens = torch.tensor(
        layout.padded_cu_seqlens, dtype=torch.int32, device=device
    )
    softmax_scale = 1.0 / math.sqrt(_GLM_QK_DIM)

    for iteration in range(args.repeats):
        torch.manual_seed(args.seed + iteration)
        torch.cuda.manual_seed(args.seed + iteration)
        query = torch.empty(
            (local_tokens, 1, _GLM_NUM_HEADS, _GLM_QK_DIM),
            dtype=torch.bfloat16,
            device=device,
        ).normal_(std=args.std)
        key = torch.empty(
            (layout.global_tokens, 1, 1, _GLM_QK_DIM),
            dtype=torch.bfloat16,
            device=device,
        ).normal_(std=args.std)
        query.requires_grad_(True)
        key.requires_grad_(True)

        if args.mode == "full-indexer":
            q_indexer = torch.empty(
                (local_tokens, 1, _GLM_INDEXER_HEADS, _GLM_INDEXER_DIM),
                dtype=torch.bfloat16,
                device=device,
            ).normal_(std=args.std)
            k_indexer = torch.empty(
                (layout.global_tokens, 1, _GLM_INDEXER_DIM),
                dtype=torch.bfloat16,
                device=device,
            ).normal_(std=args.std)
            indexer_weights = torch.empty(
                (local_tokens, 1, _GLM_INDEXER_HEADS),
                dtype=torch.bfloat16,
                device=device,
            ).normal_(std=args.std)
            multi_packed_kwargs = {}
            if len(layout.padded_doc_lengths) > 1:
                multi_packed_kwargs = {
                    "packed_cu_seqlens_q": packed_cu_seqlens,
                    "packed_cu_seqlens_k": packed_cu_seqlens,
                    "packed_max_seqlen_q": max(layout.padded_doc_lengths),
                    "packed_max_seqlen_k": max(layout.padded_doc_lengths),
                    "packed_cp_size": args.cp_size,
                }
            output, indexer_loss = dsa_cudnn_kernels.fused_indexer_sparse_attn(
                query=query,
                kv_full=key.squeeze(2).contiguous(),
                q_indexer=q_indexer,
                k_indexer=k_indexer,
                weights=indexer_weights,
                indexer_topk=_GLM_TOPK,
                softmax_scale=softmax_scale,
                loss_coeff=0.0,
                sparse_loss=False,
                calculate_per_token_loss=True,
                d_v=_GLM_V_DIM,
                varlen_starts=varlen_starts,
                varlen_ends=varlen_ends,
                key_positions=None,
                query_valid_rows=query_valid_rows,
                use_local_indexer_varlen=True,
                single_packed_thd_sequence=len(layout.padded_doc_lengths) == 1,
                local_packed_cp_rank=args.cp_rank,
                local_packed_cp_query_start=0,
                local_packed_cp_query_len=local_tokens,
                **multi_packed_kwargs,
            )
            if indexer_loss.item() != 0.0:
                raise RuntimeError(
                    f"frozen production indexer returned nonzero loss: {indexer_loss.item()}"
                )
        else:
            q_indexer = k_indexer = indexer_weights = indexer_loss = None
            output = dsa_cudnn_kernels.run_fused_absorbed_sparse_attention(
                query=query,
                key=key,
                topk_indices=topk_indices,
                softmax_scale=softmax_scale,
                v_channels=_GLM_V_DIM,
                topk_length=topk_length,
            )
            if output is None:
                raise RuntimeError(
                    "the production fused sparse-attention path declined this shape"
                )
        torch.cuda.synchronize()
        print(f"iteration={iteration} forward=ok")

        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        torch.cuda.synchronize()
        if query.grad is None or key.grad is None:
            raise RuntimeError(
                "sparse-attention backward did not return query/key gradients"
            )
        if not torch.isfinite(query.grad).all() or not torch.isfinite(key.grad).all():
            raise RuntimeError(
                "sparse-attention backward returned non-finite gradients"
            )
        print(
            f"iteration={iteration} backward=ok "
            f"dq_absmax={query.grad.float().abs().max().item():.6g} "
            f"dkv_absmax={key.grad.float().abs().max().item():.6g}"
        )

        del (
            grad_output,
            output,
            query,
            key,
            q_indexer,
            k_indexer,
            indexer_weights,
            indexer_loss,
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
