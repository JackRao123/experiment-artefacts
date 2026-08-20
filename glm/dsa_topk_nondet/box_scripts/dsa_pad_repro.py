#!/usr/bin/env python3
"""Standalone GLM-5.2 DSA THD+CP padding determinism/leakage repro (LPS-1063-class).

Launch (on a >=CP-GPU node, inside the trainers server venv):

    torchrun --nproc_per_node=4 dsa_pad_repro.py --cp 4 --repeats 8

Three arms, one process run, same module weights:
  aligned       doc lens all multiples of 2*cp  -> padded == unpadded (control)
  padded-zero   unaligned doc lens, pad rows zero-filled
  padded-poison same lens, pad-row inputs filled with a large constant

Per arm we measure, across N repeated forwards with bitwise-identical inputs:
  * bitwise determinism of the final output (all rows / real rows only)
  * bitwise determinism of probe tensors: indexer top-k indices, sparse-attn out/lse
  * top-k legality: every selected index within [doc_start, query_pos] (global padded
    coords); count of PAD keys selected by REAL queries
Cross-arm: real-row outputs of padded-zero vs padded-poison must match bitwise if
pad content never reaches real rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist

import megatron.core.parallel_state as parallel_state
from megatron.core.extensions.transformer_engine import TELinear, TENorm
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.experimental_attention_variant import (
    dsa_cudnn_kernels,
    dsa_layout,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerSubmodules,
    DSAttention,
    DSAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import init_method_normal, scaled_init_method_normal


def log0(msg: str) -> None:
    if dist.get_rank() == 0:
        print(msg, flush=True)


def make_config(cp: int) -> MLATransformerConfig:
    kwargs = dict(
        multi_latent_attention=True,
        experimental_attention_variant="dsa",
        num_layers=1,
        hidden_size=7168,
        num_attention_heads=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        dsa_indexer_n_heads=64,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=2048,
        dsa_indexer_loss_coeff=0.0,  # GLM LoRA config: indexer frozen, no loss
        dsa_indexer_use_sparse_loss=False,
        calculate_per_token_loss=False,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        layernorm_epsilon=1e-6,
        normalization="RMSNorm",
        qk_layernorm=True,
        layernorm_zero_centered_gamma=False,
        expert_model_parallel_size=1,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        context_parallel_size=cp,
        cp_comm_type="allgather",
        apply_rope_fusion=False,
        rope_type="rope",
        rotary_scaling_factor=40,
        mscale=1.0,
        mscale_all_dim=1.0,
        rotary_base=10000,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        rotary_interleaved=False,
        recompute_granularity=None,
        recompute_modules=[],
        fine_grained_activation_offloading=False,
        gradient_accumulation_fusion=False,
        fp8=False,
        fp4=False,
        init_method=init_method_normal(0.02),
        output_layer_init_method=scaled_init_method_normal(0.02, 1, multiplier=2.0),
        kv_channels=128,
        num_query_groups=128,
        batch_invariant_mode=False,
        cache_mla_latents=False,
        use_cpu_initialization=False,
        perform_initialization=True,
        symmetric_ar_type=None,
        disable_parameter_transpose_cache=False,
        init_model_with_meta_device=False,
        delay_wgrad_compute=False,
        tp_comm_overlap=False,
        softmax_scale=None,
        attention_backend=AttnBackend.auto,
        dsa_kernel_backend="cudnn",
    )
    # Newer/older MLATransformerConfig may not accept every kwarg; drop unknowns.
    while True:
        try:
            return MLATransformerConfig(**kwargs)
        except TypeError as e:
            msg = str(e)
            dropped = None
            for k in list(kwargs):
                if f"'{k}'" in msg:
                    dropped = k
                    kwargs.pop(k)
                    break
            if dropped is None:
                raise
            log0(f"[config] dropping unsupported kwarg: {dropped}")


def make_attention(config, pg_collection):
    indexer_submodules = DSAIndexerSubmodules(
        linear_wq_b=ModuleSpec(module=TELinear),
        linear_wk=ModuleSpec(module=TELinear),
        k_norm=ModuleSpec(module=TENorm),
        linear_weights_proj=ModuleSpec(module=TELinear),
    )
    indexer_spec = ModuleSpec(module=DSAIndexer, submodules=indexer_submodules)
    return DSAttention(
        config=config,
        submodules=DSAttentionSubmodules(indexer=indexer_spec),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        cp_comm_type="allgather",
        pg_collection=pg_collection,
    ).cuda()


def tensor_hash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()[:16]


PROBES: dict[str, list] = {"topk": [], "sparse": []}
PROBES_ON = {"flag": False}


@contextmanager
def probe_patches():
    orig_topk = dsa_cudnn_kernels._indexer_topk_bshd
    orig_sparse = dsa_cudnn_kernels._run_sparse_attention_forward

    def topk_wrap(*args, **kwargs):
        out = orig_topk(*args, **kwargs)
        if PROBES_ON["flag"]:
            idx, length, _payload = out
            PROBES["topk"].append(
                (idx.detach().clone(), None if length is None else length.detach().clone())
            )
        return out

    def sparse_wrap(*args, **kwargs):
        out = orig_sparse(*args, **kwargs)
        if PROBES_ON["flag"]:
            out_flat, lse = out[0], out[1]
            PROBES["sparse"].append((out_flat.detach().clone(), lse.detach().clone()))
        return out

    dsa_cudnn_kernels._indexer_topk_bshd = topk_wrap
    dsa_cudnn_kernels._run_sparse_attention_forward = sparse_wrap
    try:
        yield
    finally:
        dsa_cudnn_kernels._indexer_topk_bshd = orig_topk
        dsa_cudnn_kernels._run_sparse_attention_forward = orig_sparse


def build_pack(doc_real_lens, pad_multiple):
    padded = [(n + pad_multiple - 1) // pad_multiple * pad_multiple for n in doc_real_lens]
    cu_real = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(doc_real_lens), 0).tolist()), dtype=torch.int32
    )
    cu_pad = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(padded), 0).tolist()), dtype=torch.int32
    )
    return cu_real, cu_pad, max(padded)


def real_row_mask(cu_pad: torch.Tensor, doc_real_lens, positions: torch.Tensor) -> torch.Tensor:
    """True for rows whose global padded position is a real token."""
    cu = cu_pad.to(positions.device, torch.int64)
    doc = torch.searchsorted(cu[1:], positions, right=True)
    starts = cu[doc]
    real = torch.tensor(doc_real_lens, device=positions.device, dtype=torch.int64)[doc]
    return (positions - starts) < real


def run_arm(
    name,
    attention,
    cp,
    cp_rank,
    doc_real_lens,
    pad_multiple,
    pad_fill,
    repeats,
    cfg,
    results,
    tile_period=0,
):
    device = torch.device("cuda")
    cu_real, cu_pad, max_padded_doc = build_pack(doc_real_lens, pad_multiple)
    total = int(cu_pad[-1])
    local_rows = total // cp
    cu_real = cu_real.to(device)
    cu_pad = cu_pad.to(device)

    # Global tensors, identical on every rank (fixed seed, rank-independent).
    g = torch.Generator(device="cuda")
    g.manual_seed(20260819)
    absorbed = cfg.kv_lora_rank + cfg.qk_pos_emb_head_dim

    def grand(*shape):
        return torch.randn(*shape, device=device, dtype=torch.bfloat16, generator=g)

    def gmake(*shape):
        """Random rows, or one random period tiled to `total` rows (gate-like data)."""
        if tile_period <= 0:
            return grand(*shape)
        block = grand(tile_period, *shape[1:])
        reps = (shape[0] + tile_period - 1) // tile_period
        return block.repeat(reps, *([1] * (len(shape) - 1)))[: shape[0]].contiguous()

    q_g = gmake(total, 1, cfg.num_attention_heads, absorbed)
    k_g = gmake(total, 1, 1, absorbed)
    x_g = gmake(total, 1, cfg.hidden_size) * 8.0
    qr_g = gmake(total, 1, cfg.q_lora_rank) * 8.0
    upv = grand(cfg.num_attention_heads, cfg.v_head_dim, cfg.kv_lora_rank)

    pos_all = torch.arange(total, device=device, dtype=torch.int64)
    realmask_g = real_row_mask(cu_pad, doc_real_lens, pos_all)
    padmask_g = ~realmask_g
    if pad_fill == "zero":
        for t in (q_g, k_g, x_g, qr_g):
            t[padmask_g] = 0.0
    elif pad_fill == "poison":
        for t in (q_g, k_g, x_g, qr_g):
            t[padmask_g] = 1000.0

    # This rank's rows under the trainer's zigzag THD CP sharding.
    positions = dsa_layout.build_packed_allgather_cp_local_positions(
        cu_pad, cp, cp_rank, device, output_size=local_rows
    )
    assert positions.numel() == local_rows
    q_l = q_g.index_select(0, positions).contiguous()
    k_l = k_g.index_select(0, positions).contiguous()
    x_l = x_g.index_select(0, positions).contiguous()
    qr_l = qr_g.index_select(0, positions).contiguous()
    local_real = realmask_g.index_select(0, positions)

    psp = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_real,
        cu_seqlens_kv=cu_real,
        cu_seqlens_q_padded=cu_pad,
        cu_seqlens_kv_padded=cu_pad,
        max_seqlen_q=max_padded_doc,
        max_seqlen_kv=max_padded_doc,
    )

    outs = []
    topk_snaps = []
    sparse_snaps = []
    for r in range(repeats):
        PROBES["topk"].clear()
        PROBES["sparse"].clear()
        PROBES_ON["flag"] = True
        with torch.no_grad():
            out = attention(
                query=q_l,
                key=k_l,
                value=None,
                attention_mask=None,
                x=x_l,
                qr=qr_l,
                attn_mask_type=AttnMaskType.causal,
                packed_seq_params=psp,
                up_v_weight=upv,
            )
        PROBES_ON["flag"] = False
        torch.cuda.synchronize()
        outs.append(out.detach().clone())
        topk_snaps.append([t[0] for t in PROBES["topk"]])
        sparse_snaps.append([(s[0].clone(), s[1].clone()) for s in PROBES["sparse"]])

    ref = outs[0]
    all_bitwise = True
    real_bitwise = True
    max_diff_real = 0.0
    max_diff_pad = 0.0
    for r in range(1, repeats):
        d = (outs[r].float() - ref.float()).abs()
        d_flat = d.reshape(local_rows, -1).amax(dim=1)
        if not torch.equal(outs[r], ref):
            all_bitwise = False
        dr = d_flat[local_real].max().item() if local_real.any() else 0.0
        dp = d_flat[~local_real].max().item() if (~local_real).any() else 0.0
        if dr > 0:
            real_bitwise = False
        max_diff_real = max(max_diff_real, dr)
        max_diff_pad = max(max_diff_pad, dp)

    topk_bitwise = True
    for r in range(1, repeats):
        if len(topk_snaps[r]) != len(topk_snaps[0]) or any(
            not torch.equal(a, b) for a, b in zip(topk_snaps[r], topk_snaps[0])
        ):
            topk_bitwise = False
            break
    sparse_out_bitwise = True
    sparse_lse_bitwise = True
    for r in range(1, repeats):
        for (o_r, l_r), (o_0, l_0) in zip(sparse_snaps[r], sparse_snaps[0]):
            if not torch.equal(o_r, o_0):
                sparse_out_bitwise = False
            if not torch.equal(l_r, l_0):
                sparse_lse_bitwise = False

    # Top-k legality on repeat 0 (indices are global padded coords for the CP path).
    illegal = 0
    pad_selected_by_real = 0
    checked = 0
    for idx_t in topk_snaps[0]:
        if idx_t.dim() != 3:
            continue
        idx = idx_t[0].to(torch.int64)  # (sq_local, k)
        valid = idx >= 0
        pos = positions.view(-1, 1)
        doc = torch.searchsorted(cu_pad.to(torch.int64)[1:], pos.expand_as(idx).clamp(min=0))
        doc_q = torch.searchsorted(cu_pad.to(torch.int64)[1:], positions, right=True)
        starts_q = cu_pad.to(torch.int64)[doc_q].view(-1, 1)
        legal = (~valid) | ((idx >= starts_q) & (idx <= pos))
        illegal += int((~legal).sum().item())
        # pad keys selected by real queries
        idx_c = idx.clamp(min=0)
        kdoc = torch.searchsorted(cu_pad.to(torch.int64)[1:], idx_c, right=True)
        kstart = cu_pad.to(torch.int64)[kdoc]
        kreal = torch.tensor(doc_real_lens, device=device, dtype=torch.int64)[kdoc]
        key_is_pad = (idx_c - kstart) >= kreal
        pad_selected_by_real += int(
            (valid & key_is_pad & local_real.view(-1, 1)).sum().item()
        )
        checked += int(valid.sum().item())

    def reduce_min_flag(flag: bool) -> bool:
        t = torch.tensor([1 if flag else 0], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return bool(t.item())

    def reduce_max_val(v: float) -> float:
        t = torch.tensor([v], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t.item())

    def reduce_sum_val(v: int) -> int:
        t = torch.tensor([v], device=device, dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return int(t.item())

    summary = {
        "arm": name,
        "doc_real_lens": list(doc_real_lens),
        "pad_fill": pad_fill,
        "repeats": repeats,
        "out_bitwise_all_rows": reduce_min_flag(all_bitwise),
        "out_bitwise_real_rows": reduce_min_flag(real_bitwise),
        "max_diff_real_rows": reduce_max_val(max_diff_real),
        "max_diff_pad_rows": reduce_max_val(max_diff_pad),
        "topk_bitwise": reduce_min_flag(topk_bitwise),
        "sparse_out_bitwise": reduce_min_flag(sparse_out_bitwise),
        "sparse_lse_bitwise": reduce_min_flag(sparse_lse_bitwise),
        "topk_illegal_entries": reduce_sum_val(illegal),
        "pad_keys_selected_by_real_queries": reduce_sum_val(pad_selected_by_real),
        "topk_valid_checked": reduce_sum_val(checked),
    }
    log0(json.dumps(summary))
    results[name] = summary
    # Keep repeat-0 output + real mask for cross-arm leakage compare.
    return outs[0], local_real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument(
        "--doc-lens", default="4093,2049,3111", help="real doc lens for padded arms"
    )
    args = ap.parse_args()

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, context_parallel_size=args.cp
    )
    model_parallel_cuda_manual_seed(1234)
    torch.manual_seed(1234)

    cp_rank = parallel_state.get_context_parallel_rank()
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
    cfg = make_config(args.cp)
    attention = make_attention(cfg, pg_collection).eval()

    pad_multiple = 2 * args.cp
    lens_padded = [int(x) for x in args.doc_lens.split(",")]
    lens_aligned = [
        (n + pad_multiple - 1) // pad_multiple * pad_multiple for n in lens_padded
    ]

    results: dict[str, dict] = {}
    with probe_patches():
        run_arm(
            "aligned-tiled37",
            attention,
            args.cp,
            cp_rank,
            lens_aligned,
            pad_multiple,
            "none",
            args.repeats,
            cfg,
            results,
            tile_period=37,
        )
        run_arm(
            "padded-zero-tiled37",
            attention,
            args.cp,
            cp_rank,
            lens_padded,
            pad_multiple,
            "zero",
            args.repeats,
            cfg,
            results,
            tile_period=37,
        )
        run_arm(
            "aligned",
            attention,
            args.cp,
            cp_rank,
            lens_aligned,
            pad_multiple,
            "none",
            args.repeats,
            cfg,
            results,
        )
        out_zero, mask_zero = run_arm(
            "padded-zero",
            attention,
            args.cp,
            cp_rank,
            lens_padded,
            pad_multiple,
            "zero",
            args.repeats,
            cfg,
            results,
        )
        out_poison, mask_poison = run_arm(
            "padded-poison",
            attention,
            args.cp,
            cp_rank,
            lens_padded,
            pad_multiple,
            "poison",
            args.repeats,
            cfg,
            results,
        )

    # Cross-arm pad leakage: real-row outputs must be bitwise equal if pad content
    # never reaches real rows.
    assert torch.equal(mask_zero, mask_poison)
    zr = out_zero.reshape(out_zero.size(0), -1)[mask_zero]
    pr = out_poison.reshape(out_poison.size(0), -1)[mask_poison]
    leak_bitwise = torch.equal(zr, pr)
    leak_max = (zr.float() - pr.float()).abs().max().item() if zr.numel() else 0.0
    n_leak_rows = int(((zr.float() - pr.float()).abs().amax(dim=1) > 0).sum().item())
    t = torch.tensor(
        [0 if leak_bitwise else 1, n_leak_rows], device="cuda", dtype=torch.int64
    )
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    tm = torch.tensor([leak_max], device="cuda", dtype=torch.float64)
    dist.all_reduce(tm, op=dist.ReduceOp.MAX)
    log0(
        json.dumps(
            {
                "check": "pad_content_leakage_into_real_rows",
                "ranks_with_leak": int(t[0].item()),
                "leaked_real_rows_total": int(t[1].item()),
                "max_abs_leak": float(tm.item()),
            }
        )
    )
    log0("DONE")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
