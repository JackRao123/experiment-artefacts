"""100-step identical-forward soak for the cudnn-frontend wheel migration.

Two phases:
  A. kernel-level indexer_fwd at the LPS-1003 race geometry (4 GiB out,
     prefill > 2**31 bytes), empty_cache each rep -> re-arms the allocator
     cold-start path; double-exec within rep + bitwise vs rep-0 baseline.
  B. full fused absorbed-MLA DSA module forward (cudnn backend, seqlen 2048),
     identical input every rep, empty_cache each rep, bitwise vs rep-0.

Exit 0 = all reps bitwise-stable. Any disagreement prints the rep and the
finite-mask-flip (erasure) count and exits 1.
"""

import sys

import torch

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def phase_a() -> int:
    from cudnn.deepseek_sparse_attention.indexer_forward import _interface as iface

    torch.manual_seed(1234)
    device = "cuda"
    n_heads, head_dim = 32, 128
    total_q, seg_k, n_segs = 8192, 131072, 4  # out = 4 GiB > 2**31 B
    seg_q = total_q // n_segs
    total_k = seg_k * n_segs
    cu_q = torch.arange(0, total_q + 1, seg_q, dtype=torch.int32, device=device)
    cu_k = torch.arange(0, total_k + 1, seg_k, dtype=torch.int32, device=device)
    offs = torch.full((n_segs,), seg_k - seg_q, dtype=torch.int32, device=device)
    q = torch.randn(total_q, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_k, 1, head_dim, dtype=torch.bfloat16, device=device)
    w = torch.randn(total_q, n_heads, dtype=torch.bfloat16, device=device)

    def call():
        out = iface.indexer_fwd(
            q, k, w, ratio=1,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=seg_q, max_seqlen_k=seg_k,
            q_causal_offsets=offs,
        )
        torch.cuda.synchronize()
        return out

    bad = 0
    baseline_sum = None  # keep a cheap fingerprint, not the 4 GiB tensor
    baseline = None
    for rep in range(REPS):
        torch.cuda.empty_cache()
        o1, o2 = call(), call()
        ok = torch.equal(o1, o2)
        if baseline is None:
            baseline = o1.clone()
        elif not torch.equal(o1, baseline):
            ok = False
        if not ok:
            bad += 1
            f1, f2 = torch.isfinite(o1), torch.isfinite(o2)
            print(f"A rep {rep}: DISAGREE (erasure flips {(f1 != f2).sum().item()})",
                  flush=True)
        del o1, o2
        if rep % 20 == 0:
            print(f"A rep {rep}: ok={ok}", flush=True)
    del baseline
    torch.cuda.empty_cache()
    return bad


def phase_b() -> int:
    sys.path.insert(0, ".")
    from megatron.core.transformer.enums import AttnBackend
    from tests.unit_tests.transformer.experimental_attention_variant import (
        dsa_native_parity_utils as pu,
    )
    from tests.unit_tests.test_utilities import Utils
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_dsa_module_spec_for_backend,
    )
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.packed_seq_params import PackedSeqParams

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    model_parallel_cuda_manual_seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    config = pu._make_config(
        use_sparse_loss=False, calculate_per_token_loss=False, dsa_kernel_backend="cudnn"
    )
    object.__setattr__(config, "attention_backend", AttnBackend.auto)
    spec = get_dsa_module_spec_for_backend(config=config, backend=TESpecProvider())
    layer = build_module(
        spec, config=config, layer_number=1, cp_comm_type=None, pg_collection=None
    ).cuda().to(torch.bfloat16)
    layer.eval()

    seqlen, batch = 2048, 1
    hidden = torch.randn(
        seqlen, batch, config.hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    cu = torch.tensor([0, seqlen], dtype=torch.int32, device="cuda")
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu, cu_seqlens_kv=cu,
        max_seqlen_q=seqlen, max_seqlen_kv=seqlen,
    )

    bad = 0
    baseline = None
    with torch.no_grad():
        for rep in range(REPS):
            torch.cuda.empty_cache()
            out = layer(
                hidden, attention_mask=None, packed_seq_params=packed
            )
            out_t = out[0] if isinstance(out, tuple) else out
            torch.cuda.synchronize()
            if baseline is None:
                baseline = out_t.clone()
            elif not torch.equal(out_t, baseline):
                bad += 1
                d = (out_t.float() - baseline.float()).abs()
                print(
                    f"B rep {rep}: DISAGREE maxabs={d.max().item():.3e} "
                    f"n_diff={(d > 0).sum().item()}",
                    flush=True,
                )
            if rep % 20 == 0:
                print(f"B rep {rep}: ok={baseline is not None and bad == 0}", flush=True)
    return bad


if __name__ == "__main__":
    phases = sys.argv[2] if len(sys.argv) > 2 else "ab"
    bad_a = bad_b = 0
    if "a" in phases:
        bad_a = phase_a()
        print(f"PHASE A: {bad_a}/{REPS} bad reps", flush=True)
    if "b" in phases:
        try:
            bad_b = phase_b()
            print(f"PHASE B: {bad_b}/{REPS} bad reps", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"PHASE B: errored — {type(exc).__name__}: {exc}", flush=True)
            bad_b = -1
    sys.exit(0 if (bad_a == 0 and bad_b == 0) else 1)
