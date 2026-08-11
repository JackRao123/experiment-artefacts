#!/usr/bin/env python3
"""Parity test for FIX B — BT_DSA_CP_LAYOUT_CACHE (packed CP layout caching).

Verifies the per-microbatch cache on the packed_seq_params carrier:

  1. cached results are bitwise identical to direct builder calls (first
     computation and all cache hits), across cp ranks, multi-doc cu_seqlens,
     empty docs, and padding tails (both cover flags);
  2. the builder runs exactly once per (key, inputs) — all later calls hit;
  3. cache safety: a DIFFERENT cu_seqlens tensor object (even with equal
     values) misses and recomputes — no stale reads; a changed size/key
     misses too;
  4. the carrier dict pattern matches the existing top-k holder lifetime
     (entries live on the carrier, no module-global state).

Runs on Mac CPU and on the box GPU (device parameter swept when CUDA exists).
The builder's CPU-only guard validates our generated cu_seqlens divisibility.

Usage: python test_dsa_cp_layout_cache_parity.py
"""

import os
import sys
import types

# --- bootstrap: import vendored mcore without the triton-dependent ops leaf ---
def _find_mcore():
    cand = os.environ.get("BT_TEST_MCORE_PATH")
    if cand:
        return cand
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        p = os.path.join(
            d, "server", "vendor", "megatron-bridge", "3rdparty", "Megatron-LM"
        )
        if os.path.isdir(p):
            return p
        d = os.path.dirname(d)
    raise RuntimeError("vendored Megatron-LM not found; set BT_TEST_MCORE_PATH")


sys.path.insert(0, _find_mcore())
_stub = types.ModuleType("megatron.core.transformer.moe.ops.paged_stash")
_stub.GLOBAL_BLOCK_SIZE = 128
_stub.paged_stash_copy_kernel = None
_stub.paged_stash_pop_kernel = None
sys.modules.setdefault("megatron.core.transformer.moe.ops.paged_stash", _stub)

import torch  # noqa: E402

import megatron.core.transformer.experimental_attention_variant.dsa as dsa  # noqa: E402
import megatron.core.transformer.experimental_attention_variant.dsa_layout as dl  # noqa: E402

FAILURES = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name)
    if not cond:
        FAILURES.append(name)


def _make_cu_seqlens(doc_lens, device):
    """cu_seqlens int32 tensor from doc lengths (may include 0-length docs)."""
    vals = [0]
    for L in doc_lens:
        vals.append(vals[-1] + L)
    return torch.tensor(vals, dtype=torch.int32, device=device)


def _builder_calls(cu_q, cu_kv, cp_size, cp_rank, device, out_q, out_g, cover_q, cover_kv, counter):
    def _build():
        counter[0] += 1
        return dl.build_packed_allgather_cp_query_positions_and_key_reorder(
            cu_seqlens_q=cu_q,
            cu_seqlens_kv=cu_kv,
            cp_size=cp_size,
            cp_rank=cp_rank,
            device=device,
            local_output_size=out_q,
            key_local_output_size=out_q,
            global_output_size=out_g,
            query_cu_seqlens_cover_output=cover_q,
            key_cu_seqlens_cover_output=cover_kv,
        )

    return _build


def _run_case(device, cp_size, cp_rank, doc_lens, pad_tail, cover, seed):
    """One cache-parity case: reference (direct) vs cached (2x) bitwise."""
    g = torch.Generator().manual_seed(seed)
    total = sum(doc_lens)
    # output sizes are the PER-RANK local shard sizes (dsa.py passes sq = local rows)
    out_q = (total + pad_tail) // cp_size
    out_g = out_q * cp_size
    cu_q = _make_cu_seqlens(doc_lens, device)
    cu_kv = _make_cu_seqlens(doc_lens, device)  # same values, distinct object
    cover_q, cover_kv = cover

    # reference: direct builder call
    ref_pos, ref_idx = dl.build_packed_allgather_cp_query_positions_and_key_reorder(
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        cp_size=cp_size,
        cp_rank=cp_rank,
        device=device,
        local_output_size=out_q,
        key_local_output_size=out_q,
        global_output_size=out_g,
        query_cu_seqlens_cover_output=cover_q,
        key_cu_seqlens_cover_output=cover_kv,
    )

    carrier = types.SimpleNamespace()  # stand-in for packed_seq_params
    counter = [0]
    key = ("qk_pos_reorder", cp_size, cp_rank, out_q, out_g, cover_q, cover_kv)
    build = _builder_calls(cu_q, cu_kv, cp_size, cp_rank, device, out_q, out_g, cover_q, cover_kv, counter)

    pos1, idx1 = dsa._cached_cp_layout(carrier, key, cu_q, cu_kv, build)
    pos2, idx2 = dsa._cached_cp_layout(carrier, key, cu_q, cu_kv, build)

    tag = f"dev={device} cp={cp_size} rank={cp_rank} docs={len(doc_lens)} pad={pad_tail} cover={cover}"
    check(f"{tag}: first (miss) positions bitwise", torch.equal(pos1, ref_pos))
    check(f"{tag}: first (miss) reorder bitwise", torch.equal(idx1, ref_idx))
    check(f"{tag}: second (hit) positions bitwise", torch.equal(pos2, ref_pos))
    check(f"{tag}: second (hit) reorder bitwise", torch.equal(idx2, ref_idx))
    check(f"{tag}: builder ran exactly once", counter[0] == 1)
    check(f"{tag}: hit returns identical objects", pos1 is pos2 and idx1 is idx2)

    # safety: different tensor objects (equal values) must miss and recompute
    cu_q2 = cu_q.clone()
    cu_kv2 = cu_kv.clone()
    counter2 = [0]
    build2 = _builder_calls(cu_q2, cu_kv2, cp_size, cp_rank, device, out_q, out_g, cover_q, cover_kv, counter2)
    pos3, idx3 = dsa._cached_cp_layout(carrier, key, cu_q2, cu_kv2, build2)
    check(f"{tag}: new tensor objects miss (recompute)", counter2[0] == 1)
    check(f"{tag}: recomputed values still bitwise", torch.equal(pos3, ref_pos) and torch.equal(idx3, ref_idx))

    # safety: different size key misses
    counter3 = [0]
    build3 = _builder_calls(cu_q, cu_kv, cp_size, cp_rank, device, out_q, out_g, cover_q, cover_kv, counter3)
    dsa._cached_cp_layout(carrier, key + ("alt",), cu_q, cu_kv, build3)
    check(f"{tag}: different key misses (recompute)", counter3[0] == 1)


def main():
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        for cp_size in (2, 4, 16):
            for cp_rank in (0, cp_size // 2, cp_size - 1):
                # doc lengths must be divisible by 2*cp_size (builder CPU guard)
                base = 2 * cp_size
                doc_lens = [base * 4, base * 2, base * 6]
                _run_case(device, cp_size, cp_rank, doc_lens, pad_tail=0, cover=(True, True), seed=cp_size * 100 + cp_rank)
                # with a padding tail not covered by cu_seqlens
                _run_case(device, cp_size, cp_rank, doc_lens, pad_tail=base, cover=(False, False), seed=cp_size * 200 + cp_rank)
        # empty doc + single doc + many docs (cp=4)
        _run_case(device, 4, 1, [16, 0, 24, 8], pad_tail=0, cover=(True, True), seed=999)
        _run_case(device, 4, 2, [32], pad_tail=0, cover=(True, True), seed=998)
        _run_case(device, 4, 0, [8] * 12, pad_tail=8, cover=(False, False), seed=997)

    # gate reader sanity
    os.environ.pop("BT_DSA_CP_LAYOUT_CACHE", None)
    check("gate default off", dsa._cp_layout_cache_enabled() is False)
    os.environ["BT_DSA_CP_LAYOUT_CACHE"] = "1"
    try:
        check("gate on reads 1", dsa._cp_layout_cache_enabled() is True)
    finally:
        os.environ.pop("BT_DSA_CP_LAYOUT_CACHE", None)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES: {FAILURES}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
