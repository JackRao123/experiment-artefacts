#!/usr/bin/env python3
"""Parity test for FIX F — BT_THD_ROPE_HOST_CACHE (THD RoPE cu_seqlens host cache).

Verifies that reading per-sequence lengths/offsets from a cached host copy of
cu_seqlens is bitwise identical to the unpatched per-call `.tolist()`/`.item()`
device reads, through the real `_apply_rotary_pos_emb_thd`:

  1. gate OFF vs ON outputs bitwise equal (torch.equal) across cp sizes/ranks,
     multi-doc cu_seqlens, both freqs formats (packed-length CASE 1 and
     max-seqlen CASE 2), and both interleave modes;
  2. the host cache takes ONE D2H per unique tensor object (second call with
     the same tensor hits; a different tensor object misses);
  3. gate OFF behavior is byte-unchanged (reads the device tensor directly).

Runs on Mac CPU (the patched path degenerates to reading the same tensor —
`.cpu()` returns self — so the exact same arithmetic is exercised) and on the
box GPU (where the cache additionally eliminates real D2H syncs).

Usage: python test_thd_rope_host_cache_parity.py
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

import megatron.core.models.common.embeddings.rope_utils as ru  # noqa: E402

GATE = "BT_THD_ROPE_HOST_CACHE"
FAILURES = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name)
    if not cond:
        FAILURES.append(name)


class _FakeCpGroup:
    def __init__(self, size, rank):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _run_rope(t, cu_seqlens, freqs, cp_size, cp_rank, mla_interleaved):
    return ru._apply_rotary_pos_emb_thd(
        t,
        cu_seqlens,
        freqs,
        rotary_interleaved=False,
        mla_rotary_interleaved=mla_interleaved,
        mscale=1.0,
        cp_group=_FakeCpGroup(cp_size, cp_rank),
    )


def _case(device, cp_size, cp_rank, doc_lens, freqs_kind, mla_interleaved, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    total = sum(doc_lens)
    local = total // cp_size
    hd, d_rot = 2, 8
    t = torch.randn(local, hd, d_rot, generator=g, device=device)
    cu = torch.tensor([0] + list(torch.tensor(doc_lens).cumsum(0).tolist()), dtype=torch.int32, device=device)
    if freqs_kind == "packed":
        freqs = torch.randn(total, 1, 1, d_rot // 2, generator=g, device=device)
    else:  # max-seqlen freqs (CASE 2): per-sequence restart
        freqs = torch.randn(max(doc_lens), 1, 1, d_rot // 2, generator=g, device=device)

    os.environ.pop(GATE, None)
    out_ref = _run_rope(t, cu, freqs, cp_size, cp_rank, mla_interleaved)

    ru._THD_ROPE_HOST_CACHE.clear()
    os.environ[GATE] = "1"
    try:
        out_pat = _run_rope(t, cu, freqs, cp_size, cp_rank, mla_interleaved)
        # second call with the same tensor: cache hit, same result
        out_pat2 = _run_rope(t, cu, freqs, cp_size, cp_rank, mla_interleaved)
    finally:
        os.environ.pop(GATE, None)

    tag = f"dev={device} cp={cp_size} rank={cp_rank} docs={doc_lens} freqs={freqs_kind} mla={mla_interleaved}"
    check(f"{tag}: gate ON == OFF bitwise", torch.equal(out_ref, out_pat))
    check(f"{tag}: cache-hit call == reference", torch.equal(out_ref, out_pat2))
    check(f"{tag}: one cache entry for the tensor", len(ru._THD_ROPE_HOST_CACHE) == 1)
    entry = next(iter(ru._THD_ROPE_HOST_CACHE.values()))
    check(f"{tag}: cache holds strong ref to source", entry[0] is cu)

    # different tensor object (equal values) -> miss -> new entry, same result
    cu2 = cu.clone()
    os.environ[GATE] = "1"
    try:
        out_pat3 = _run_rope(t, cu2, freqs, cp_size, cp_rank, mla_interleaved)
    finally:
        os.environ.pop(GATE, None)
    check(f"{tag}: cloned tensor misses but bitwise equal", torch.equal(out_ref, out_pat3))
    check(f"{tag}: second tensor added a cache entry", len(ru._THD_ROPE_HOST_CACHE) == 2)


def _mutation_case(device, cp_size, cp_rank, seed):
    """In-place mutation of a CACHED cu_seqlens must miss, not serve stale."""
    g = torch.Generator(device=device).manual_seed(seed)
    base = 2 * cp_size
    doc_lens = [base * 3, base * 2, base * 4]
    total = sum(doc_lens)
    local = total // cp_size
    hd, d_rot = 2, 8
    t = torch.randn(local, hd, d_rot, generator=g, device=device)
    freqs = torch.randn(total, 1, 1, d_rot // 2, generator=g, device=device)
    cu = torch.tensor([0] + list(torch.tensor(doc_lens).cumsum(0).tolist()), dtype=torch.int32, device=device)

    ru._THD_ROPE_HOST_CACHE.clear()
    os.environ[GATE] = "1"
    try:
        out_before = _run_rope(t, cu, freqs, cp_size, cp_rank, True)
        ver_before = cu._version
        # in-place mutation: shift `base` tokens from the last doc to the first
        # (totals preserved so torch.split stays consistent)
        cu[1] += base
        cu[2] += base
        check("mutation bumped _version", cu._version > ver_before)
        out_after = _run_rope(t, cu, freqs, cp_size, cp_rank, True)
        entry = ru._THD_ROPE_HOST_CACHE.get(id(cu))
    finally:
        os.environ.pop(GATE, None)

    # fresh reference for the mutated tensor via the unpatched path
    out_after_ref = _run_rope(t, cu, freqs, cp_size, cp_rank, True)

    tag = f"dev={device} cp={cp_size} rank={cp_rank}"
    check(
        f"{tag}: in-place mutation misses stale cache (== fresh reference)",
        torch.equal(out_after, out_after_ref),
    )
    check(
        f"{tag}: mutation actually changed the output (test is sensitive)",
        not torch.equal(out_before, out_after),
    )
    check(
        f"{tag}: cache entry re-cached with new version",
        entry is not None and entry[0] is cu and entry[1] == cu._version,
    )


def main():
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        for cp_size in (2, 4, 16):
            for cp_rank in (0, cp_size - 1):
                base = 2 * cp_size
                for freqs_kind in ("packed", "maxlen"):
                    for mla in (False, True):
                        _case(
                            device,
                            cp_size,
                            cp_rank,
                            [base * 3, base * 1, base * 2],
                            freqs_kind,
                            mla,
                            seed=cp_size * 1000 + cp_rank * 10 + len(freqs_kind) + int(mla),
                        )
        # single doc + longer doc list
        _case(device, 4, 1, [64], "packed", True, seed=1)
        _case(device, 4, 3, [8, 16, 24, 32, 8], "maxlen", False, seed=2)
        # in-place mutation of a cached tensor must miss (sign-off hardening)
        _mutation_case(device, 4, 1, seed=31337)
        _mutation_case(device, 16, 5, seed=31338)

    # gate reader sanity
    os.environ.pop(GATE, None)
    check("gate default off", ru._thd_rope_host_cache_enabled() is False)
    os.environ[GATE] = "1"
    try:
        check("gate on reads 1", ru._thd_rope_host_cache_enabled() is True)
    finally:
        os.environ.pop(GATE, None)

    # FIFO eviction bound
    ru._THD_ROPE_HOST_CACHE.clear()
    os.environ[GATE] = "1"
    try:
        for i in range(ru._THD_ROPE_HOST_CACHE_MAX + 10):
            ru._host_cu_seqlens(torch.tensor([0, 8, 16], dtype=torch.int32))
        check(
            "cache FIFO-evicts at max size",
            len(ru._THD_ROPE_HOST_CACHE) == ru._THD_ROPE_HOST_CACHE_MAX,
        )
    finally:
        os.environ.pop(GATE, None)
        ru._THD_ROPE_HOST_CACHE.clear()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES: {FAILURES}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
