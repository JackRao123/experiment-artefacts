#!/usr/bin/env python3
"""Parity test for FIX A — BT_DSA_BWD_ASYNC_NONEMPTY (DSA sparse-attn backward).

Verifies the env-gated async all-rows-nonempty probe against the unpatched
syncing `torch.nonzero(topk_length > 0)` path:

  1. arange substitution is bitwise identical to nonzero when all rows are
     non-empty (the fast path's core claim);
  2. `_run_sparse_attention_backward` with nonempty_rows_verified=True vs False
     produces bitwise-identical grad_query / grad_kv_full when all rows are
     non-empty, through the SAME compaction dataflow (dummy row, index_selects);
  3. FORCED-FALLBACK case (REVIEW_FIXA §7 binding): rows with topk_length == 0
     take the original nonzero path and exclude exactly the zeroed rows;
  4. probe gating: gate off / CPU tensor / no-grad / missing probe all fall
     back to the original path;
  5. (GPU only) the real async kick: side-stream D2H + event, pinned flag
     correct for both all-nonempty and with-empty-rows inputs;
  6. (GPU only, optional) end-to-end FusedSparseAttentionFunc fwd+bwd parity
     with gate OFF vs ON — uses the real cuDNN/FlashMLA wrappers when present,
     else deterministic fakes.

Runs on Mac CPU (sections 1-4 + fake-wrapper function test) and on the box GPU.
Target the vendored mcore with BT_TEST_MCORE_PATH if the default walk-up does
not find it (e.g. when this file is copied elsewhere).

Usage: python test_dsa_bwd_async_nonempty_parity.py
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

import megatron.core.transformer.experimental_attention_variant.dsa_cudnn_kernels as dk  # noqa: E402

GATE = "BT_DSA_BWD_ASYNC_NONEMPTY"
FAILURES = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name)
    if not cond:
        FAILURES.append(name)


class _FakeDsaBackend:
    """Deterministic stand-in for the cuDNN DSA namespace (backward only)."""

    def __init__(self):
        self.calls = []

    def sparse_attention_backward_wrapper(
        self, q, kv, out, dO, lse, sink, idxs, *, softmax_scale, topk_length
    ):
        self.calls.append(
            {"q_rows": q.size(0), "topk_length": topk_length.clone(), "idxs_rows": idxs.size(0)}
        )
        dq = q * 0.5 + dO * 0.25 + out * 0.125
        dkv = kv * 0.75
        return {"dq": dq, "dkv": dkv}


def _make_bwd_inputs(N, H, D, DV, C, K, zero_rows=(), seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(N, H, D, generator=g)
    kv = torch.randn(N + 3, C, generator=g)  # skv*b rows
    sink = torch.full((H,), float("-inf"))
    idxs = torch.randint(0, N + 3, (N, K), generator=g)
    out = torch.randn(N, H, DV, generator=g)
    lse = torch.randn(N, H, generator=g)
    tlen = torch.randint(1, K + 1, (N,), generator=g, dtype=torch.int32)
    for r in zero_rows:
        tlen[r] = 0
    dO = torch.randn(N, H, DV, generator=g)
    args = dict(
        q_flat=q,
        kv_flat=kv,
        attn_sink=sink,
        global_idxs=idxs,
        out_flat=out,
        lse=lse,
        topk_length=tlen.to(device),
        softmax_scale=0.25,
        sq=N,
        b=1,
        num_heads=H,
        d=D,
        skv=N + 3,
        grad_output=dO,
    )
    return args


def _run_bwd(args, verified, fake):
    old_ns, old_ensure = dk._cudnn_dsa, dk._ensure_dsa_namespace
    dk._cudnn_dsa, dk._ensure_dsa_namespace = fake, (lambda: None)
    try:
        return dk._run_sparse_attention_backward(
            **args, nonempty_rows_verified=verified
        )
    finally:
        dk._cudnn_dsa, dk._ensure_dsa_namespace = old_ns, old_ensure


def main():
    torch.manual_seed(0)

    # 1. arange == nonzero when all rows non-empty (bitwise, the fast-path claim)
    for N in (1, 7, 128, 16384):
        tlen = torch.randint(1, 5, (N,), dtype=torch.int32)
        ref = torch.nonzero(tlen > 0, as_tuple=False).flatten()
        fast = torch.arange(N, dtype=torch.int64)
        check(f"arange==nonzero all-nonempty N={N}", torch.equal(ref, fast))

    # 2. backward parity, all rows non-empty: verified=True vs False bitwise,
    #    and the compaction dataflow is preserved (wrapper sees N+1 rows).
    args = _make_bwd_inputs(96, 4, 8, 8, 16, 5)
    fake_a, fake_b = _FakeDsaBackend(), _FakeDsaBackend()
    dq_ref, dkv_ref = _run_bwd(args, False, fake_a)
    dq_fast, dkv_fast = _run_bwd(args, True, fake_b)
    check("bwd all-nonempty: grad_query bitwise", torch.equal(dq_ref, dq_fast))
    check("bwd all-nonempty: grad_kv bitwise", torch.equal(dkv_ref, dkv_fast))
    check(
        "bwd all-nonempty: compaction dataflow preserved (N+1 rows both)",
        fake_a.calls[0]["q_rows"] == 97 and fake_b.calls[0]["q_rows"] == 97,
    )
    check(
        "bwd all-nonempty: wrapper topk_length identical",
        torch.equal(fake_a.calls[0]["topk_length"], fake_b.calls[0]["topk_length"]),
    )

    # 3. FORCED FALLBACK: zeroed rows -> original nonzero path, excludes exactly
    #    the zeroed rows, and matches an explicit nonzero reference run.
    zeroed = (3, 17, 42)
    args = _make_bwd_inputs(64, 4, 8, 8, 16, 5, zero_rows=zeroed, seed=7)
    fake_c, fake_d = _FakeDsaBackend(), _FakeDsaBackend()
    dq_fb, dkv_fb = _run_bwd(args, False, fake_c)  # what the patch runs on flag=False
    dq_ref2, dkv_ref2 = _run_bwd(args, False, fake_d)
    check("forced-fallback: deterministic nonzero path (grad_query)", torch.equal(dq_fb, dq_ref2))
    check("forced-fallback: deterministic nonzero path (grad_kv)", torch.equal(dkv_fb, dkv_ref2))
    z = torch.tensor(list(zeroed))
    check(
        "forced-fallback: zeroed rows get zero grad",
        dq_fb[z].abs().sum().item() == 0.0,
    )
    expected_valid = torch.tensor(
        [i for i in range(64) if i not in zeroed], dtype=torch.int64
    )
    check(
        "forced-fallback: wrapper received N-len(zeroed)+1 rows",
        fake_c.calls[0]["q_rows"] == 64 - len(zeroed) + 1,
    )
    # the nonzero selection equals the expected valid set
    tlen = args["topk_length"]
    check(
        "forced-fallback: nonzero selection == expected valid rows",
        torch.equal(torch.nonzero(tlen > 0, as_tuple=False).flatten(), expected_valid),
    )

    # 4. gating: kick returns None when gate off / CPU tensor. NOTE: grad mode
    #    deliberately does NOT gate the kick — PyTorch runs every autograd
    #    Function.forward under no-grad, so an is_grad_enabled() check is always
    #    False inside FusedSparseAttentionFunc.forward (the v1 inertness bug).
    #    The discarded no-grad first pass kicks too (harmless: ctx dropped).
    os.environ.pop(GATE, None)
    check("gate off: no kick", dk._maybe_kick_nonempty_probe(torch.ones(8, dtype=torch.int32)) is None)
    os.environ[GATE] = "1"
    try:
        check(
            "gate on, CPU tensor: no kick",
            dk._maybe_kick_nonempty_probe(torch.ones(8, dtype=torch.int32)) is None,
        )
        with torch.no_grad():
            check(
                "gate on, no-grad + CPU tensor: no kick (is_cuda, not grad mode)",
                dk._maybe_kick_nonempty_probe(torch.ones(8, dtype=torch.int32)) is None,
            )
        check(
            "read probe: missing attr -> False (original path)",
            dk._read_nonempty_probe(object()) is False,
        )
        # loud inertness: gated backward read with no probe counts probes_missing
        before = dict(dk._NONEMPTY_PROBE_STATS)
        dk._read_nonempty_probe(object())
        check(
            "read probe: missing probe with gate on counts probes_missing",
            dk._NONEMPTY_PROBE_STATS["probes_missing"] == before["probes_missing"] + 1,
        )
        # regression guard for the v1 inertness bug (CPU-checkable): the kick
        # must NOT gate on grad mode — inside autograd Function.forward,
        # is_grad_enabled() is always False, so such a gate disables the probe
        # unconditionally (the replay's ctx is the one backward reads).
        import inspect

        kick_src = inspect.getsource(dk._maybe_kick_nonempty_probe)
        check(
            "kick has no is_grad_enabled gate (v1 inertness regression guard)",
            "if not torch.is_grad_enabled()" not in kick_src
            and "if torch.is_grad_enabled()" not in kick_src,
        )
    finally:
        os.environ.pop(GATE, None)

    # 5. function-level parity through FusedSparseAttentionFunc with fakes
    #    (CPU: gate-on degenerates to the original path since no CUDA probe;
    #    GPU: exercises the real ctx stash + event read).
    _function_level_test()

    # 6. REAL call context: the Function under torch.utils.checkpoint (no-grad
    #    first pass + enable_grad replay) — the exact structure in which v1's
    #    is_grad_enabled gate made the patch inert. Liveness assertions are
    #    GPU-only; bitwise parity runs everywhere. This test FAILS on v1.
    _checkpoint_replay_test()

    # 7. GPU-only: real async kick semantics
    if torch.cuda.is_available():
        _gpu_probe_test()
    else:
        print("SKIP - GPU probe test (no CUDA)")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES: {FAILURES}")
        sys.exit(1)
    print("ALL PASS")


import contextlib as _contextlib


class _NvtxStub:
    @staticmethod
    def range(*a, **k):
        return _contextlib.nullcontext()


@_contextlib.contextmanager
def _fakes_installed():
    """Deterministic fakes for the cuDNN/FlashMLA wrappers + CPU-safe stubs."""
    fake_dsa = _FakeDsaBackend()

    def fake_fwd(q, kv_3d, indices, softmax_scale, *, d_v, attn_sink, topk_length, indexer_topk):
        out = q[..., :d_v] * 0.5 + 0.1
        lse = torch.randn(q.size(0), q.size(1), device=q.device)
        return out, None, lse

    saved = (
        dk._cudnn_dsa,
        dk._ensure_dsa_namespace,
        dk._flash_mla_sparse_fwd,
        getattr(dk, "_ensure_flash_mla", None),
        dk._get_topk_alignment,
        dk._get_head_padding,
    )
    dk._cudnn_dsa = fake_dsa
    dk._ensure_dsa_namespace = lambda: None
    dk._flash_mla_sparse_fwd = fake_fwd
    dk._get_topk_alignment = lambda: 1
    dk._get_head_padding = lambda h: h
    if hasattr(dk, "_ensure_flash_mla"):
        dk._ensure_flash_mla = lambda: None
    old_nvtx = torch.cuda.nvtx
    torch.cuda.nvtx = _NvtxStub
    try:
        yield fake_dsa
    finally:
        (
            dk._cudnn_dsa,
            dk._ensure_dsa_namespace,
            dk._flash_mla_sparse_fwd,
            _ensure,
            dk._get_topk_alignment,
            dk._get_head_padding,
        ) = saved
        if _ensure is not None:
            dk._ensure_flash_mla = _ensure
        torch.cuda.nvtx = old_nvtx


def _mk_inputs(N=48, H=4, D=8, DV=8, C=16, K=5, seed=3, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    query = torch.randn(N, 1, H, D, generator=g)
    kv = torch.randn(N + 3, 1, C, generator=g)
    topk_indices = torch.randint(0, N + 3, (1, N, K), generator=g)
    topk_length = torch.randint(1, K + 1, (1, N), generator=g, dtype=torch.int32)
    return (
        query.to(device),
        kv.to(device),
        topk_indices.to(device),
        topk_length.to(device),
    )


def _function_level_test():
    query, kv, topk_indices, topk_length = _mk_inputs()
    with _fakes_installed():
        grads = {}
        for gate in ("0", "1"):
            os.environ[GATE] = gate
            q_in = query.clone().requires_grad_(True)
            kv_in = kv.clone().requires_grad_(True)
            out = dk.FusedSparseAttentionFunc.apply(
                q_in, kv_in, topk_indices, 0.25, 8, topk_length
            )
            out.backward(torch.randn_like(out, generator=torch.Generator().manual_seed(11)))
            grads[gate] = (q_in.grad.clone(), kv_in.grad.clone())
        os.environ.pop(GATE, None)
    check("function-level: grad_query bitwise gate off vs on", torch.equal(grads["0"][0], grads["1"][0]))
    check("function-level: grad_kv bitwise gate off vs on", torch.equal(grads["0"][1], grads["1"][1]))


def _checkpoint_replay_test():
    """The REAL call context: FusedSparseAttentionFunc inside
    torch.utils.checkpoint (no-grad first pass + enable_grad replay), matching
    mcore full recompute (tensor_parallel/random.py:580/:620).

    Regression coverage for the v1 inertness bug: v1 gated the kick on
    torch.is_grad_enabled() inside Function.forward, where PyTorch forces
    no-grad — so the probe never fired. A direct-call probe test could not see
    this; only a test through the autograd Function + checkpoint replay can.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query, kv, topk_indices, topk_length = _mk_inputs(device=device)
    grad_out = torch.randn(48, 1, 4, 8, generator=torch.Generator().manual_seed(11)).to(device)

    kicks = []
    real_kick = dk._maybe_kick_nonempty_probe

    def _spy(t):
        r = real_kick(t)
        kicks.append(r is not None)
        return r

    with _fakes_installed():
        dk._maybe_kick_nonempty_probe = _spy
        try:
            grads = {}
            for gate in ("0", "1"):
                os.environ[GATE] = gate
                stats_before = dict(dk._NONEMPTY_PROBE_STATS)
                kicks.clear()
                q_in = query.clone().requires_grad_(True)
                kv_in = kv.clone().requires_grad_(True)

                def run(q, k):
                    return dk.FusedSparseAttentionFunc.apply(
                        q, k, topk_indices, 0.25, 8, topk_length
                    )

                out = torch.utils.checkpoint.checkpoint(run, q_in, kv_in, use_reentrant=False)
                out.backward(grad_out)
                grads[gate] = (q_in.grad.clone(), kv_in.grad.clone())
                if gate == "1" and device == "cuda":
                    check(
                        "replay context: probe kicked in both passes (first + replay)",
                        len(kicks) == 2 and all(kicks),
                    )
                    check(
                        "replay context: backward found the probe (probes_missing +0)",
                        dk._NONEMPTY_PROBE_STATS["probes_missing"]
                        == stats_before["probes_missing"],
                    )
                    check(
                        "replay context: backward read counted, no fallback",
                        dk._NONEMPTY_PROBE_STATS["reads"] == stats_before["reads"] + 1
                        and dk._NONEMPTY_PROBE_STATS["fallbacks"]
                        == stats_before["fallbacks"],
                    )
            check(
                "replay context: grad_query bitwise gate off vs on",
                torch.equal(grads["0"][0], grads["1"][0]),
            )
            check(
                "replay context: grad_kv bitwise gate off vs on",
                torch.equal(grads["0"][1], grads["1"][1]),
            )
        finally:
            dk._maybe_kick_nonempty_probe = real_kick
            os.environ.pop(GATE, None)
    if device == "cpu":
        print("SKIP - replay liveness assertions (no CUDA); bitwise parity ran")


def _gpu_probe_test():
    os.environ[GATE] = "1"
    try:
        # all non-empty -> flag True
        tlen = torch.randint(1, 5, (4096,), dtype=torch.int32, device="cuda")
        probe = dk._maybe_kick_nonempty_probe(tlen)
        check("gpu: probe created", probe is not None)
        event, pinned, flag_gpu = probe
        event.synchronize()
        check("gpu: all-nonempty flag True", bool(pinned.item()) is True)

        # forced empty row -> flag False
        tlen2 = tlen.clone()
        tlen2[123] = 0
        probe2 = dk._maybe_kick_nonempty_probe(tlen2)
        event2, pinned2, _ = probe2
        event2.synchronize()
        check("gpu: empty-row flag False (fallback would fire)", bool(pinned2.item()) is False)

        # ctx read path with a real probe
        ctx = types.SimpleNamespace(dsa_nonempty_probe=probe)
        check("gpu: _read_nonempty_probe True", dk._read_nonempty_probe(ctx) is True)
        ctx2 = types.SimpleNamespace(dsa_nonempty_probe=probe2)
        check("gpu: _read_nonempty_probe False", dk._read_nonempty_probe(ctx2) is False)

        # v2 semantics: the discarded no-grad first pass ALSO kicks (harmless —
        # its ctx is dropped). v1 gated on is_grad_enabled() and never fired.
        with torch.no_grad():
            probe3 = dk._maybe_kick_nonempty_probe(tlen)
        check(
            "gpu: no-grad pass also kicks (v2; v1 bug would return None)",
            probe3 is not None,
        )
        if probe3 is not None:
            probe3[0].synchronize()
    finally:
        os.environ.pop(GATE, None)


if __name__ == "__main__":
    main()
