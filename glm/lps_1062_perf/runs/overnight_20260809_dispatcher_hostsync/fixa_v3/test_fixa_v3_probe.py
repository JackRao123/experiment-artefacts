#!/usr/bin/env python3
"""Mac-CPU test for FIX A v3 — BT_DSA_BWD_ASYNC_NONEMPTY_V3 (first-pass-anchored
DSA-bwd nonempty probe).

Drives the REAL v3 code (dsa_cudnn_kernels.py helpers, the FusedSparseAttentionFunc
plumbing, the dsa_kernels pass-through) through the REAL checkpoint machinery
(megatron.core.tensor_parallel.checkpoint -> CheckpointFunction) on Mac CPU:

  1. gate OFF: code inert — no kick, no stash, no fetch, counters zero, the
     fused call sees no probe kwarg.
  2. first-pass kick + stash / replay fetch through the REAL CheckpointFunction:
     the kick fires exactly once (no-grad first pass), the replay pops the
     IDENTICAL probe object (attr survival), counters {kicks:1, hits:1,
     misses:0}. This is the v1-lesson guard: the discrimination lives in
     _v3_probe_for_pass (a plain function), and this test fails if the kick
     can silently never fire.
  3. plumbing: the probe flows run_fused_absorbed_sparse_attention ->
     FusedSparseAttentionFunc.apply -> ctx -> backward's _read_nonempty_probe_v3
     (real read of the fake probe: event synced, pinned value True ->
     nonempty_rows_verified=True), backward return arity grows by exactly one
     None for the opaque probe input.
  4. dsa_kernels conditional pass: the probe kwarg reaches the backend hook
     only when present (unknown backends see a byte-identical call otherwise).
  5. carrier disciplines: pop is one-shot (second fetch misses), wrong layer
     key misses, empty stash misses, eval pass (no-grad, no checkpoint) kicks
     but never fetches, non-recompute training (grad on, no checkpoint)
     fetches-and-misses loudly.
  6. verify mode: matching fresh flag passes (verifies+=1); a corrupted
     stashed flag raises RuntimeError inside the replay forward.
  7. source guards: the kick helper has no is_grad_enabled gate (the v1 trap —
     discrimination lives at the dsa.py level), and dsa.py's call site uses
     _v3_probe_for_pass.

The probe's CUDA mechanics (side stream, pinned copy) are v2-identical and
GPU-only; on CPU the kick is monkeypatched with a fake probe where noted —
everything else (pass discrimination, carrier stash/fetch, plumbing, autograd
arity, verify, counters) is the real vendored code.

Runs on Mac CPU. BT_TEST_MCORE_PATH overrides the tree walk-up.

Usage: python test_fixa_v3_probe.py
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
    for _ in range(10):
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

import megatron.core.tensor_parallel.random as tp_random  # noqa: E402
import megatron.core.transformer.experimental_attention_variant.dsa_cudnn_kernels as dk  # noqa: E402
import megatron.core.transformer.experimental_attention_variant.dsa_kernels as dkm  # noqa: E402
from megatron.core import tensor_parallel  # noqa: E402
from megatron.core.packed_seq_params import PackedSeqParams  # noqa: E402

FAILURES = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name)
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Fakes / stubs
# ---------------------------------------------------------------------------


class _FakeEvent:
    def __init__(self, spy):
        self.spy = spy

    def synchronize(self):
        self.spy["event_syncs"] += 1


class _FakePinned:
    def __init__(self, val):
        self.val = val

    def item(self):
        return self.val


def _fake_probe(spy, flag=True):
    return dk._NonemptyProbeV3(_FakeEvent(spy), _FakePinned(flag), None)


def _install_rng_stubs():
    """CPU-only RNG state for the real CheckpointFunction (CUDA RNG absent)."""
    tp_random._get_all_rng_states = lambda: (torch.get_rng_state(), None, None)

    def _set_all(cpu_state, cuda_state, tracker_state):
        torch.set_rng_state(cpu_state)

    tp_random._set_all_rng_states = _set_all


def _reset_v3_state():
    dk._V3_GATE_LOGGED[0] = False
    dk._V3_VERIFY_GATE_LOGGED[0] = False
    dk._V3_ARMED_LOGGED[0] = False
    dk._V3_HIT_LOGGED[0] = False
    for k in dk._V3_STATS:
        dk._V3_STATS[k] = 0
    dk._V3_WINDOW[0] = 0
    dk._V3_WINDOW[1] = 0


def _fresh_spy():
    return {"kicks": 0, "event_syncs": 0}


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def case_gate_off_inert():
    print("\n== case 1: gate OFF — code inert ==")
    os.environ.pop("BT_DSA_BWD_ASYNC_NONEMPTY_V3", None)
    _reset_v3_state()
    psp = PackedSeqParams()
    t = torch.ones(8, dtype=torch.int32)
    check("gate off: _v3_probe_for_pass returns None in both pass kinds",
          dk._v3_probe_for_pass(psp, 3, t) is None)
    with torch.enable_grad():
        check("gate off: replay-flavored call also None", dk._v3_probe_for_pass(psp, 3, t) is None)
    check("gate off: no stash on the carrier", not hasattr(psp, dk._V3_STASH_ATTR))
    check("gate off: all counters zero", all(v == 0 for v in dk._V3_STATS.values()))
    check("gate off: real kick returns None (gate check precedes the is_cuda check)",
          dk._maybe_kick_nonempty_probe_v3(t) is None)


def case_checkpoint_kick_fetch():
    print("\n== case 2: first-pass kick + stash / replay fetch through the real CheckpointFunction ==")
    os.environ["BT_DSA_BWD_ASYNC_NONEMPTY_V3"] = "1"
    _reset_v3_state()
    spy = _fresh_spy()
    real_kick = dk._maybe_kick_nonempty_probe_v3
    kicked = []

    def fake_kick(t):
        spy["kicks"] += 1
        dk._V3_STATS["kicks"] += 1  # mirror the real kick's counter
        p = _fake_probe(spy)
        kicked.append(p)
        return p

    dk._maybe_kick_nonempty_probe_v3 = fake_kick
    psp = PackedSeqParams()
    seen = []
    topk_length = torch.ones(16, dtype=torch.int32)

    def layer_fn(h):
        probe = dk._v3_probe_for_pass(psp, 7, topk_length)
        seen.append(probe)
        return h * 2

    try:
        hidden = torch.randn(8, 4, requires_grad=True)
        out = tensor_parallel.checkpoint(layer_fn, False, hidden)
        out.sum().backward()
    finally:
        dk._maybe_kick_nonempty_probe_v3 = real_kick

    check("kick fired exactly once (no-grad first pass only)", spy["kicks"] == 1)
    check("first pass saw no probe (returns None)", seen[0] is None)
    check("replay fetched a probe", seen[1] is not None)
    check(
        "replay's probe IS the first-pass object (attr survival on the carrier)",
        len(kicked) == 1 and seen[1] is kicked[0],
    )
    check("counters: 1 kick, 1 hit, 0 misses",
          dk._V3_STATS["kicks"] == 1 and dk._V3_STATS["hits"] == 1 and dk._V3_STATS["misses"] == 0)
    check("stash empty after the pop (one-shot eviction)",
          not getattr(psp, dk._V3_STASH_ATTR, {}))


def case_plumbing_through_function():
    print("\n== case 3: plumbing through FusedSparseAttentionFunc (probe -> ctx -> backward read) ==")
    os.environ["BT_DSA_BWD_ASYNC_NONEMPTY_V3"] = "1"
    _reset_v3_state()
    spy = _fresh_spy()

    sq, b, nh, d, dv, skv, kdim, topk = 4, 1, 2, 8, 8, 6, 8, 3
    query = torch.randn(sq, b, nh, d, requires_grad=True)
    key = torch.randn(skv, b, 1, kdim)
    topk_indices = torch.zeros(b, sq, topk, dtype=torch.int64)
    topk_length = torch.ones(b, sq, dtype=torch.int32)

    recorded = {}

    def fake_fwd(query, kv_full, topk_indices, softmax_scale, d_v, topk_length=None,
                 _nonempty_probe_out=None):
        q_flat = query.reshape(sq * b, nh, d)
        out_flat = q_flat[..., :d_v] * 0.5
        lse = torch.zeros(sq * b, nh)
        attn_sink = torch.zeros(nh)
        global_idxs = torch.zeros(sq * b, topk, dtype=torch.int64)
        tlf = torch.ones(sq * b, dtype=torch.int32)
        return out_flat, lse, q_flat, kv_full, attn_sink, global_idxs, tlf

    def fake_bwd(**kw):
        recorded["nonempty_rows_verified"] = kw["nonempty_rows_verified"]
        grad_query = torch.ones(sq, b, nh, d)
        grad_kv = torch.ones(skv, b, kdim)
        return grad_query, grad_kv

    saved = (
        dk._run_sparse_attention_forward,
        dk._run_sparse_attention_backward,
        dk._supports_flash_mla_value_layout,
        dk._flash_mla_supports_head_count,
    )
    dk._run_sparse_attention_forward = fake_fwd
    dk._run_sparse_attention_backward = fake_bwd
    dk._supports_flash_mla_value_layout = lambda *a, **k: True
    dk._flash_mla_supports_head_count = lambda *a, **k: True
    probe = _fake_probe(spy, flag=True)
    try:
        out = dk.run_fused_absorbed_sparse_attention(
            query, key, topk_indices, 1.0, dv, topk_length, nonempty_probe_v3=probe
        )
        check("fused call returned an output (guards passed)", out is not None)
        out.sum().backward()
    finally:
        (dk._run_sparse_attention_forward, dk._run_sparse_attention_backward,
         dk._supports_flash_mla_value_layout, dk._flash_mla_supports_head_count) = saved

    check("backward read the first-pass probe (event synced once)", spy["event_syncs"] == 1)
    check("backward saw nonempty_rows_verified=True (arange path)",
          recorded.get("nonempty_rows_verified") is True)
    check("grad flowed to query", query.grad is not None and query.grad.abs().sum() > 0)
    check("one read, no drops/fallbacks",
          dk._V3_STATS["probe_dropped"] == 0 and dk._V3_STATS["fallbacks"] == 0)

    # arity: the Function's backward must return one extra None for the probe
    # input (2 real grads + 5 Nones = 7 values for 7 forward inputs)
    import inspect

    src = inspect.getsource(dk.FusedSparseAttentionFunc.backward)
    ret_line = [l for l in src.splitlines() if l.strip().startswith("return grad_query")][0]
    n_nones = ret_line.count("None")
    check("backward return arity grew by exactly one None (2 grads + 5 Nones)", n_nones == 5)


def case_hook_conditional_pass():
    print("\n== case 4: dsa_kernels conditional pass (kwarg only when present) ==")
    calls = []

    def spy_hook(*args, **kwargs):
        calls.append(kwargs)
        return "ok"

    saved = dkm._resolve_fused_hook
    dkm._resolve_fused_hook = lambda config, name: spy_hook
    try:
        dkm.run_fused_absorbed_sparse_attention(None, "q", "k", "ti", 1.0, 8, "tl")
        dkm.run_fused_absorbed_sparse_attention(None, "q", "k", "ti", 1.0, 8, "tl",
                                                nonempty_probe_v3="PROBE")
    finally:
        dkm._resolve_fused_hook = saved
    check("no probe: hook called without the kwarg", "nonempty_probe_v3" not in calls[0])
    check("with probe: hook received nonempty_probe_v3='PROBE'",
          calls[1].get("nonempty_probe_v3") == "PROBE")


def case_carrier_disciplines():
    print("\n== case 5: carrier disciplines (one-shot pop, wrong key, empty stash, eval, non-recompute) ==")
    os.environ["BT_DSA_BWD_ASYNC_NONEMPTY_V3"] = "1"
    _reset_v3_state()
    spy = _fresh_spy()
    psp = PackedSeqParams()
    probe = _fake_probe(spy)

    dk._v3_stash_probe(psp, 5, probe)
    check("stash/fetch round-trip returns the identical object",
          dk._v3_fetch_probe(psp, 5) is probe)
    check("pop is one-shot: second fetch misses", dk._v3_fetch_probe(psp, 5) is None)
    check("wrong layer key misses", dk._v3_fetch_probe(psp, 6) is None)
    check("empty stash on a fresh carrier misses", dk._v3_fetch_probe(PackedSeqParams(), 5) is None)
    check("counters: 1 hit, 3 misses",
          dk._V3_STATS["hits"] == 1 and dk._V3_STATS["misses"] == 3)

    # eval pass (no-grad, no checkpoint): kicks but never fetches — harmless.
    _reset_v3_state()
    real_kick = dk._maybe_kick_nonempty_probe_v3

    def fake_kick(t):
        spy["kicks"] += 1
        return _fake_probe(spy)

    dk._maybe_kick_nonempty_probe_v3 = fake_kick
    try:
        with torch.no_grad():
            r = dk._v3_probe_for_pass(psp, 9, torch.ones(4, dtype=torch.int32))
        check("eval pass: returns None (nothing to consume yet)", r is None)
        check("eval pass: kicked + stashed on the eval carrier (harmless)",
              spy["kicks"] == 1 and bool(getattr(psp, dk._V3_STASH_ATTR, {})))
        # non-recompute training (grad on, no checkpoint): fetch misses loudly
        r2 = dk._v3_probe_for_pass(PackedSeqParams(), 9, torch.ones(4, dtype=torch.int32))
        check("non-recompute training: fetch returns None (fallback)", r2 is None)
        check("non-recompute training: miss counted (loud)", dk._V3_STATS["misses"] == 1)
    finally:
        dk._maybe_kick_nonempty_probe_v3 = real_kick


def case_verify_mode():
    print("\n== case 6: verify mode — match passes, mismatch raises ==")
    os.environ["BT_DSA_BWD_ASYNC_NONEMPTY_V3"] = "1"
    os.environ["BT_DSA_BWD_ASYNC_NONEMPTY_V3_VERIFY"] = "1"
    _reset_v3_state()
    spy = _fresh_spy()

    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.dsa_nonempty_probe_v3 = _fake_probe(spy, flag=True)
    fresh_all_nonempty = torch.ones(8, dtype=torch.int32)
    dk._verify_nonempty_probe_v3(ctx, fresh_all_nonempty)
    check("verify: matching fresh flag passes", dk._V3_STATS["verifies"] == 1)

    ctx2 = _Ctx()
    ctx2.dsa_nonempty_probe_v3 = _fake_probe(spy, flag=False)  # corrupted stash
    raised = False
    try:
        dk._verify_nonempty_probe_v3(ctx2, fresh_all_nonempty)
    except RuntimeError as e:
        raised = "BT_DSA_BWD_ASYNC_NONEMPTY_V3_VERIFY" in str(e)
    check("verify: mismatched fresh flag raises RuntimeError", raised)
    os.environ.pop("BT_DSA_BWD_ASYNC_NONEMPTY_V3_VERIFY", None)


def case_source_guards():
    print("\n== case 7: source guards (v1 lesson) ==")
    import inspect

    kick_src = inspect.getsource(dk._maybe_kick_nonempty_probe_v3)
    check(
        "the kick helper has no is_grad_enabled gate (discrimination lives at "
        "the dsa.py level, a plain function — the v1 trap)",
        "is_grad_enabled" not in kick_src,
    )
    pass_src = inspect.getsource(dk._v3_probe_for_pass)
    check(
        "_v3_probe_for_pass discriminates via is_grad_enabled at the plain-function level",
        "is_grad_enabled" in pass_src,
    )
    import megatron.core.transformer.experimental_attention_variant.dsa as dsa_mod

    dsa_src = open(dsa_mod.__file__).read()
    check(
        "dsa.py's call site uses _v3_probe_for_pass",
        "_v3_probe_for_pass(" in dsa_src,
    )


def main():
    _install_rng_stubs()
    case_gate_off_inert()
    case_checkpoint_kick_fetch()
    case_plumbing_through_function()
    case_hook_conditional_pass()
    case_carrier_disciplines()
    case_verify_mode()
    case_source_guards()

    print("\n" + ("=" * 60))
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for name in FAILURES:
            print("  FAIL -", name)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
