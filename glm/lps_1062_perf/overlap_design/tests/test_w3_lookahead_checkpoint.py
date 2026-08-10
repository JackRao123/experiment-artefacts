#!/usr/bin/env python3
"""CPU tests for W3 — BT_MOE_LOOKAHEAD_RECOMPUTE (lookahead recompute).

Verifies `megatron/core/lookahead_checkpoint.py` against mcore's status-quo
`CheckpointFunction` (tensor_parallel.checkpoint), on Mac CPU:

  1. BITWISE PARITY WITH DROPOUT=0.5 (the RNG-isolation mechanism proof
     requested by fibonacci): a 3-chunk chain (Linear -> dropout(0.5) ->
     Linear per chunk) under lookahead checkpointing must produce
     bitwise-identical forward outputs AND input grads vs the unkicked
     status-quo order. With dropout>0, any leak in the kick's RNG
     fork/set/restore (kicked recompute(L-1) interleaved between
     recompute(L) and bwd(L)) changes a dropout mask somewhere and flips
     this comparison. (Production runs dropout=0 and the gate asserts it;
     this proves the mechanism, not the prod case.)
  2. kick/skip coverage via the telemetry counters: 3 chunks => kicks=2,
     stash_hits=2, stash_misses=1 (the LAST chunk is never kick-targeted and
     must recompute inline), fallbacks=0, sweeps=0.
  3. eviction discipline: every entry popped on consume; registry empty at
     microbatch end; no sweep fired; a second microbatch on a FRESH carrier
     starts clean (no cross-datum stash leak — the +2-4GiB/datum leak
     fibonacci flagged).
  4. fallback: carrier=None degrades to status-quo inline recompute
     (bitwise-equal to the reference) with fallbacks counted loudly.
  5. RNG round-trip of the save/restore helpers the Function relies on.

The real RNG helpers in tensor_parallel.random touch CUDA RNG state, so the
test monkeypatches CPU equivalents (get/set torch's CPU generator state) —
the fork/restore LOGIC under test is identical; the CUDA path is exercised
by the on-box canary.

Usage: python test_w3_lookahead_checkpoint.py
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
    p = os.path.expanduser(
        "~/Documents/trainers/server/vendor/megatron-bridge/3rdparty/Megatron-LM"
    )
    if os.path.isdir(p):
        return p
    raise RuntimeError("vendored Megatron-LM not found; set BT_TEST_MCORE_PATH")


sys.path.insert(0, _find_mcore())
_stub = types.ModuleType("megatron.core.transformer.moe.ops.paged_stash")
_stub.GLOBAL_BLOCK_SIZE = 128
_stub.paged_stash_copy_kernel = None
_stub.paged_stash_pop_kernel = None
sys.modules.setdefault("megatron.core.transformer.moe.ops.paged_stash", _stub)

import torch  # noqa: E402

import megatron.core.tensor_parallel.random as tp_random  # noqa: E402
from megatron.core.tensor_parallel import checkpoint as tp_checkpoint  # noqa: E402
import megatron.core.lookahead_checkpoint as lc  # noqa: E402

FAILURES = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name, flush=True)
    if not cond:
        FAILURES.append(name)


# --- CPU RNG stubs (the real helpers touch CUDA RNG state) ---
def _cpu_get_all_rng_states():
    return (torch.get_rng_state(),)


def _cpu_set_all_rng_states(cpu_state):
    torch.set_rng_state(cpu_state)


tp_random._get_all_rng_states = _cpu_get_all_rng_states
tp_random._set_all_rng_states = _cpu_set_all_rng_states
# _fork_rng composes the two patched helpers, so it needs no patch itself.

H = 8
N_CHUNKS = 3


def make_chunk(seed, p):
    g = torch.Generator().manual_seed(seed)
    w1 = torch.randn(2 * H, H, generator=g) / H**0.5
    w2 = torch.randn(H, 2 * H, generator=g) / H**0.5

    def chunk_fn(x):
        h = torch.nn.functional.linear(x, w1)
        h = torch.nn.functional.dropout(h, p=p, training=True)
        return torch.nn.functional.linear(h, w2)

    return chunk_fn


def run_chain(chunks, x, use_lookahead, carrier):
    """fwd+bwd through the checkpointed chunk chain; returns (out, x_leaf.grad)."""
    x_leaf = x.detach().requires_grad_(True)
    h = x_leaf
    for i, cf in enumerate(chunks):
        if use_lookahead:
            h = lc.lookahead_checkpoint(cf, False, i, carrier, h)
        else:
            h = tp_checkpoint(cf, False, h)
    loss = (h**2).sum()
    loss.backward()
    assert x_leaf.grad is not None, "no input grad produced (graph broken)"
    return h.detach(), x_leaf.grad.detach()


def fresh_stats():
    return dict(lc._LOOKAHEAD_STATS)


def test_1_bitwise_parity_dropout():
    """RNG-isolation mechanism proof: dropout=0.5, kicked vs unkicked."""
    p = 0.5
    for seed in (0, 1, 2):
        chunks = [make_chunk(100 + i, p) for i in range(N_CHUNKS)]
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(5, H, generator=g)

        os.environ["BT_MOE_LOOKAHEAD_RECOMPUTE"] = "1"
        carrier = types.SimpleNamespace()
        torch.manual_seed(777)  # identical forward RNG start for both runs
        out_la, grad_la = run_chain(chunks, x, True, carrier)

        torch.manual_seed(777)
        out_ref, grad_ref = run_chain(chunks, x, False, None)

        check(
            f"sec1 seed{seed}: lookahead outputs == status-quo outputs (bitwise)",
            torch.equal(out_la, out_ref),
        )
        check(
            f"sec1 seed{seed}: lookahead input grads == status-quo (bitwise)",
            torch.equal(grad_la, grad_ref),
        )


def test_2_kick_coverage_counters():
    """3 chunks => kicks=2, stash_hits=2, stash_misses=1 (last chunk inline)."""
    os.environ["BT_MOE_LOOKAHEAD_RECOMPUTE"] = "1"
    before = fresh_stats()
    chunks = [make_chunk(200 + i, 0.5) for i in range(N_CHUNKS)]
    x = torch.randn(5, H, generator=torch.Generator().manual_seed(5))
    run_chain(chunks, x, True, types.SimpleNamespace())
    after = fresh_stats()
    delta = {k: after[k] - before[k] for k in after}
    check("sec2: kicks == N-1", delta["kicks"] == N_CHUNKS - 1)
    check("sec2: stash_hits == N-1", delta["stash_hits"] == N_CHUNKS - 1)
    check("sec2: stash_misses == 1 (last chunk recomputes inline)", delta["stash_misses"] == 1)
    check("sec2: fallbacks == 0", delta["fallbacks"] == 0)
    check("sec2: sweeps == 0 (clean consume, nothing left)", delta["sweeps"] == 0)


def test_3_eviction():
    """Registry popped on consume + empty at microbatch end; fresh carrier clean."""
    os.environ["BT_MOE_LOOKAHEAD_RECOMPUTE"] = "1"
    carrier = types.SimpleNamespace()
    chunks = [make_chunk(300 + i, 0.5) for i in range(N_CHUNKS)]
    x = torch.randn(5, H, generator=torch.Generator().manual_seed(6))
    run_chain(chunks, x, True, carrier)
    registry = getattr(carrier, lc._REGISTRY_ATTR)
    check("sec3: registry order empty after backward", registry["order"] == [])
    check("sec3: registry chunks empty after backward", registry["chunks"] == {})

    # second microbatch on a FRESH carrier: no cross-datum residue, same result
    # as a first run on another fresh carrier (stash reuse would corrupt it)
    chunks2 = [make_chunk(300 + i, 0.5) for i in range(N_CHUNKS)]
    x2 = torch.randn(5, H, generator=torch.Generator().manual_seed(6))
    torch.manual_seed(888)
    out_a, grad_a = run_chain(chunks2, x2, True, types.SimpleNamespace())
    chunks3 = [make_chunk(300 + i, 0.5) for i in range(N_CHUNKS)]
    torch.manual_seed(888)
    out_b, grad_b = run_chain(chunks3, x2, True, types.SimpleNamespace())
    check("sec3: two fresh-carrier microbatches identical (outputs)", torch.equal(out_a, out_b))
    check("sec3: two fresh-carrier microbatches identical (grads)", torch.equal(grad_a, grad_b))


def test_4_fallback_no_carrier():
    """carrier=None degrades to status-quo inline recompute, loudly counted."""
    os.environ["BT_MOE_LOOKAHEAD_RECOMPUTE"] = "1"
    before = fresh_stats()
    chunks = [make_chunk(400 + i, 0.5) for i in range(N_CHUNKS)]
    x = torch.randn(5, H, generator=torch.Generator().manual_seed(7))
    torch.manual_seed(999)
    out_fb, grad_fb = run_chain(chunks, x, True, None)  # carrier=None
    torch.manual_seed(999)
    out_ref, grad_ref = run_chain(chunks, x, False, None)
    after = fresh_stats()
    delta = {k: after[k] - before[k] for k in after}
    check("sec4: fallback outputs == status-quo (bitwise)", torch.equal(out_fb, out_ref))
    check("sec4: fallback grads == status-quo (bitwise)", torch.equal(grad_fb, grad_ref))
    check("sec4: fallbacks counted", delta["fallbacks"] == N_CHUNKS)
    check("sec4: no kicks without a carrier", delta["kicks"] == 0)


def test_5_rng_roundtrip():
    """The save/restore semantics the Function relies on (CPU stub version)."""
    torch.manual_seed(12345)
    state = tp_random._get_all_rng_states()
    a = torch.randn(4)
    tp_random._set_all_rng_states(*state)
    b = torch.randn(4)
    check("sec5: RNG restore reproduces the sequence", torch.equal(a, b))


def test_6_gate_telemetry():
    old = os.environ.pop("BT_MOE_LOOKAHEAD_RECOMPUTE", None)
    try:
        check("sec6: gate off by default", lc._lookahead_enabled() is False)
        os.environ["BT_MOE_LOOKAHEAD_RECOMPUTE"] = "1"
        check("sec6: gate on with env=1", lc._lookahead_enabled() is True)
    finally:
        if old is None:
            os.environ.pop("BT_MOE_LOOKAHEAD_RECOMPUTE", None)
        else:
            os.environ["BT_MOE_LOOKAHEAD_RECOMPUTE"] = old


def main():
    test_6_gate_telemetry()
    test_5_rng_roundtrip()
    test_1_bitwise_parity_dropout()
    test_2_kick_coverage_counters()
    test_3_eviction()
    test_4_fallback_no_carrier()
    if FAILURES:
        raise SystemExit(f"{len(FAILURES)} FAILURES: {FAILURES}")
    print("OK — all W3 lookahead-checkpoint CPU checks passed")


if __name__ == "__main__":
    main()
