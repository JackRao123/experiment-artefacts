#!/usr/bin/env python3
"""Parity + mechanism test for FIX C — BT_MOE_DISPATCH_REPLAY_CACHE.

Drives a tiny but REAL MoEAlltoAllTokenDispatcher through the REAL checkpoint
machinery (megatron.core.tensor_parallel.checkpoint -> CheckpointFunction,
plus the real pass-marker from megatron.core.recompute) on Mac CPU:

  1. gate OFF: code inert — no wrap, no stores/hits, all-gather + D2H + event
     sync run in BOTH the first pass and the replay; outputs recorded as the
     bitwise reference.
  2. gate ON: the replay hits the cache — the tp_ep all-gather, the D2H event
     record, and d2h_event.synchronize() each run exactly once (first pass
     only); the splits handed to all_to_all in the replay are bitwise the
     first pass's; outputs bitwise-equal the gate-off reference.
  3. four microbatches in flight x two dispatcher instances (layers): every
     replay consumes ITS OWN microbatch's entry (carrier-keyed, not ordered).
  4. an eval forward (no checkpoint frame) between fwd and bwd does not store
     and does not disturb the later replay hit.
  5. forced fallbacks: replay with no stored entry (key miss) and replay with
     a changed routing shape both fall back to the full recompute path with
     the counters incremented.
  6. verify mode (BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY=1): the replay
     recomputes at full cost and asserts bitwise equality (passes clean); a
     corrupted cache entry raises RuntimeError.
  7. v1-lesson guards: the dispatcher never reads torch.is_grad_enabled (the
     grad-mode read lives in the recompute marker, a plain callable executed
     BY the checkpoint, not an autograd.Function), and the end-to-end hit
     assertions above fail if the gate can silently never fire.

Runs on Mac CPU. CUDA / collectives / RNG-tracker are faked at the edges
(marked below); everything in between — the dispatcher, the checkpoint
Function, the pass marker, permute/unpermute/sort_chunks — is the real
patched code from the vendored tree.

Usage: python test_fixc_replay_cache.py
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

import megatron.core.recompute as recompute  # noqa: E402
import megatron.core.tensor_parallel.random as tp_random  # noqa: E402
import megatron.core.transformer.moe.token_dispatcher as td  # noqa: E402
from megatron.core import tensor_parallel  # noqa: E402
from megatron.core.packed_seq_params import PackedSeqParams  # noqa: E402

FAILURES = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name)
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# CPU fakes for the CUDA / distributed edges (and ONLY the edges)
# ---------------------------------------------------------------------------


class _FakeEvent:
    def __init__(self, spy):
        self.spy = spy

    def synchronize(self):
        self.spy["event_syncs"] += 1


class _FakeStream:
    def __init__(self, spy):
        self.spy = spy

    def wait_stream(self, other):
        pass

    def record_event(self):
        self.spy["record_events"] += 1
        return _FakeEvent(self.spy)


class _FakeStreamCtx:
    def __init__(self, stream):
        self.stream = stream

    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


class _FakeGroup:
    def __init__(self, size, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _install_cuda_fakes(spy):
    """Patch torch.cuda.{Stream,current_stream,stream} for a CPU-only box."""
    main_stream = _FakeStream(spy)
    torch.cuda.Stream = lambda: _FakeStream(spy)
    torch.cuda.current_stream = lambda: main_stream
    torch.cuda.stream = lambda s: _FakeStreamCtx(s)


def _install_rng_stubs():
    """CPU-only RNG state for the real CheckpointFunction (CUDA RNG absent)."""
    tp_random._get_all_rng_states = lambda: (torch.get_rng_state(), None, None)

    def _set_all(cpu_state, cuda_state, tracker_state):
        torch.set_rng_state(cpu_state)

    tp_random._set_all_rng_states = _set_all


def _install_collective_fakes(spy):
    """Single-process stand-ins for the tp_ep all-gather and the EP all-to-all.

    The all-gather fake mimics a world of EP_SIZE ranks whose local token
    counts are identical (sufficient: the split math under test is a pure
    function of the gathered values). The all-to-all fake is identity but
    records the splits it was handed so the test can compare fwd vs replay.
    Also bypass the distributed-initialized guard in utils.get_pg_size/rank so
    the fake groups' sizes are honored without a real process group.
    """

    def fake_gather(t, group=None, **kwargs):
        spy["gathers"] += 1
        return torch.cat([t, t], dim=0)

    def _snapshot(splits):
        # On GPU the splits are numpy (as_numpy=True in maybe_move_tensor_to_cpu);
        # on CPU they stay torch tensors (the move is a no-op). Snapshot either.
        if splits is None:
            return None
        return splits.clone() if torch.is_tensor(splits) else splits.copy()

    def fake_all_to_all(group, tokens, output_splits, input_splits, use_nccl_stream=False):
        spy["a2a"].append((_snapshot(output_splits), _snapshot(input_splits)))
        # Shape-consistent stand-in: the output row count is sum(output_splits)
        # (what the real a2a delivers); rows are a deterministic tiling of the
        # input (identical across runs, differentiable).
        out_rows = int(sum(int(s) for s in output_splits))
        n = tokens.size(0)
        if n == out_rows:
            return tokens
        reps = (out_rows + n - 1) // n
        return tokens.repeat((reps,) + (1,) * (tokens.dim() - 1))[:out_rows]

    td.gather_from_sequence_parallel_region = fake_gather
    td.all_to_all = fake_all_to_all
    td.utils.get_pg_size = lambda group=None: group.size() if group is not None else 1
    td.utils.get_pg_rank = lambda group=None: group.rank() if group is not None else 0


# ---------------------------------------------------------------------------
# Fixture: a real MoEAlltoAllTokenDispatcher on CPU
# ---------------------------------------------------------------------------

NUM_TOKENS = 16
HIDDEN = 8
NUM_EXPERTS = 4
TOPK = 2
EP_SIZE = 2
NUM_LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE


def _make_config():
    return types.SimpleNamespace(
        num_moe_experts=NUM_EXPERTS,
        moe_permute_fusion=False,  # unfused path: CPU-safe, exercises the host-move branch
        moe_pad_expert_input_to_capacity=False,
        moe_expert_capacity_factor=None,
        moe_router_padding_for_quantization=False,
        moe_router_topk=TOPK,
        cuda_graph_impl="none",
        cuda_graph_modules=[],
    )


def _make_pg_collection():
    return types.SimpleNamespace(
        ep=_FakeGroup(EP_SIZE),
        expt_tp=_FakeGroup(1),
        tp_ep=_FakeGroup(EP_SIZE),
    )


def _make_dispatcher(spy):
    td.MoEAlltoAllTokenDispatcher.cuda_dtoh_stream = None  # force re-init with fake stream
    return td.MoEAlltoAllTokenDispatcher(
        num_local_experts=NUM_LOCAL_EXPERTS,
        local_expert_indices=list(range(NUM_LOCAL_EXPERTS)),
        config=_make_config(),
        pg_collection=_make_pg_collection(),
    )


def _routing_from_hidden(hidden):
    """Deterministic routing from hidden values: a NEW tensor each call with
    the same values for the same hidden — exactly what the recompute replay
    produces."""
    scores = hidden[:, :NUM_EXPERTS].abs()
    topk_idx = scores.argsort(dim=1, descending=True)[:, :TOPK]
    routing_map = torch.zeros(hidden.size(0), NUM_EXPERTS, dtype=torch.bool)
    routing_map.scatter_(1, topk_idx, True)
    probs = hidden[:, :NUM_EXPERTS].sigmoid()
    return routing_map, probs


def _make_chunk(dispatcher):
    """A fake one-MoE-layer chunk: the full dispatch/combine flow."""

    def chunk(hidden_states):
        routing_map, probs = _routing_from_hidden(hidden_states)
        hs, ps = dispatcher.dispatch_preprocess(hidden_states, routing_map, probs)
        gs, gp = dispatcher.token_dispatch(hs, ps)
        gi, tpe, gp2 = dispatcher.dispatch_postprocess(gs, gp)
        out = dispatcher.combine_preprocess(gi)  # fake experts: identity
        out = dispatcher.token_combine(out)
        out = dispatcher.combine_postprocess(out)
        return out

    return chunk


def _run_checkpointed(chunk, psp, hidden):
    """The production wiring: marker-wrap the chunk, then checkpoint it."""
    wrapped = recompute._wrap_checkpoint_chunk_pass(chunk, psp)
    out = tensor_parallel.checkpoint(wrapped, False, hidden)
    loss = out.square().sum()
    loss.backward()
    return out


def _reset_fixc_state():
    """Reset the gate logs / stats between cases (test-only)."""
    td._REPLAY_CACHE_GATE_LOGGED[0] = False
    td._REPLAY_VERIFY_GATE_LOGGED[0] = False
    td._REPLAY_ARMED_LOGGED[0] = False
    td._REPLAY_HIT_LOGGED[0] = False
    for k in td._REPLAY_STATS:
        td._REPLAY_STATS[k] = 0
    td._REPLAY_WINDOW[0] = 0
    td._REPLAY_WINDOW[1] = 0


def _fresh_spy():
    return {"gathers": 0, "event_syncs": 0, "record_events": 0, "a2a": []}


def _make_hidden(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(NUM_TOKENS, HIDDEN, generator=g, requires_grad=True)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def case_gate_off_inert():
    print("\n== case 1: gate OFF — code inert ==")
    os.environ.pop("BT_MOE_DISPATCH_REPLAY_CACHE", None)
    _reset_fixc_state()
    spy = _fresh_spy()
    _install_cuda_fakes(spy)
    _install_collective_fakes(spy)
    dispatcher = _make_dispatcher(spy)
    chunk = _make_chunk(dispatcher)
    psp = PackedSeqParams()

    wrapped = recompute._wrap_checkpoint_chunk_pass(chunk, psp)
    check("gate off: _wrap_checkpoint_chunk_pass returns the function unchanged", wrapped is chunk)

    hidden = _make_hidden(0)
    out = _run_checkpointed(chunk, psp, hidden)
    check("gate off: all-gather ran in both passes", spy["gathers"] == 2)
    check("gate off: event record in both passes", spy["record_events"] == 2)
    check("gate off: event sync in both passes", spy["event_syncs"] == 2)
    check("gate off: no stores/hits/misses", all(v == 0 for v in td._REPLAY_STATS.values()))
    check("gate off: no cache attribute on the carrier", not hasattr(psp, td._REPLAY_CACHE_ATTR))
    return out.detach(), spy


def case_gate_on_hit(reference_out, reference_spy):
    print("\n== case 2: gate ON — replay hits, replay syncs skipped, bitwise parity ==")
    os.environ["BT_MOE_DISPATCH_REPLAY_CACHE"] = "1"
    _reset_fixc_state()
    spy = _fresh_spy()
    _install_cuda_fakes(spy)
    _install_collective_fakes(spy)
    dispatcher = _make_dispatcher(spy)
    chunk = _make_chunk(dispatcher)
    psp = PackedSeqParams()

    wrapped = recompute._wrap_checkpoint_chunk_pass(chunk, psp)
    check("gate on: chunk is wrapped by the pass marker", wrapped is not chunk)
    check(
        "gate on: marker is a plain callable, not an autograd.Function",
        not issubclass(type(wrapped), torch.autograd.Function),
    )

    hidden = _make_hidden(0)
    out = _run_checkpointed(wrapped, psp, hidden)

    check("gate on: all-gather ran once (first pass only)", spy["gathers"] == 1)
    check("gate on: event record once (first pass only)", spy["record_events"] == 1)
    check("gate on: event sync once (first pass only)", spy["event_syncs"] == 1)
    check(
        "gate on: one store, one hit, no misses",
        td._REPLAY_STATS["stores"] == 1
        and td._REPLAY_STATS["hits"] == 1
        and td._REPLAY_STATS["misses"] == 0
        and td._REPLAY_STATS["shape_mismatches"] == 0,
    )
    check("gate on: output bitwise-equal to gate-off", torch.equal(out, reference_out))
    check(
        "gate on: a2a call count matches gate-off (3 fwd + 3 replay)",
        len(spy["a2a"]) == len(reference_spy["a2a"]) == 6,
    )
    same_splits = all(
        np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        for a, b in zip(spy["a2a"], reference_spy["a2a"])
    )
    check("gate on: splits handed to all_to_all bitwise-equal to gate-off", same_splits)
    replay_uses_fwd = np.array_equal(spy["a2a"][0][0], spy["a2a"][3][0]) and np.array_equal(
        spy["a2a"][0][1], spy["a2a"][3][1]
    )
    check("gate on: replay a2a splits bitwise-equal to its own first pass", replay_uses_fwd)


def case_four_datums_two_layers():
    print("\n== case 3: 4 microbatches in flight x 2 layers — carrier keying ==")
    os.environ["BT_MOE_DISPATCH_REPLAY_CACHE"] = "1"
    _reset_fixc_state()

    def run_schedule():
        spy = _fresh_spy()
        _install_cuda_fakes(spy)
        _install_collective_fakes(spy)
        dispatchers = [_make_dispatcher(spy), _make_dispatcher(spy)]
        psps = [PackedSeqParams() for _ in range(4)]
        hiddens = [_make_hidden(100 + i) for i in range(4)]
        outs = []
        losses = []
        # fwd all microbatches (both layers), then bwd all microbatches
        for i in range(4):
            h = hiddens[i]
            for dispatcher in dispatchers:
                wrapped = recompute._wrap_checkpoint_chunk_pass(_make_chunk(dispatcher), psps[i])
                h = tensor_parallel.checkpoint(wrapped, False, h)
            outs.append(h)
            losses.append(h.square().sum())
        for loss in losses:
            loss.backward()
        return [o.detach() for o in outs], spy

    os.environ.pop("BT_MOE_DISPATCH_REPLAY_CACHE", None)
    _reset_fixc_state()
    ref_outs, ref_spy = run_schedule()
    check("4mb gate off: gathers = 2 passes x 4 mb x 2 layers", ref_spy["gathers"] == 16)
    check("4mb gate off: event syncs = 16", ref_spy["event_syncs"] == 16)

    os.environ["BT_MOE_DISPATCH_REPLAY_CACHE"] = "1"
    _reset_fixc_state()
    outs, spy = run_schedule()
    check("4mb gate on: gathers = 8 (first passes only)", spy["gathers"] == 8)
    check("4mb gate on: event syncs = 8 (first passes only)", spy["event_syncs"] == 8)
    check(
        "4mb gate on: 8 stores, 8 hits, no misses",
        td._REPLAY_STATS["stores"] == 8
        and td._REPLAY_STATS["hits"] == 8
        and td._REPLAY_STATS["misses"] == 0
        and td._REPLAY_STATS["shape_mismatches"] == 0,
    )
    check(
        "4mb gate on: outputs bitwise-equal to gate-off",
        all(torch.equal(a, b) for a, b in zip(outs, ref_outs)),
    )
    same_log = len(spy["a2a"]) == len(ref_spy["a2a"]) and all(
        np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        for a, b in zip(spy["a2a"], ref_spy["a2a"])
    )
    check("4mb gate on: full a2a splits log bitwise-equal to gate-off", same_log)


def case_eval_between_fwd_and_bwd():
    print("\n== case 4: eval forward between fwd and bwd does not pollute ==")
    os.environ["BT_MOE_DISPATCH_REPLAY_CACHE"] = "1"
    _reset_fixc_state()
    spy = _fresh_spy()
    _install_cuda_fakes(spy)
    _install_collective_fakes(spy)
    dispatcher = _make_dispatcher(spy)
    chunk = _make_chunk(dispatcher)
    psp = PackedSeqParams()

    hidden = _make_hidden(7)
    wrapped = recompute._wrap_checkpoint_chunk_pass(chunk, psp)
    out = tensor_parallel.checkpoint(wrapped, False, hidden)
    stores_after_fwd = td._REPLAY_STATS["stores"]

    # Eval-style forward: no checkpoint, no marker, grad disabled, different data.
    with torch.no_grad():
        eval_hidden = _make_hidden(999)
        chunk(eval_hidden)
    check("eval: ran the status-quo path (all-gather fired)", spy["gathers"] == 2)
    check("eval: no store without a checkpoint frame", td._REPLAY_STATS["stores"] == stores_after_fwd)

    loss = out.square().sum()
    loss.backward()
    check("eval: replay still hits its own entry", td._REPLAY_STATS["hits"] == 1)
    check("eval: no misses", td._REPLAY_STATS["misses"] == 0)


def case_forced_fallback_miss():
    print("\n== case 5: forced fallback — replay with no stored entry ==")
    os.environ["BT_MOE_DISPATCH_REPLAY_CACHE"] = "1"
    _reset_fixc_state()
    spy = _fresh_spy()
    _install_cuda_fakes(spy)
    _install_collective_fakes(spy)
    dispatcher = _make_dispatcher(spy)
    chunk = _make_chunk(dispatcher)
    psp = PackedSeqParams()

    # Simulate a replay whose first pass never stored: call the marker-wrapped
    # chunk directly under enable_grad (the marker classifies it as a replay).
    wrapped = recompute._wrap_checkpoint_chunk_pass(chunk, psp)
    hidden = _make_hidden(11)
    with torch.enable_grad():
        wrapped(hidden)
    check("miss: counter incremented", td._REPLAY_STATS["misses"] == 1)
    check("miss: no hit", td._REPLAY_STATS["hits"] == 0)
    check("miss: fallback ran the full path (all-gather + event sync)",
          spy["gathers"] == 1 and spy["event_syncs"] == 1)


def case_shape_mismatch():
    print("\n== case 6: forced fallback — routing shape changed ==")
    os.environ["BT_MOE_DISPATCH_REPLAY_CACHE"] = "1"
    _reset_fixc_state()
    spy = _fresh_spy()
    _install_cuda_fakes(spy)
    _install_collective_fakes(spy)
    dispatcher = _make_dispatcher(spy)
    psp = PackedSeqParams()

    # Real first pass stores an entry for this carrier.
    hidden = _make_hidden(13)
    wrapped = recompute._wrap_checkpoint_chunk_pass(_make_chunk(dispatcher), psp)
    tensor_parallel.checkpoint(wrapped, False, hidden)
    check("shape: entry stored", td._REPLAY_STATS["stores"] == 1)

    # Replay with the same carrier but a different token count: the shape
    # guard must fall back instead of serving the stale entry.
    def bigger_chunk(hidden_states):
        routing_map, probs = _routing_from_hidden(hidden_states)
        hs, ps = dispatcher.dispatch_preprocess(hidden_states, routing_map, probs)
        gs, gp = dispatcher.token_dispatch(hs, ps)
        gi, tpe, gp2 = dispatcher.dispatch_postprocess(gs, gp)
        out = dispatcher.combine_preprocess(gi)
        out = dispatcher.token_combine(out)
        return dispatcher.combine_postprocess(out)

    wrapped_big = recompute._wrap_checkpoint_chunk_pass(bigger_chunk, psp)
    big_hidden = torch.randn(2 * NUM_TOKENS, HIDDEN, requires_grad=True)
    with torch.enable_grad():
        wrapped_big(big_hidden)
    check("shape: mismatch counted", td._REPLAY_STATS["shape_mismatches"] == 1)
    check("shape: no hit", td._REPLAY_STATS["hits"] == 0)
    check("shape: fallback ran the full path", spy["gathers"] == 2 and spy["event_syncs"] == 2)


def case_verify_mode():
    print("\n== case 7: verify mode — recompute + bitwise assert; corruption raises ==")
    os.environ["BT_MOE_DISPATCH_REPLAY_CACHE"] = "1"
    os.environ["BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY"] = "1"
    _reset_fixc_state()
    spy = _fresh_spy()
    _install_cuda_fakes(spy)
    _install_collective_fakes(spy)
    dispatcher = _make_dispatcher(spy)
    psp = PackedSeqParams()

    hidden = _make_hidden(17)
    wrapped = recompute._wrap_checkpoint_chunk_pass(_make_chunk(dispatcher), psp)
    out = _run_checkpointed(wrapped, psp, hidden)
    check("verify: replay recomputed (all-gather ran twice)", spy["gathers"] == 2)
    check("verify: replay event sync ran (full cost paid)", spy["event_syncs"] == 2)
    check("verify: one verify, one hit, no misses",
          td._REPLAY_STATS["verifies"] == 1 and td._REPLAY_STATS["hits"] == 1
          and td._REPLAY_STATS["misses"] == 0)

    # Corrupt the cached entry before the replay: verify must raise.
    _reset_fixc_state()
    spy2 = _fresh_spy()
    _install_cuda_fakes(spy2)
    _install_collective_fakes(spy2)
    dispatcher2 = _make_dispatcher(spy2)
    psp2 = PackedSeqParams()
    hidden2 = _make_hidden(19)
    wrapped2 = recompute._wrap_checkpoint_chunk_pass(_make_chunk(dispatcher2), psp2)
    out2 = tensor_parallel.checkpoint(wrapped2, False, hidden2)
    store = getattr(psp2, td._REPLAY_CACHE_ATTR)
    entry = store[id(dispatcher2)]
    entry.input_splits_host = (
        entry.input_splits_host.clone()
        if torch.is_tensor(entry.input_splits_host)
        else entry.input_splits_host.copy()
    )
    entry.input_splits_host[0] += 1  # corrupt
    raised = False
    try:
        out2.square().sum().backward()
    except RuntimeError as e:
        raised = "BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY" in str(e)
    check("verify: corrupted entry raises RuntimeError in the replay", raised)
    os.environ.pop("BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY", None)


def case_no_frame_plain_training():
    print("\n== case 8: gate on, no checkpoint (plain training) — status quo ==")
    os.environ["BT_MOE_DISPATCH_REPLAY_CACHE"] = "1"
    _reset_fixc_state()
    spy = _fresh_spy()
    _install_cuda_fakes(spy)
    _install_collective_fakes(spy)
    dispatcher = _make_dispatcher(spy)
    chunk = _make_chunk(dispatcher)

    hidden = _make_hidden(23)
    out = chunk(hidden)  # grad enabled, no marker, no checkpoint
    out.square().sum().backward()
    check("no frame: all-gather + event sync ran", spy["gathers"] == 1 and spy["event_syncs"] == 1)
    check("no frame: no stores/hits/misses", all(v == 0 for v in td._REPLAY_STATS.values()))


def case_carrier_none_fallback():
    print("\n== case 9: packed_seq_params=None — marker instance is the key ==")
    os.environ["BT_MOE_DISPATCH_REPLAY_CACHE"] = "1"
    _reset_fixc_state()
    spy = _fresh_spy()
    _install_cuda_fakes(spy)
    _install_collective_fakes(spy)
    dispatcher = _make_dispatcher(spy)
    chunk = _make_chunk(dispatcher)

    hidden = _make_hidden(29)
    wrapped = recompute._wrap_checkpoint_chunk_pass(chunk, None)
    out = _run_checkpointed(wrapped, None, hidden)
    check("None carrier: store + hit", td._REPLAY_STATS["stores"] == 1 and td._REPLAY_STATS["hits"] == 1)
    check("None carrier: replay all-gather skipped", spy["gathers"] == 1)


def case_source_guards():
    print("\n== case 10: v1-lesson source guards ==")
    src = open(td.__file__).read()
    check(
        "dispatcher never reads torch.is_grad_enabled (grad-mode read lives in the "
        "recompute marker, which the checkpoint executes as a plain callable)",
        "is_grad_enabled" not in src,
    )
    rsrc = open(recompute.__file__).read()
    check(
        "the marker's grad-mode read is in recompute.py (plain functor, not "
        "an autograd.Function.forward)",
        "is_grad_enabled" in rsrc
        and not issubclass(recompute._CheckpointChunkPassMarker, torch.autograd.Function),
    )


def case_fused_mode():
    print("\n== case 11: moe_permute_fusion=True (ship config) — device-form num_global ==")
    os.environ["BT_MOE_DISPATCH_REPLAY_CACHE"] = "1"
    _reset_fixc_state()

    # Fused permute/unpermute/sort_chunks need TE kernels; stub them with
    # deterministic CPU fakes. The cache-relevant branch under test is the
    # moe_permute_fusion=True guard: num_global_tokens_per_local_expert is NOT
    # host-moved, so the entry carries only its device form and the replay
    # must reinstall that exact tensor.
    sort_spy = []

    def fake_permute(tokens, routing_map, probs=None, num_out_tokens=None, fused=False,
                     drop_and_pad=False):
        flat_idx = routing_map.view(-1).nonzero().view(-1)
        token_idx = flat_idx // routing_map.size(1)
        permuted = tokens[token_idx]
        p = probs.view(-1)[flat_idx].unsqueeze(-1) if probs is not None else None
        return permuted, p, token_idx, None, None

    def fake_unpermute(permuted, mapping, restore_shape, routing_map, fused=False,
                       drop_and_pad=False):
        out = torch.zeros(restore_shape, dtype=permuted.dtype, device=permuted.device)
        out.index_add_(0, mapping, permuted)
        return out

    def fake_sort_chunks(input, split_sizes, sorted_idxs, probs=None, fused=False):
        sort_spy.append(split_sizes)
        return input, probs

    real_permute, real_unpermute, real_sort = td.permute, td.unpermute, td.sort_chunks_by_idxs
    td.permute, td.unpermute, td.sort_chunks_by_idxs = (
        fake_permute,
        fake_unpermute,
        fake_sort_chunks,
    )
    try:
        spy = _fresh_spy()
        _install_cuda_fakes(spy)
        _install_collective_fakes(spy)
        dispatcher = _make_dispatcher(spy)
        dispatcher.config.moe_permute_fusion = True  # flip post-construction (CPU has no cuda arange)
        psp = PackedSeqParams()
        hidden = _make_hidden(31)
        wrapped = recompute._wrap_checkpoint_chunk_pass(_make_chunk(dispatcher), psp)
        _run_checkpointed(wrapped, psp, hidden)

        check("fused: store + hit, no misses",
              td._REPLAY_STATS["stores"] == 1 and td._REPLAY_STATS["hits"] == 1
              and td._REPLAY_STATS["misses"] == 0)
        check("fused: replay all-gather + event sync skipped",
              spy["gathers"] == 1 and spy["event_syncs"] == 1)
        # The replay's sort_chunks must have received the CACHED device
        # num_global tensor: dispatch_postprocess ravels it (a view — same
        # storage as the first pass's), combine_preprocess's .T.ravel() copies
        # (values only).
        check("fused: sort_chunks saw 2 passes x 2 calls", len(sort_spy) == 4)
        check(
            "fused: replay sort_chunks split_sizes share storage with the first pass's cached tensor",
            sort_spy[0].data_ptr() == sort_spy[2].data_ptr()
            and torch.equal(sort_spy[1], sort_spy[3]),
        )
    finally:
        td.permute, td.unpermute, td.sort_chunks_by_idxs = real_permute, real_unpermute, real_sort

    # Inspect the entry of a fresh first pass: no host form recorded in fused mode.
    _reset_fixc_state()
    sort_spy.clear()
    td.permute, td.unpermute, td.sort_chunks_by_idxs = fake_permute, fake_unpermute, fake_sort_chunks
    try:
        spy2 = _fresh_spy()
        _install_cuda_fakes(spy2)
        _install_collective_fakes(spy2)
        dispatcher2 = _make_dispatcher(spy2)
        dispatcher2.config.moe_permute_fusion = True
        psp2 = PackedSeqParams()
        hidden2 = _make_hidden(37)
        wrapped2 = recompute._wrap_checkpoint_chunk_pass(_make_chunk(dispatcher2), psp2)
        tensor_parallel.checkpoint(wrapped2, False, hidden2)  # first pass only
        entry = getattr(psp2, td._REPLAY_CACHE_ATTR)[id(dispatcher2)]
        check(
            "fused: entry carries no host form of num_global_tokens_per_local_expert",
            entry.num_global_tokens_per_local_expert_host is None,
        )
        check(
            "fused: entry device num_global is the tensor the first pass used",
            entry.num_global_tokens_per_local_expert_dev is not None
            and entry.num_global_tokens_per_local_expert_dev.data_ptr()
            == sort_spy[0].data_ptr(),
        )
    finally:
        td.permute, td.unpermute, td.sort_chunks_by_idxs = real_permute, real_unpermute, real_sort


def main():
    _install_rng_stubs()
    reference_out, reference_spy = case_gate_off_inert()
    case_gate_on_hit(reference_out, reference_spy)
    case_four_datums_two_layers()
    case_eval_between_fwd_and_bwd()
    case_forced_fallback_miss()
    case_shape_mismatch()
    case_verify_mode()
    case_no_frame_plain_training()
    case_carrier_none_fallback()
    case_source_guards()
    case_fused_mode()

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
