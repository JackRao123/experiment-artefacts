#!/usr/bin/env python3
"""T1 CPU test for W1 — BT_MOE_PROBS_A2A_COMM (deferred-wait A2A + probs comm).

Verifies the issue/wait split primitive (`_AllToAllDeferredWait` /
`all_to_all_deferred` / `wait_deferred_a2a` in
megatron/core/tensor_parallel/mappings.py) and the dispatcher-side gate
helpers, using the gloo backend on CPU (2 procs):

  1. deferred issue + wait is value-identical to the reference
     `all_to_all_single` (unequal splits; the dispatcher's dropless case);
  2. the W1 pattern — tokens A2A on the EP group + probs A2A on a SECOND
     communicator over the same ranks, both issued before either wait, with a
     stand-in "shared fc1" op launched in between — produces the reference
     results on both tensors (the second communicator carries the identical
     messages the EP communicator would);
  3. work-handle lifecycle: `_deferred_a2a_work` attached after issue, cleared
     by wait, second wait is a no-op; world_size==1 bypass returns the input
     unchanged and wait_deferred_a2a no-ops on it;
  4. backward parity: input grads through the deferred path are bitwise
     identical to the status-quo `_AllToAll` path (reverse A2A with swapped
     splits), for both the tokens and the probs tensors;
  5. gate + telemetry: `_probs_a2a_comm_enabled()` tracks the env var, and
     `_probs_a2a_comm_note_dispatch()` accumulates issue/wait counters (the
     armed/hit telemetry that proves the gate can fire — the v1 lesson).

On-box the tokens payload is bf16 [rows, 6144] and probs f32 [rows]; gloo's
bf16 collectives support is uneven, so this test uses f32 throughout — the
issue/wait mechanics are dtype-agnostic and T3 (on-box canary) covers the
real dtypes end-to-end.

Runs on Mac CPU. Target the vendored mcore with BT_TEST_MCORE_PATH if the
default walk-up does not find it.

Usage: python test_w1_probs_a2a.py
"""

import os
import sys
import tempfile
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
    # overlap_design/tests/ lives outside the repo; fall back to the known checkout
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
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

import megatron.core.tensor_parallel.mappings as mappings  # noqa: E402
import megatron.core.transformer.moe.token_dispatcher as token_dispatcher  # noqa: E402

FAILURES = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name, flush=True)
    if not cond:
        FAILURES.append(name)


# Unequal-split layout for 2 ranks (rows per peer). Rank 0 sends [3, 5],
# rank 1 sends [2, 4]; each rank's output splits are the column of the other's.
IN_SPLITS = {0: [3, 5], 1: [2, 4]}
OUT_SPLITS = {0: [3, 2], 1: [5, 4]}
H = 4  # hidden width for the tokens stand-in


def _make(rank, seed):
    g = torch.Generator().manual_seed(seed + rank)
    n = sum(IN_SPLITS[rank])
    tokens = torch.randn(n, H, generator=g)
    probs = torch.randn(n, generator=g)
    return tokens, probs


def _reference_a2a(group, input_, out_splits, in_splits):
    out = input_.new_empty([sum(out_splits)] + list(input_.size()[1:]))
    dist.all_to_all_single(
        out, input_.contiguous(), output_split_sizes=out_splits,
        input_split_sizes=in_splits, group=group,
    )
    return out


def _sec1_deferred_matches_reference(rank):
    group = dist.group.WORLD
    tokens, probs = _make(rank, 100)
    out = mappings.all_to_all_deferred(
        group, tokens, OUT_SPLITS[rank], IN_SPLITS[rank]
    )
    mappings.wait_deferred_a2a(out)
    ref = _reference_a2a(group, tokens, OUT_SPLITS[rank], IN_SPLITS[rank])
    check("sec1 deferred tokens == reference all_to_all_single", torch.equal(out, ref))

    out_p = mappings.all_to_all_deferred(
        group, probs, OUT_SPLITS[rank], IN_SPLITS[rank]
    )
    mappings.wait_deferred_a2a(out_p)
    ref_p = _reference_a2a(group, probs, OUT_SPLITS[rank], IN_SPLITS[rank])
    check("sec1 deferred probs == reference all_to_all_single", torch.equal(out_p, ref_p))

    # equal-split path (output_split_sizes=None) — requires uniform row counts
    # across ranks, so use a fresh same-size tensor (not the unequal one)
    g = torch.Generator().manual_seed(1000 + rank)
    uniform = torch.randn(4, H, generator=g)
    out_eq = mappings.all_to_all_deferred(group, uniform, None, None)
    mappings.wait_deferred_a2a(out_eq)
    ref_eq = _reference_a2a(group, uniform, [2, 2], [2, 2])
    check("sec1 deferred equal-split == reference", torch.equal(out_eq, ref_eq))


def _sec2_two_comm_pattern(rank):
    """The W1 dispatch order: issue tokens, issue probs, stand-in shared fc1,
    wait tokens, wait probs.

    gloo shares one transport across all groups and mismatches concurrent
    collectives on DIFFERENT communicators (NCCL — the production target —
    has per-communicator streams/channels and is unaffected; the two-comm
    concurrent pattern is covered on-box by T3). So: (a) exercises the exact
    issue/issue/fc1/wait/wait mechanics on one group; (b) exercises the second
    communicator (new_group over the same ranks) for value equivalence with
    the probs A2A it will carry.
    """
    ep_group = dist.group.WORLD
    probs_group = dist.new_group(dist.get_process_group_ranks(ep_group))
    tokens, probs = _make(rank, 200)

    # (a) exact W1 ordering, one group (gloo-safe)
    out_t = mappings.all_to_all_deferred(
        ep_group, tokens, OUT_SPLITS[rank], IN_SPLITS[rank]
    )
    out_p = mappings.all_to_all_deferred(
        ep_group, probs, OUT_SPLITS[rank], IN_SPLITS[rank]
    )
    # stand-in for shared_experts.linear_fc1_forward_and_act launched between
    # the issues and the waits (must not disturb the in-flight collectives)
    _fc1_stand_in = tokens.sum() * 2.0
    mappings.wait_deferred_a2a(out_t)
    mappings.wait_deferred_a2a(out_p)

    ref_t = _reference_a2a(ep_group, tokens, OUT_SPLITS[rank], IN_SPLITS[rank])
    ref_p = _reference_a2a(ep_group, probs, OUT_SPLITS[rank], IN_SPLITS[rank])
    check("sec2a issue/issue/fc1/wait/wait: tokens == reference", torch.equal(out_t, ref_t))
    check("sec2a issue/issue/fc1/wait/wait: probs == reference", torch.equal(out_p, ref_p))
    check("sec2a fc1 stand-in computed", torch.isfinite(_fc1_stand_in))

    # (b) second communicator over the same ranks carries identical messages
    out_p2 = mappings.all_to_all_deferred(
        probs_group, probs, OUT_SPLITS[rank], IN_SPLITS[rank]
    )
    mappings.wait_deferred_a2a(out_p2)
    check("sec2b probs on second comm == reference on EP comm", torch.equal(out_p2, ref_p))


def _sec3_handle_lifecycle(rank):
    group = dist.group.WORLD
    tokens, _ = _make(rank, 300)
    out = mappings.all_to_all_deferred(group, tokens, OUT_SPLITS[rank], IN_SPLITS[rank])
    check(
        "sec3 work handle attached after issue",
        getattr(out, "_deferred_a2a_work", None) is not None,
    )
    mappings.wait_deferred_a2a(out)
    check(
        "sec3 work handle cleared after wait",
        getattr(out, "_deferred_a2a_work", None) is None,
    )
    # second wait is a no-op (no handle)
    mappings.wait_deferred_a2a(out)
    check("sec3 second wait no-op", True)

    # world_size == 1 bypass: single-rank subgroup returns the input contents
    # unchanged and attaches no work handle (autograd may wrap the tensor, so
    # compare values, not identity)
    single = dist.new_group([rank])
    out1 = mappings.all_to_all_deferred(single, tokens, None, None)
    check("sec3 world_size==1 bypass returns input values", torch.equal(out1, tokens))
    check(
        "sec3 world_size==1 bypass attaches no handle",
        getattr(out1, "_deferred_a2a_work", None) is None,
    )
    mappings.wait_deferred_a2a(out1)  # must not raise despite no handle
    check("sec3 world_size==1 wait no-op", True)


def _sec4_backward_parity(rank):
    group = dist.group.WORLD
    g = torch.Generator().manual_seed(400 + rank)

    for name in ("tokens", "probs"):
        tokens, probs = _make(rank, 500)
        x = tokens if name == "tokens" else probs
        grad_out = torch.randn(
            [sum(OUT_SPLITS[rank])] + list(x.size()[1:]), generator=g
        )

        # deferred path
        x_def = x.clone().requires_grad_(True)
        out_def = mappings.all_to_all_deferred(
            group, x_def, OUT_SPLITS[rank], IN_SPLITS[rank]
        )
        mappings.wait_deferred_a2a(out_def)
        out_def.backward(grad_out)

        # Reference: the backward of an all-to-all is the all-to-all with
        # swapped splits — exactly what the upstream `_AllToAll.backward`
        # computes (it re-applies `_AllToAll` with input/output splits
        # swapped). The upstream Function itself cannot run on CPU
        # (torch.cuda.current_device() in its output allocation), so the
        # reference is the equivalent plain collective here; T3 covers the
        # upstream path end-to-end on-box.
        ref_grad = _reference_a2a(group, grad_out, IN_SPLITS[rank], OUT_SPLITS[rank])

        check(
            f"sec4 backward parity ({name}): deferred grad == reverse-A2A reference",
            torch.equal(x_def.grad, ref_grad),
        )


def _sec5_gate_and_telemetry(rank):
    td = token_dispatcher
    old = os.environ.pop("BT_MOE_PROBS_A2A_COMM", None)
    try:
        check("sec5 gate off by default", td._probs_a2a_comm_enabled() is False)
        os.environ["BT_MOE_PROBS_A2A_COMM"] = "1"
        check("sec5 gate on with env=1", td._probs_a2a_comm_enabled() is True)
        before = dict(td._PROBS_A2A_COMM_STATS)
        td._probs_a2a_comm_note_dispatch()
        after = td._PROBS_A2A_COMM_STATS
        check(
            "sec5 telemetry counters increment (2 issues, 2 waits per dispatch)",
            after["token_issues"] == before["token_issues"] + 1
            and after["probs_issues"] == before["probs_issues"] + 1
            and after["waits"] == before["waits"] + 2,
        )
    finally:
        if old is None:
            os.environ.pop("BT_MOE_PROBS_A2A_COMM", None)
        else:
            os.environ["BT_MOE_PROBS_A2A_COMM"] = old


def _sec6_ep_span_guard(rank):
    """The new_group rendezvous guard (helmholtz's PR-review finding): with
    >1 EP group (e.g. world-32 EP16) the default new_group contract is unsafe;
    the gate must refuse to arm unless the EP group spans the world."""
    td = token_dispatcher
    check("sec6 guard: full-world EP group spans",
          td._probs_a2a_ep_spans_world([0, 1], 2) is True)
    check("sec6 guard: unsorted full-world ranks still span",
          td._probs_a2a_ep_spans_world([1, 0], 2) is True)
    check("sec6 guard: partial EP group does not span (world-32 EP16 class)",
          td._probs_a2a_ep_spans_world(list(range(16)), 32) is False)
    check("sec6 guard: the second EP group of world-32 does not span either",
          td._probs_a2a_ep_spans_world(list(range(16, 32)), 32) is False)


def _worker(rank, world_size, store_path):
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
    try:
        _sec1_deferred_matches_reference(rank)
        _sec2_two_comm_pattern(rank)
        _sec3_handle_lifecycle(rank)
        _sec4_backward_parity(rank)
        if rank == 0:
            _sec5_gate_and_telemetry(rank)
            _sec6_ep_span_guard(rank)
        dist.barrier()
    finally:
        dist.destroy_process_group()
    if FAILURES:
        raise SystemExit(f"rank {rank}: {len(FAILURES)} FAILURES: {FAILURES}")
    print(f"rank {rank}: all checks passed", flush=True)


def main():
    world_size = 2
    with tempfile.TemporaryDirectory() as tmp:
        store_path = os.path.join(tmp, "gloo_store")
        mp.spawn(_worker, args=(world_size, store_path), nprocs=world_size, join=True)
    print("OK — all W1 T1 checks passed on", world_size, "gloo procs")


if __name__ == "__main__":
    main()
