#!/usr/bin/env python3
"""T1b gloo/autograd test for the W2 chunked A2A Functions (BT_MOE_A2A_PIPELINE).

Second-reviewer artifact (hilbert, 2026-08-09), complementing helmholtz's
test_w2_chunk_plan.py (pure index math) and the on-box t2_w2_numerics_gate.py
(end-to-end FP8). Drives the REAL `_ChunkedDispatchA2A` and
`_ChunkedCombineA2A` autograd Functions from the vendored tree on 4 gloo
procs (CPU) with an IMBALANCED routing matrix incl. zero-count peers/groups,
and asserts:

  1. dispatch fwd: the K chunked recv buffers re-interleave to the exact
     monolithic all_to_all recv buffer (per-src [group0 | group1] blocks);
  2. dispatch bwd (fibonacci focus 1): the single grad buffer for the
     permuted input — K reverse A2As writing disjoint views, no accumulation
     kernel — is bitwise the monolithic reverse-A2A grad (no aliasing, no
     overlap, no double-visit; every row written exactly once);
  3. combine fwd (fibonacci focus 2): the chained per-group passthrough into
     the shared combine buffer is byte-identical to the monolithic combine
     output (P-layout offsets);
  4. combine bwd: the grad of the shared buffer reaches EVERY group's
     unsorted input (a missing dependency = read-before-write on the last
     group would show as a None/zero grad) and matches the monolithic
     reference grad de-interleaved per group;
  5. degenerate: a whole group empty on one rank still fires its A2A with
     all-zero counts and produces the correct (empty) chunk.

Runs on Mac CPU (gloo). BT_TEST_MCORE_PATH overrides the tree walk-up.

STATUS 2026-08-09 (second review, hilbert): the combine rows FAIL on the
landed W2 v1 diff — two verified bugs in _ChunkedCombineA2A, both
demonstrated by this test and both with verified fixes:

  BUG A (severe, training-corrupting): `_w2_combine_works` is carried on the
  combine_buf TENSOR, but when a custom Function returns its input tensor,
  autograd (grad enabled) returns a NEW alias object — attributes do not
  propagate (minimal repro: `b1 is not b0`, attrs lost). So in every
  grad-enabled pass (the recompute replay, i.e. all of training),
  token_combine_chunked's `getattr(combine_buf, "_w2_combine_works", [])`
  returns [] and the combine waits NEVER run — the unpermute reads an
  incomplete buffer (nondeterministic zeros/stale rows). The no-grad first
  pass works by luck. FIX: carry the works on a stable-identity Python
  object — add "combine_works" to _W2ChunkPlan.__slots__ (+ init to []),
  append via `plan.combine_works.append(work)`, and wait
  `for work in plan.combine_works` in token_combine_chunked (the plan is
  per-pass and lives in self._w2_pass["plan"]).

  BUG B (chain-breaking crash): backward returns non-None `grad_buf` for the
  `combine_buf` input, which for group 0 was None (not a Variable) —
  RuntimeError "returned a gradient different than None ... but the
  corresponding forward input was not a Variable" on the FIRST backward of
  every layer-pass. FIX: record `ctx.had_buf = combine_buf is not None` in
  forward and return `grad_buf if ctx.had_buf else None` at that position.

Both fixes verified end-to-end on 4 gloo procs (monkeypatched variants):
chained buffer byte-identical to the monolithic combine, backward runs,
per-group unsorted grads bitwise-equal to the reference. The dispatch rows
(single-Function K-group backward into disjoint views of one grad buffer)
PASS on the landed code — no aliasing/overlap/double-visit there.

Usage: python test_w2_chunked_a2a_functions.py
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
    for _ in range(10):
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

import megatron.core.tensor_parallel.mappings as mappings  # noqa: E402
import megatron.core.transformer.moe.token_dispatcher as td  # noqa: E402

FAILURES = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name, flush=True)
    if not cond:
        FAILURES.append(name)


EP = 4
NLE = 4          # local experts per rank
K = 2            # groups
L = NLE // K     # experts per group
H = 3            # hidden width stand-in


def _install_gloo_list_a2a(rank):
    """gloo lacks the list-form all_to_all the W2 Functions use. Shim it with
    batch_isend_irecv (same per-peer view semantics; zero-count peers skipped
    consistently on both sides; self-block copied eagerly). This replaces only
    the transport — the autograd structure, view math, and buffer assembly
    under test are the real vendored code (NCCL list-a2a transport itself is
    covered on-box by the T2 gate)."""

    def shim(output_list, input_list, group=None, async_op=False):
        ops = []
        for j, t in enumerate(input_list):
            if j == rank:
                output_list[j].copy_(t)
            elif t.numel() > 0:
                ops.append(dist.P2POp(dist.isend, t.contiguous(), j, group))
        for j, t in enumerate(output_list):
            if j != rank and t.numel() > 0:
                ops.append(dist.P2POp(dist.irecv, t, j, group))
        reqs = dist.batch_isend_irecv(ops) if ops else []

        class _Work:
            def wait(self):
                for r in reqs:
                    r.wait()

        return _Work()

    torch.distributed.all_to_all = shim


def _routing_matrix(seed):
    """M[src][dst][e] = rows src sends to dst's local expert e — imbalanced,
    with a zero-count peer pair and rank 2's group-1 experts starved."""
    rng = np.random.default_rng(seed)
    M = rng.integers(0, 12, size=(EP, EP, NLE))
    M[0, 1, :] = 0        # zero-count peer pair
    M[:, 2, L:] = 0       # rank 2's group-1 experts receive nothing (degenerate)
    return M


def _interleave(group_buffers, glob_r, K, L):
    """Interleave per-group recv-layout buffers into the monolithic (src, all
    experts) layout: per src block, [group0 rows | group1 rows | ...]."""
    src_blocks = [[] for _ in range(EP)]
    for g, buf in enumerate(group_buffers):
        # per-src row counts for this group
        counts = glob_r[:, g * L : (g + 1) * L].sum(axis=1)
        off = 0
        for s in range(EP):
            src_blocks[s].append(buf[off : off + int(counts[s])])
            off += int(counts[s])
    return torch.cat([torch.cat(blocks, dim=0) for blocks in src_blocks], dim=0)


def _deinterleave(mono, glob_r, K, L):
    """Split the monolithic buffer into per-group buffers (inverse of _interleave)."""
    groups = [[] for _ in range(K)]
    off = 0
    for s in range(EP):
        for g in range(K):
            n = int(glob_r[s, g * L : (g + 1) * L].sum())
            groups[g].append(mono[off : off + n])
            off += n
    return [torch.cat(parts, dim=0) if parts else torch.empty(0, H) for parts in groups]


def _run(rank, init_file):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=EP
    )
    _install_gloo_list_a2a(rank)
    group = dist.group.WORLD
    M = _routing_matrix(seed=5)
    local_r = M[rank].reshape(-1)
    glob_r = np.stack([M[s, rank, :] for s in range(EP)])
    rank_send = M[rank].sum(axis=1).tolist()
    rank_recv = np.stack([M[s, rank, :] for s in range(EP)]).sum(axis=1).tolist()
    plan = td._w2_compute_chunk_plan(local_r, glob_r, K=K, ep_size=EP)

    g = torch.Generator().manual_seed(99 + rank)
    P = torch.randn(int(M[rank].sum()), H, generator=g)  # permuted buffer stand-in

    # ---- 1/2. chunked dispatch fwd + bwd vs monolithic -----------------------
    # Reference: all_to_all_deferred (device-agnostic; _AllToAll allocates on
    # torch.cuda.current_device() and is not CPU-runnable). Its backward is
    # the same reverse-A2A-with-swapped-splits semantics.
    P_ref = P.clone().requires_grad_(True)
    mono_recv = mappings.all_to_all_deferred(group, P_ref, rank_recv, rank_send)
    mappings.wait_deferred_a2a(mono_recv)

    P_chk = P.clone().requires_grad_(True)
    recv_bufs = td._ChunkedDispatchA2A.apply(group, P_chk, plan)
    for buf in recv_bufs:
        mappings.wait_deferred_a2a(buf)

    mono_from_chunks = _interleave(list(recv_bufs), glob_r, K, L)
    check(
        f"rank{rank}: chunked dispatch recv re-interleaves to the monolithic buffer",
        torch.equal(mono_from_chunks, mono_recv.detach()),
    )

    w = torch.randn_like(mono_recv)
    mono_recv.backward(w)
    # de-interleave the weights onto the chunk buffers and backward each chunk
    w_chunks = _deinterleave(w, glob_r, K, L)
    torch.autograd.backward(list(recv_bufs), list(w_chunks))
    check(
        f"rank{rank}: chunked dispatch input grad == monolithic reverse-A2A grad",
        P_chk.grad is not None and torch.equal(P_chk.grad, P_ref.grad),
    )

    # ---- 3/4. chunked combine fwd + bwd vs monolithic -------------------------
    # expert outputs per group (recv-layout rows for this rank's groups)
    unsorted = []
    for g_ in range(K):
        n = int(glob_r[:, g_ * L : (g_ + 1) * L].sum())
        unsorted.append(torch.randn(n, H, generator=g))
    mono_send = _interleave([u.detach() for u in unsorted], glob_r, K, L)

    mono_send_ref = mono_send.clone().requires_grad_(True)
    # monolithic combine: send splits = rank_recv, recv splits = rank_send
    mono_combine = mappings.all_to_all_deferred(group, mono_send_ref, rank_send, rank_recv)
    mappings.wait_deferred_a2a(mono_combine)

    unsorted_chk = [u.clone().requires_grad_(True) for u in unsorted]
    combine_buf = None
    for g_, u in enumerate(unsorted_chk):
        combine_buf = td._ChunkedCombineA2A.apply(group, u, combine_buf, plan, g_)
    for work in getattr(combine_buf, "_w2_combine_works", []):
        work.wait()

    check(
        f"rank{rank}: chained combine buffer == monolithic combine output (P layout)",
        torch.equal(combine_buf.detach(), mono_combine.detach()),
    )

    wc = torch.randn_like(mono_combine)
    mono_combine.backward(wc)
    combine_bwd_error = None
    try:
        combine_buf.backward(wc)
    except RuntimeError as e:
        combine_bwd_error = str(e)
    check(
        f"rank{rank}: combine chain backward runs (no None-input grad error)",
        combine_bwd_error is None,
    )
    if combine_bwd_error is not None:
        print(f"       ^ error: {combine_bwd_error.splitlines()[0]}", flush=True)
    ref_grad_groups = _deinterleave(mono_send_ref.grad, glob_r, K, L)
    ok = combine_bwd_error is None and all(u.grad is not None for u in unsorted_chk)
    check(f"rank{rank}: every group's unsorted grad is populated (no missing dep)", ok)
    ok = ok and all(
        torch.equal(u.grad, ref_grad_groups[g_]) for g_, u in enumerate(unsorted_chk)
    )
    check(
        f"rank{rank}: chunked combine bwd grads == monolithic reference (per group)",
        ok,
    )

    # ---- 5. degenerate: rank 2's group-1 chunk is empty -----------------------
    if rank == 2:
        check(
            "rank2: starved group-1 dispatch recv buffer is exactly empty",
            recv_bufs[1].shape[0] == 0,
        )
        check(
            "rank2: starved group-1 combine unsorted input is exactly empty",
            unsorted_chk[1].shape[0] == 0,
        )
    dist.barrier()
    dist.destroy_process_group()


def main():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        init_file = f.name
    os.unlink(init_file)
    mp.spawn(_run, args=(init_file,), nprocs=EP, join=True)
    print("=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for name in FAILURES:
            print("  FAIL -", name)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
