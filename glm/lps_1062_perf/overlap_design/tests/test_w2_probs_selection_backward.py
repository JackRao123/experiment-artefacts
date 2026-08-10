#!/usr/bin/env python3
"""Off-box guard for the W2 T2-gate defect class (subset-vs-permutation
autograd contract of the chunk-sort used for per-group probs selection).

The T2 gate failed on-box with: "GeneratedBackwardFor_te_moe_chunk_sort_fwd_
defaultBackward returned an invalid gradient at index 3 — got [33216] but
expected shape compatible with [65440]" — the TE fused sort's generated
backward assumes a PERMUTATION (output rows == input rows); the W2 probs
selection used it as a SUBSET (group rows out of the full buffer), so the
backward returned a group-sized grad for a full-size input.

Why it escaped (fibonacci requirement a): the Mac suites never execute TE's
generated backwards (no TE / no GPU on the Mac); the T1 decomposition tests
validate the index MATH, not the autograd contract of the kernel consuming
those indices; hilbert's T1b covers the hand-written Functions on gloo. The
fused-subset misuse is invisible at the plan level by construction — it
needed the on-box T2 gate, which caught it exactly as designed.

This test (Mac CPU, no TE needed — the unfused path is pure torch):

  1. the FIXED call — unfused sort_chunks_by_idxs as a subset selection —
     produces a backward whose grad w.r.t. the full-size input is FULL-SIZE,
     with values landing exactly at the selected rows and zeros elsewhere
     (bitwise vs a manual scatter reference);
  2. the two groups' selection grads are disjoint and their sum equals the
     full scatter (the engine-add path the dispatcher relies on);
  3. the permutation invariant the FUSED calls depend on: the config's
     sort_input_chunk / restore_output_chunk are full permutations (every
     chunk index covered exactly once) — the precondition for TE's generated
     backward to be shape-correct — for several (K, L, EP) geometries;
  4. a torch.autograd.gradcheck-style contract probe: for every sort_chunks
     call shape used in the W2 path, grad.numel() == input.numel().

Usage: python test_w2_probs_selection_backward.py
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

from megatron.core.transformer.moe.moe_utils import sort_chunks_by_idxs  # noqa: E402
from megatron.core.transformer.moe.token_dispatcher import (  # noqa: E402
    _W2PipelineConfig,
    _w2_compute_chunk_plan,
)

FAILURES = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name, flush=True)
    if not cond:
        FAILURES.append(name)


def _manual_scatter_reference(full_rows, counts, keep_idxs, grad_out):
    """Full-size grad: grad_out rows scattered to the kept chunks' rows."""
    ref = torch.zeros(full_rows)
    chunks = torch.split(torch.arange(full_rows), [int(c) for c in counts])
    kept_rows = [chunks[i] for i in keep_idxs]
    offset = 0
    for rows in kept_rows:
        ref[rows] = grad_out[offset : offset + len(rows)]
        offset += len(rows)
    return ref


def test_subset_selection_backward():
    """The fixed call: unfused subset selection, backward is full-size and exact."""
    EP, LE, K = 4, 4, 2  # small geometry
    L = LE // K
    T = 37
    g = torch.Generator().manual_seed(0)
    # random multihot routing -> local counts; random receive matrix
    routing = torch.zeros(T, EP * LE, dtype=torch.bool)
    for t in range(T):
        routing[t, torch.randperm(EP * LE, generator=g)[:3]] = True
    local_counts = routing.sum(dim=0).numpy()
    import numpy as np

    glob = np.random.default_rng(3).integers(0, 4, size=(EP, LE))
    plan = _w2_compute_chunk_plan(local_counts, glob, K=K, ep_size=EP)
    cfg = _W2PipelineConfig(K, L, 1, EP, "cpu")

    full_rows = int(glob.sum())
    counts_flat = glob.reshape(-1)  # numpy: the unfused path calls .tolist()
    for grp in range(K):
        probs_full = torch.randn(full_rows, generator=g, requires_grad=True)
        keep = cfg.probs_keep_idxs_host[grp]
        probs_g = sort_chunks_by_idxs(probs_full, counts_flat, keep, fused=False)[0]
        assert probs_g.numel() == plan.total_recv_rows[grp]
        grad_out = torch.randn(probs_g.numel(), generator=g)
        probs_g.backward(grad_out)
        grad = probs_full.grad
        check(
            f"sec1 group{grp}: subset-selection grad is FULL-SIZE "
            f"({grad.numel()} == {full_rows})",
            grad.numel() == full_rows,
        )
        ref = _manual_scatter_reference(full_rows, counts_flat, list(keep), grad_out)
        check(
            f"sec1 group{grp}: grad values land exactly at selected rows",
            torch.equal(grad, ref),
        )
    # sec2: the two groups' grads are disjoint and sum to the full scatter
    probs_full = torch.randn(full_rows, generator=g, requires_grad=True)
    grads = []
    grad_outs = []
    for grp in range(K):
        probs_full.grad = None
        keep = cfg.probs_keep_idxs_host[grp]
        probs_g = sort_chunks_by_idxs(probs_full, counts_flat, keep, fused=False)[0]
        grad_out = torch.randn(probs_g.numel(), generator=g)
        probs_g.backward(grad_out)
        grads.append(probs_full.grad.clone())
        grad_outs.append(grad_out)
    overlap = (grads[0] != 0) & (grads[1] != 0)
    check("sec2: group selection grads are disjoint", not overlap.any().item())
    ref_full = _manual_scatter_reference(
        full_rows,
        counts_flat,
        list(cfg.probs_keep_idxs_host[0]) + list(cfg.probs_keep_idxs_host[1]),
        torch.cat(grad_outs),
    )
    check(
        "sec2: group grads sum to the full scatter (engine-add exactness)",
        torch.equal(grads[0] + grads[1], ref_full),
    )


def test_permutation_invariant():
    """The FUSED calls' precondition: sort/restore chunk idxs are full
    permutations (every chunk covered exactly once) for several geometries."""
    for EP, LE, K in ((16, 16, 2), (4, 4, 2), (3, 6, 3), (2, 8, 4)):
        L = LE // K
        cfg = _W2PipelineConfig(K, L, 1, EP, "cpu")
        n = EP * L
        for name, idxs in (
            ("sort_input_chunk", cfg.sort_input_chunk),
            ("restore_output_chunk", cfg.restore_output_chunk),
        ):
            seen = torch.zeros(n, dtype=torch.long)
            seen[idxs] += 1
            check(
                f"sec3 EP{EP}/LE{LE}/K{K}: {name} is a full permutation",
                bool((seen == 1).all()),
            )
            # the fused calls pass len(split_sizes) == n chunks — the
            # generated backward is only shape-correct when
            # len(sorted_idxs) == len(split_sizes):
            check(
                f"sec3 EP{EP}/LE{LE}/K{K}: {name} covers all {n} chunks",
                idxs.numel() == n,
            )


def test_grad_numel_contract():
    """Contract probe: every sort_chunks call shape used in the W2 path must
    return grad.numel() == input.numel() through autograd."""
    EP, LE, K = 4, 4, 2
    L = LE // K
    cfg = _W2PipelineConfig(K, L, 1, EP, "cpu")
    import numpy as np

    counts = np.array([2, 0, 3, 1, 0, 2, 1, 3, 1, 1, 0, 2, 2, 3, 1, 0])  # incl. zero chunks
    # (a) the unfused subset selection (probs path)
    x = torch.randn(sum(counts), requires_grad=True)
    out = sort_chunks_by_idxs(x, counts, cfg.probs_keep_idxs_host[0], fused=False)[0]
    out.backward(torch.randn(out.numel()))
    check("sec4a: subset selection grad numel == input numel", x.grad.numel() == x.numel())
    # (b) full-permutation sorts (token sort / unsort shapes), unfused here
    # (CPU has no TE; the permutation property is what the fused kernel needs —
    # sec3 — and the unfused twin proves the index math is grad-shape-safe)
    x = torch.randn(sum(counts), requires_grad=True)
    out = sort_chunks_by_idxs(x, counts, cfg.sort_input_chunk, fused=False)[0]
    out.backward(torch.randn(out.numel()))
    check("sec4b: full-permutation sort grad numel == input numel", x.grad.numel() == x.numel())


def main():
    test_subset_selection_backward()
    test_permutation_invariant()
    test_grad_numel_contract()
    if FAILURES:
        raise SystemExit(f"{len(FAILURES)} FAILURES: {FAILURES}")
    print("OK — all W2 probs-selection backward guard checks passed")


if __name__ == "__main__":
    main()
