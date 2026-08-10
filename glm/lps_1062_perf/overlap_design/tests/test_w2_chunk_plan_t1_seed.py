#!/usr/bin/env python3
"""T1-seed property test for W2 — _w2_compute_chunk_plan (BT_MOE_A2A_PIPELINE).

Second-reviewer artifact (hilbert, 2026-08-09). Drives the REAL
`_w2_compute_chunk_plan` from the vendored tree against a consistent
EP-wide routing matrix with IMBALANCED counts and zero-count peers/experts
(the TRACE_ACCEPTANCE W2.3 cases) and checks, for every rank and group:

  1. NCCL consistency, dispatch: rank r's send-to-j view length == rank j's
     recv-from-r view length == M[r,j,group experts].sum().
  2. NCCL consistency, combine: rank r's combine send-to-j length == rank j's
     combine recv-from-r length == M[j,r,group].sum() (rows j originally sent
     to r's group-g experts, which r returns).
  3. Combine-buffer tiling: the union of all groups' combine recv views
     exactly tiles the UNCHUNKED combine buffer — per-sender-d blocks of
     rank_send[d] rows (what this rank originally sent d), in d order,
     local-expert order within each block; no overlaps, no gaps, no overflow.
  4. Restriction: per-group input/output splits sum to the unchunked
     input_splits / output_splits.
  5. Buffer size: plan.total_rows == sum(rank_send) (the unchunked combine
     buffer is this rank's tokens x topk, NOT sum(rank_recv)).

A SYMMETRIC fixture cannot catch send/recv-side confusion (local == glob
numerically when all ranks route identically) — imbalanced M is mandatory.

STATUS 2026-08-09: checks 2, 3, 5 FAIL on the in-progress diff — the
combine-recv side of the plan is computed from receive-side counts
(glob_g / rank_recv / out_g) where it must use send-side counts
(local_g / rank_send / in_g). The verified fix (20/20 random seeds pass):

    # in _w2_compute_chunk_plan, replace the combine_recv_bounds block:
    off_in_sender_block = (
        local_g[:, :g, :].sum(axis=(1, 2)) if g > 0 else np.zeros(ep_size, np.int64)
    )
    plan.combine_recv_bounds.append(
        [(send_base[d] + int(off_in_sender_block[d]), in_g[d]) for d in range(ep_size)]
    )
    # and size the shared combine buffer from the send side:
    plan.total_rows = int(rank_send.sum())   # not sum(plan.total_recv_rows)

Runs on Mac CPU. BT_TEST_MCORE_PATH overrides the tree walk-up.

Usage: python test_w2_chunk_plan_t1_seed.py
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

import megatron.core.transformer.moe.token_dispatcher as td  # noqa: E402

FAILURES = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name)
    if not cond:
        FAILURES.append(name)


def run_case(seed, EP, NLE, K):
    L = NLE // K
    rng = np.random.default_rng(seed)
    # M[src][dst][e] = rows src sends to dst's local expert e — imbalanced,
    # with a zero-count peer pair and zero-count experts.
    M = rng.integers(0, 40, size=(EP, EP, NLE))
    M[0 % EP, 1 % EP, :] = 0
    if EP > 3:
        M[2, 3, 1:3] = 0

    plans = []
    for r in range(EP):
        local_r = M[r].reshape(-1)  # rows r sends to each global expert
        glob_r = np.stack([M[s, r, :] for s in range(EP)])  # rows r receives per (src, expert)
        plans.append(td._w2_compute_chunk_plan(local_r, glob_r, K=K, ep_size=EP))

    tag = f"EP{EP} nle{NLE} K{K} seed{seed}"

    # 1. dispatch NCCL consistency
    ok = True
    for r in range(EP):
        for j in range(EP):
            for g in range(K):
                expect = int(M[r, j, g * L : (g + 1) * L].sum())
                ok &= plans[r].dispatch_send_bounds[g][j][1] == expect
                ok &= plans[j].dispatch_recv_bounds[g][r][1] == expect
    check(f"{tag}: dispatch NCCL consistency", ok)

    # 2. combine NCCL consistency
    ok = True
    for r in range(EP):
        for j in range(EP):
            for g in range(K):
                expect = int(M[j, r, g * L : (g + 1) * L].sum())
                ok &= plans[r].combine_send_bounds[g][j][1] == expect
                ok &= plans[j].combine_recv_bounds[g][r][1] == expect
    check(f"{tag}: combine NCCL consistency", ok)

    # 3. combine-buffer tiling vs the unchunked layout
    ok = True
    for r in range(EP):
        rank_send = M[r].sum(axis=1)
        total = int(rank_send.sum())
        covered = np.zeros(total, dtype=bool)
        ref_off = np.concatenate([[0], np.cumsum(rank_send)])
        for g in range(K):
            for d in range(EP):
                start, length = plans[r].combine_recv_bounds[g][d]
                ref_start = int(ref_off[d]) + int(M[r, d, : g * L].sum())
                ref_len = int(M[r, d, g * L : (g + 1) * L].sum())
                if not (start == ref_start and length == ref_len):
                    ok = False
                if length > 0:
                    if start + length > total or covered[start : start + length].any():
                        ok = False
                    covered[start : start + length] = True
        if not covered.all():
            ok = False
    check(f"{tag}: combine recv views tile the unchunked buffer exactly", ok)

    # 4. splits restriction
    ok = True
    for r in range(EP):
        in_sum = np.array(plans[r].input_splits).sum(axis=0)
        out_sum = np.array(plans[r].output_splits).sum(axis=0)
        ok &= bool((in_sum == M[r].sum(axis=1)).all())
        ok &= bool((out_sum == np.stack([M[s, r].sum() for s in range(EP)])).all())
    check(f"{tag}: per-group splits sum to unchunked splits", ok)

    # 5. combine buffer size from the send side
    ok = all(plans[r].total_rows == int(M[r].sum()) for r in range(EP))
    check(f"{tag}: plan.total_rows == sum(rank_send)", ok)


def run_degenerate_case():
    """K=2 with an ALL-empty group on one rank (fibonacci's case): every expert
    in group 0 of rank 1 receives zero rows from every source, and rank 2
    sends zero rows to every group-1 expert. The degenerate chunk must still
    fire its a2a with all-zero counts for list alignment: every group must
    produce exactly EP (start, length) entries on every rank, zero-length
    entries included, and the zero group's offsets must stay consistent
    (a zero-length group occupies its position in every buffer layout)."""
    EP, NLE, K = 4, 4, 2
    L = NLE // K
    rng = np.random.default_rng(7)
    M = rng.integers(0, 40, size=(EP, EP, NLE))
    M[:, 1, 0:2] = 0  # rank 1's group-0 experts receive zero rows from all sources
    M[2, :, 2:4] = 0  # rank 2 sends zero rows to every dest's group-1 experts
    tag = "degenerate(all-empty group)"

    plans = []
    for r in range(EP):
        local_r = M[r].reshape(-1)
        glob_r = np.stack([M[s, r, :] for s in range(EP)])
        plans.append(td._w2_compute_chunk_plan(local_r, glob_r, K=K, ep_size=EP))

    # Structural list alignment: K groups x EP entries on every rank, in both
    # directions, regardless of emptiness (the a2a lists must line up across
    # ranks even when a whole group is zero).
    ok = True
    for r in range(EP):
        for bounds in (
            plans[r].dispatch_send_bounds,
            plans[r].dispatch_recv_bounds,
            plans[r].combine_send_bounds,
            plans[r].combine_recv_bounds,
        ):
            ok &= len(bounds) == K and all(len(b) == EP for b in bounds)
    check(f"{tag}: K groups x EP entries on every rank/direction (list alignment)", ok)

    # The degenerate group's entries are all zero-length, and the empty
    # group's offsets stay consistent (a zero-length group occupies its slot
    # in every per-peer list; per-group recv buffers are 0-based, so an empty
    # group-0 leaves every start at 0).
    ok = True
    r = 1  # rank 1's group-0 experts receive nothing from anyone
    ok &= all(length == 0 for (_, length) in plans[r].dispatch_recv_bounds[0])
    ok &= all(length == 0 for (_, length) in plans[r].combine_send_bounds[0])
    ok &= all(start == 0 for (start, _) in plans[r].dispatch_recv_bounds[0])
    ok &= all(start == 0 for (start, _) in plans[r].combine_send_bounds[0])
    r = 2  # rank 2 sends nothing to any dest's group-1 experts
    ok &= all(length == 0 for (_, length) in plans[r].dispatch_send_bounds[1])
    # ... and each zero-length group-1 send view still sits immediately after
    # dest d's group-0 rows in P (the empty group occupies its slot)
    ok &= all(
        plans[r].dispatch_send_bounds[1][d][0]
        == plans[r].dispatch_send_bounds[0][d][0] + plans[r].dispatch_send_bounds[0][d][1]
        for d in range(EP)
    )
    check(f"{tag}: all-empty group entries are zero-length with consistent offsets", ok)

    # The full check battery still holds with the degenerate group present.
    ok = True
    for r in range(EP):
        for j in range(EP):
            for g in range(K):
                expect = int(M[r, j, g * L : (g + 1) * L].sum())
                ok &= plans[r].dispatch_send_bounds[g][j][1] == expect
                ok &= plans[j].dispatch_recv_bounds[g][r][1] == expect
                expect_c = int(M[j, r, g * L : (g + 1) * L].sum())
                ok &= plans[r].combine_send_bounds[g][j][1] == expect_c
                ok &= plans[j].combine_recv_bounds[g][r][1] == expect_c
    check(f"{tag}: NCCL consistency dispatch+combine (incl. zero entries)", ok)

    ok = True
    for r in range(EP):
        rank_send = M[r].sum(axis=1)
        total = int(rank_send.sum())
        covered = np.zeros(total, dtype=bool)
        ref_off = np.concatenate([[0], np.cumsum(rank_send)])
        for g in range(K):
            for d in range(EP):
                start, length = plans[r].combine_recv_bounds[g][d]
                ref_start = int(ref_off[d]) + int(M[r, d, : g * L].sum())
                ref_len = int(M[r, d, g * L : (g + 1) * L].sum())
                if not (start == ref_start and length == ref_len):
                    ok = False
                if length > 0:
                    if start + length > total or covered[start : start + length].any():
                        ok = False
                    covered[start : start + length] = True
        if not covered.all():
            ok = False
    check(f"{tag}: combine recv views tile the unchunked buffer exactly", ok)

    ok = all(plans[r].total_rows == int(M[r].sum()) for r in range(EP))
    check(f"{tag}: plan.total_rows == sum(rank_send)", ok)


def main():
    for seed in range(6):
        run_case(seed, EP=4, NLE=4, K=2)
    run_case(100, EP=2, NLE=2, K=2)   # minimal EP
    run_case(101, EP=4, NLE=8, K=4)   # K = nle
    run_case(102, EP=3, NLE=6, K=3)   # odd EP
    run_degenerate_case()             # all-empty group on one rank

    print("\n" + ("=" * 60))
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S) — see the module docstring for the verified fix:")
        for name in FAILURES:
            print("  FAIL -", name)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
