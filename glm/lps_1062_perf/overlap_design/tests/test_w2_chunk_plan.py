#!/usr/bin/env python3
"""T1 CPU decomposition test for W2 — BT_MOE_A2A_PIPELINE (chunked MoE A2A).

Verifies `_w2_compute_chunk_plan` (megatron/core/transformer/moe/
token_dispatcher.py) against a brute-force simulation of the unchunked
alltoall dispatcher layout — the bitwise-parity backbone of the chunked
pipeline (design: ../DESIGN_helmholtz.md §2/§5):

  1. split conservation: per-group input/output splits sum to the unchunked
     input_splits / output_splits;
  2. dispatch send views: the per-(group, dest) bounds carve P (the
     expert-id-ordered permuted buffer) into exactly the rows of each dest's
     group-g experts, in order, with no gaps or overlaps;
  3. dispatch recv views: each group's recv buffer is the (src,
     expert-in-group) restriction of the unchunked recv layout;
  4. combine reassembly (THE bitwise-critical property): writing each group's
     combine output into the shared combine buffer at combine_recv_bounds
     reproduces the unchunked combine output — which mirrors P's layout —
     byte-for-byte (row ids);
  5. tokens_per_expert per group == the unchunked slices;
  6. sort metadata: sort_input_chunk / restore_output_chunk / probs_keep_idxs
     reproduce the corresponding restrictions of the unchunked permutations;
  7. edge cases: a (peer, group) pair with ZERO rows (list-A2A zero-count
     path, R2), a whole expert with zero rows (Fp8Padding boundary logic
     downstream), and a 90%-to-one-expert imbalance.

Runs on Mac CPU, single process, no distributed init (the plan is pure host
math). Target the vendored mcore with BT_TEST_MCORE_PATH if the default
walk-up does not find it.

Usage: python test_w2_chunk_plan.py
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

from megatron.core.transformer.moe.token_dispatcher import (  # noqa: E402
    _W2PipelineConfig,
    _w2_compute_chunk_plan,
)

FAILURES = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name, flush=True)
    if not cond:
        FAILURES.append(name)


EP = 16  # EP ranks
LE = 16  # local experts per rank
E = EP * LE  # global experts
K = 2
L = LE // K


def make_routing(T, seed, zero_expert=None, zero_dest_group=None, spiky=False):
    """Random multihot routing_map [T, E] (~top-8 per row).

    zero_expert: a global expert id that gets no rows (whole-expert zero case).
    zero_dest_group: (dest_rank, group) whose experts get no rows from THIS
        rank (zero-count peer path).
    spiky: route ~90% of selections to expert 0 (imbalance case).
    """
    g = torch.Generator().manual_seed(seed)
    routing = torch.zeros(T, E, dtype=torch.bool)
    for t in range(T):
        if spiky:
            idx = [0] * 7 + [int(torch.randint(0, E, (1,), generator=g))]
        else:
            idx = torch.randperm(E, generator=g)[:8].tolist()
        routing[t, idx] = True
    if zero_expert is not None:
        routing[:, zero_expert] = False
    if zero_dest_group is not None:
        d, gr = zero_dest_group
        routing[:, d * LE + gr * L : d * LE + (gr + 1) * L] = False
    return routing


def p_layout(routing):
    """Brute-force permuted buffer P as a row-id list [(token, expert)],
    expert-id order (the unchunked dispatch buffer layout)."""
    T, E_ = routing.shape
    rows = []
    for e in range(E_):
        for t in range(T):
            if routing[t, e]:
                rows.append((t, e))
    return rows


def recv_layout(global_counts):
    """Brute-force unchunked recv buffer as row ids (src, e_local, i):
    (src, expert) block order, expert-id order within a src block."""
    rows = []
    for s in range(EP):
        for e in range(LE):
            for i in range(int(global_counts[s, e])):
                rows.append((s, e, i))
    return rows


def run_case(name, routing, global_counts):
    print(f"--- case: {name}")
    local_counts = routing.sum(dim=0).numpy().astype(np.int64)  # [E]
    glob = np.asarray(global_counts, dtype=np.int64)  # [EP, LE]
    plan = _w2_compute_chunk_plan(local_counts, glob, K=K, ep_size=EP)
    P = p_layout(routing)
    recv = recv_layout(glob)

    # 1. split conservation
    in_splits_today = local_counts.reshape(EP, LE).sum(axis=1)
    out_splits_today = glob.sum(axis=1)
    check(
        f"{name}: input_splits conserved",
        [int(x) for x in in_splits_today]
        == [sum(plan.input_splits[g][d] for g in range(K)) for d in range(EP)],
    )
    check(
        f"{name}: output_splits conserved",
        [int(x) for x in out_splits_today]
        == [sum(plan.output_splits[g][s] for g in range(K)) for s in range(EP)],
    )
    check(f"{name}: total_rows == len(P)", plan.total_rows == len(P))

    # 2. dispatch send views carve P exactly (per-view content + full partition)
    covered = [False] * len(P)
    ok = True
    for g in range(K):
        for d in range(EP):
            start, length = plan.dispatch_send_bounds[g][d]
            view = P[start : start + length]
            # expected: rows for dest d's group-g experts, expert-id order
            expected = [
                (t, e)
                for e in range(d * LE + g * L, d * LE + (g + 1) * L)
                for t in range(routing.shape[0])
                if routing[t, e]
            ]
            if view != expected:
                ok = False
            for i in range(start, start + length):
                if covered[i]:
                    ok = False  # overlap
                covered[i] = True
    check(f"{name}: dispatch send views == per-(dest,group) row sets, in order", ok)
    check(f"{name}: dispatch send views partition P (no gaps/overlaps)", ok and all(covered))

    # 3. dispatch recv views: per-(group, src) bounds tile recv_g contiguously
    # and each per-src segment equals src s's group-g rows in (src, e-in-g)
    # order (the unchunked recv layout restricted to the group).
    ok = True
    for g in range(K):
        expected_g = [r for r in recv if g * L <= r[1] < (g + 1) * L]
        if sum(l for _, l in plan.dispatch_recv_bounds[g]) != len(expected_g):
            ok = False
        pos = 0
        for s in range(EP):
            start, length = plan.dispatch_recv_bounds[g][s]
            if start != pos:
                ok = False  # not contiguous tiling
            seg = expected_g[start : start + length]
            exp_s = [r for r in expected_g if r[0] == s]
            if seg != exp_s:
                ok = False
            pos += length
    check(f"{name}: dispatch recv views == (src, expert-in-group) restriction", ok)

    # 4. THE combine reassembly check: assemble the shared combine buffer from
    # per-group writes at combine_recv_bounds; it must equal P byte-for-byte.
    combine_buf = [None] * len(P)
    for g in range(K):
        for s in range(EP):
            cstart, clen = plan.combine_recv_bounds[g][s]
            dstart, dlen = plan.dispatch_send_bounds[g][s]
            # the rows src s returns to me = the rows I sent to s (group g)
            assert clen == dlen, f"pairwise mismatch at g={g} s={s}"
            combine_buf[cstart : cstart + clen] = P[dstart : dstart + dlen]
    check(f"{name}: combine reassembly == P byte-for-byte", combine_buf == P)

    # combine send bounds: carve the (src, expert-in-group) unsorted_g buffer
    ok = True
    for g in range(K):
        total = sum(l for _, l in plan.combine_send_bounds[g])
        ok = ok and total == sum(plan.output_splits[g])
        ok = ok and [l for _, l in plan.combine_send_bounds[g]] == plan.output_splits[g]
    check(f"{name}: combine send bounds == per-group recv-row layout", ok)

    # 5. tokens_per_expert per group == unchunked slices
    tpe_today = glob.sum(axis=0)
    ok = all(
        plan.tokens_per_expert[g] == [int(x) for x in tpe_today[g * L : (g + 1) * L]]
        for g in range(K)
    )
    check(f"{name}: tokens_per_expert slices", ok)

    # 6. sort metadata (CPU-constructed config)
    cfg = _W2PipelineConfig(K, L, 1, EP, "cpu")
    # sort_input_chunk must map (src, e-in-g) chunk order to expert-major:
    # applying it to a chunk-count split yields expert-major order.
    for g in range(K):
        counts_g = glob[:, g * L : (g + 1) * L].reshape(-1)  # (src, e-in-g) ravel
        chunks = []
        off = 0
        for c in counts_g:
            chunks.append(list(range(off, off + int(c))))
            off += int(c)
        sorted_rows = [r for i in cfg.sort_input_chunk.tolist() for r in chunks[i]]
        expected = []
        for e in range(L):
            for s in range(EP):
                expected.extend(chunks[s * L + e])
        if sorted_rows != expected:
            ok = False
    check(f"{name}: sort_input_chunk reproduces expert-major restriction", ok)
    # probs_keep_idxs[g] must select group g's (src, e) flat block indices in
    # (src, e-in-g) order
    ok = True
    for g in range(K):
        expected_idx = [s * LE + g * L + i for s in range(EP) for i in range(L)]
        ok = ok and (cfg.probs_keep_idxs[g].tolist() == expected_idx)
    check(f"{name}: probs_keep_idxs selects (src, e-in-g) blocks", ok)
    # restore_output_chunk is the inverse permutation of sort_input_chunk
    inv = [0] * (EP * L)
    for i, j in enumerate(cfg.sort_input_chunk.tolist()):
        inv[j] = i
    check(
        f"{name}: restore_output_chunk inverts sort_input_chunk",
        inv == cfg.restore_output_chunk.tolist(),
    )


def main():
    g = np.random.default_rng(7)
    # baseline: random top-8 routing, random receive matrix
    routing = make_routing(97, 0)
    glob = g.integers(0, 5, size=(EP, LE))
    run_case("baseline", routing, glob)

    # zero-count (peer, group): this rank sends nothing to dest 3's group 1
    routing_zp = make_routing(97, 1, zero_dest_group=(3, 1))
    run_case("zero (peer, group) on send side", routing_zp, glob)

    # zero-count on the receive side too
    glob_z = glob.copy()
    glob_z[5, L:] = 0  # src 5 sends nothing for my group-1 experts
    run_case("zero (peer, group) both sides", routing_zp, glob_z)

    # whole expert with zero rows (expert 42 = dest 2, group 1, local 10)
    routing_ze = make_routing(97, 2, zero_expert=42)
    glob_ze = glob.copy()
    glob_ze[:, 42 % LE] = 0
    run_case("whole expert zero rows", routing_ze, glob_ze)

    # imbalance: ~90% of selections to expert 0
    routing_sp = make_routing(97, 3, spiky=True)
    glob_sp = g.integers(0, 5, size=(EP, LE))
    glob_sp[:, 0] += 90
    run_case("imbalance 90% expert 0", routing_sp, glob_sp)

    # empty row counts everywhere for group 0 on the send side
    routing_g0 = make_routing(64, 4)
    routing_g0[:, torch.arange(E) % LE < L] = False  # keep only group-1 experts
    run_case("whole group 0 empty (send side)", routing_g0, glob)

    if FAILURES:
        raise SystemExit(f"{len(FAILURES)} FAILURES: {FAILURES}")
    print("OK — all W2 chunk-plan decomposition checks passed")


if __name__ == "__main__":
    main()
