#!/usr/bin/env python3
"""CPU tests for option 6 — BT_MOE_PROBS_BWD_REORDER (probs-reverse bwd-chain reorder).

The lever (W2V2_DECISION_stub.md, un-parked 2026-08-10): today the probs grad
is produced late in the layer's backward (the fused with-probs sort emits
d(tokens) and d(probs) in ONE autograd node, gated on the full MLP backward),
so the probs reverse A2A lands exposed at the end of the backward window
(the W1 residual). Option 6 = (i) split the probs sort edge
(moe_utils.sort_probs_chunks_early_bwd — its backward depends only on
d(permuted_probs)), (ii) seq-bump the probs-path nodes above the fc1-dgrad
pack, (iii) defer the reverse's compute-stream wait to the tokens reverse
(mappings.py bwd_defer early/late carrier).

What is proven here (Mac CPU, no TE/CUDA — the fused sort ops are stubbed
with reference permutations; the REAL kernel's bitwise equivalence vs the
fused with-probs path is the on-box gate's job):

  1. sort Function semantics: forward == reference chunk permutation;
     backward == exact inverse permutation (values + grad wiring through the
     stubbed row_id_map path).
  2. THE MECHANISM PROOF (engine order): with the split + INT_MAX seq-bumps,
     the probs path's backward nodes (probs_sort.bwd, probs_a2a.bwd) fire
     BEFORE a simulated fc1-dgrad node; controls: without the bump fc1 fires
     first (natural seq order), and the joint-sort form holds the probs grad
     behind fc1 (today's hostage).
  3. mappings.py bwd_defer carrier protocol (2-proc gloo): normal early→late
     order (early stashes, late waits both, values == reverse-A2A reference);
     the pathological late-first order (early waits inline + loud warning);
     the missing-handle case (loud warning). world_size==1 bypass intact.
  4. Gate + arm predicate: env off/on; armable() reasons for each exclusion
     (no W1 comm / W2 / drop_and_pad / TP>1 / single expert / no fusion).
  5. Source guards (AST): the wiring forms that must not drift (the fused
     with-probs call preserved in the else branch; bwd_defer only under the
     gate; no new host sync (.tolist()) in the sort Function).

Usage: python3 test_option6_probs_bwd_reorder.py   (BT_TEST_MCORE_PATH to target a tree)
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
        p = os.path.join(d, "server", "vendor", "megatron-bridge", "3rdparty", "Megatron-LM")
        if os.path.isdir(p):
            return p
        d = os.path.dirname(d)
    p = os.path.expanduser("~/Documents/trainers/server/vendor/megatron-bridge/3rdparty/Megatron-LM")
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
import megatron.core.transformer.moe.moe_utils as moe_utils  # noqa: E402
import megatron.core.transformer.moe.token_dispatcher as token_dispatcher  # noqa: E402
from megatron.core.transformer.moe.shared_experts import (  # noqa: E402
    set_tensor_grad_fn_sequence_sr,
)

FAILURES = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name, flush=True)
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# CPU reference stubs for the TE fused sort ops (row permutations).
# row_id_map semantics (matches te_moe.chunk_sort_fwd): out[j] = in[rid[j]].
# ---------------------------------------------------------------------------

def _chunk_row_permutation(split_sizes, sorted_idxs):
    splits = list(split_sizes)
    rid = []
    offset = 0
    starts = []
    for s in splits:
        starts.append(offset)
        offset += s
    for c in sorted_idxs:
        rid.extend(range(starts[c], starts[c] + splits[c]))
    return torch.tensor(rid, dtype=torch.long)


def _stub_sort_fwd(inp, split_sizes, sorted_idxs):
    rid = _chunk_row_permutation(split_sizes.tolist(), sorted_idxs.tolist())
    return inp[rid], rid


def _stub_sort_bwd(d_permuted, row_id_map, num_tokens, hidden_size):
    d_global = torch.zeros(num_tokens, hidden_size, dtype=d_permuted.dtype)
    d_global[row_id_map] = d_permuted
    return d_global


def _install_sort_stubs():
    # The real ops need TE (CUDA box); stub the reference permutations and
    # flip the availability flag (the same monkeypatch pattern the W3 suite
    # uses for the CUDA-touching RNG helpers).
    moe_utils.HAVE_TE = True
    moe_utils.fused_sort_chunks_by_index_with_row_id_map = _stub_sort_fwd
    moe_utils.fused_sort_chunks_by_index_bwd = _stub_sort_bwd


# ---------------------------------------------------------------------------
# sec 1 — sort Function semantics (values + the independent backward edge)
# ---------------------------------------------------------------------------

def test_1_sort_function_semantics():
    print("\n== sec1: _SortProbsEarlyBwd values (stubbed fused ops) ==")
    _install_sort_stubs()
    split_sizes = torch.tensor([3, 0, 5, 2])  # incl. an empty chunk
    sorted_idxs = torch.tensor([2, 0, 3, 1])  # a chunk permutation
    g = torch.Generator().manual_seed(0)
    probs = torch.randn(sum(split_sizes), generator=g)

    p1 = probs.detach().clone().requires_grad_(True)
    out = moe_utils.sort_probs_chunks_early_bwd(p1, split_sizes, sorted_idxs)
    # reference: the same row permutation computed directly
    rid = _chunk_row_permutation(split_sizes.tolist(), sorted_idxs.tolist())
    check("sec1: forward == reference row permutation", torch.equal(out, probs[rid]))

    d_out = torch.randn(out.shape, generator=torch.Generator().manual_seed(1))
    p1.grad = None
    out.backward(d_out)
    # reference inverse: d_in[rid[j]] = d_out[j]
    ref = torch.zeros_like(probs)
    ref[rid] = d_out
    check("sec1: backward == exact inverse permutation", torch.equal(p1.grad, ref))


# ---------------------------------------------------------------------------
# sec 2 — THE MECHANISM PROOF: engine order with split + seq-bump
# ---------------------------------------------------------------------------

ORDER = []


class _Log(torch.autograd.Function):
    """A named logging node for the engine-order probe."""

    @staticmethod
    def forward(ctx, x, tag):
        ctx.tag = tag
        return x * 2.0

    @staticmethod
    def backward(ctx, g):
        ORDER.append(ctx.tag)
        return g * 2.0, None


class _JointSort(torch.autograd.Function):
    """Today's fused with-probs form: ONE node emitting both grads — its
    backward fires only when BOTH output grads have arrived (the hostage)."""

    @staticmethod
    def forward(ctx, t, p):
        return t * 1.0, p * 1.0

    @staticmethod
    def backward(ctx, gt, gp):
        ORDER.append("joint_sort.bwd")
        return gt, gp


def _build_graph(split, bump):
    """The layer-backward skeleton: tokens chain (fc1 <- actscale <- fc2) and
    the probs path (probs_a2a <- probs_sort <- actscale). d(permuted_probs)
    arrives at actscale.bwd (mid-chain); d(sorted_tokens) at fc1.bwd (end).
    """
    global ORDER
    ORDER = []
    g = torch.Generator().manual_seed(7)
    t = torch.randn(6, generator=g, requires_grad=True)
    p = torch.randn(6, generator=g, requires_grad=True)

    pa = _Log.apply(p, "probs_a2a.bwd")  # stand-in for the deferred probs A2A
    if bump:
        set_tensor_grad_fn_sequence_sr(pa, torch.iinfo(torch.int).max)
    if split:
        sp = _Log.apply(pa, "probs_sort.bwd")  # stand-in for _SortProbsEarlyBwd
        if bump:
            set_tensor_grad_fn_sequence_sr(sp, torch.iinfo(torch.int).max)
        st = _Log.apply(t, "tokens_sort.bwd")
    else:
        both = _JointSort.apply(t, pa)
        st = both[0]
        sp = both[1]

    h1 = _Log.apply(st, "fc1.bwd")  # fc1: created LAST in fwd => highest natural seq
    h2 = _Log.apply(h1 * 1.0 + sp * 0.0, "actscale.bwd")  # d(sp) ready at actscale.bwd
    out = _Log.apply(h2, "fc2.bwd")
    out.sum().backward()
    return ORDER


def test_2_engine_order_mechanism():
    print("\n== sec2: engine-order mechanism (split + bump vs controls) ==")
    order = _build_graph(split=True, bump=True)
    check("sec2: split+bump — probs path fires ahead of the fc1 pack",
          order.index("probs_a2a.bwd") < order.index("fc1.bwd")
          and order.index("probs_sort.bwd") < order.index("fc1.bwd"))
    check("sec2: split+bump — actscale precedes the probs path (grad arrives mid-chain)",
          order.index("actscale.bwd") < order.index("probs_sort.bwd"))

    order = _build_graph(split=True, bump=False)
    check("sec2: control (split, NO bump) — fc1 fires first (natural seq order; "
          "the split alone is not enough, the bump is load-bearing)",
          order.index("fc1.bwd") < order.index("probs_sort.bwd"))

    order = _build_graph(split=False, bump=False)
    check("sec2: control (joint sort, today) — the probs grad is held behind fc1",
          order.index("fc1.bwd") < order.index("joint_sort.bwd")
          and order.index("joint_sort.bwd") < order.index("probs_a2a.bwd"))


# ---------------------------------------------------------------------------
# sec 3 — mappings.py bwd_defer carrier protocol (2-proc gloo)
# ---------------------------------------------------------------------------

def _reference_reverse_a2a(group, grad, in_splits, out_splits):
    """The reverse A2A as one blocking call (the non-deferred reference)."""
    grad_input = grad.new_empty([sum(in_splits)] + list(grad.size()[1:]), dtype=grad.dtype)
    dist.all_to_all_single(
        grad_input, grad, output_split_sizes=in_splits, input_split_sizes=out_splits, group=group
    )
    return grad_input


def _sec3_worker(rank, ws, store_path):
    dist.init_process_group("gloo", store=dist.FileStore(store_path, ws), rank=rank, world_size=ws)
    g = dist.group.WORLD
    g2 = dist.new_group(dist.get_process_group_ranks(g))
    IN = {0: [3, 5], 1: [2, 4]}
    OUT = {0: [3, 2], 1: [5, 4]}
    gen = torch.Generator().manual_seed(rank)

    class _Mid(torch.autograd.Function):
        """Delays the LATE sibling's grad so the early one fires first."""

        @staticmethod
        def forward(ctx, x):
            return x * 1.0

        @staticmethod
        def backward(ctx, g):
            return g

    # --- case A: normal order (early fires first via an extra mid node on late) ---
    t = torch.randn(sum(IN[rank]), 4, generator=gen, requires_grad=True)
    p = torch.randn(sum(IN[rank]), 4, generator=gen, requires_grad=True)
    # Capture the early backward's RETURNED buffer (the unwaited grad). On
    # CPU/gloo the engine's AccumulateGrad copies it into the leaf's .grad
    # before the collective fills it — a leaf consumer is a CPU artifact; the
    # real graph's only consumer (permute1.bwd) is downstream of the late
    # sibling's wait, so on CUDA the read is stream-ordered and safe. What we
    # assert here: the returned buffer itself ends up correct after the
    # protocol (the late sibling's wait covers it).
    hooked = {}
    p.register_hook(lambda gg: hooked.setdefault("buf", gg))
    carrier = {}
    ot = mappings.all_to_all_deferred(
        g, t, OUT[rank], IN[rank], bwd_defer={"role": "late", "carrier": carrier}
    )
    op = mappings.all_to_all_deferred(
        g2, p, OUT[rank], IN[rank], bwd_defer={"role": "early", "carrier": carrier}
    )
    mappings.wait_deferred_a2a(ot)
    mappings.wait_deferred_a2a(op)
    # late's grad arrives via _Mid (one extra node) => early.bwd fires first
    loss = _Mid.apply(ot).sum() + op.sum()
    loss.backward()
    ref_t = _reference_reverse_a2a(g, torch.ones(sum(OUT[rank]), 4), IN[rank], OUT[rank])
    ref_p = _reference_reverse_a2a(g2, torch.ones(sum(OUT[rank]), 4), IN[rank], OUT[rank])
    check("sec3 case A: carrier drained (late popped early's handle)", "early_work" not in carrier)
    check("sec3 case A: late-done latched", carrier.get("late_done") is True)
    check("sec3 case A: tokens reverse grad == reference", torch.allclose(t.grad, ref_t))
    check("sec3 case A: early reverse's returned buffer == reference after the late wait",
          "buf" in hooked and torch.allclose(hooked["buf"], ref_p))

    # --- case B: pathological late-first order (bump late above early) ---
    t2 = torch.randn(sum(IN[rank]), 4, generator=gen, requires_grad=True)
    p2 = torch.randn(sum(IN[rank]), 4, generator=gen, requires_grad=True)
    carrier2 = {}
    ot2 = mappings.all_to_all_deferred(
        g, t2, OUT[rank], IN[rank], bwd_defer={"role": "late", "carrier": carrier2}
    )
    op2 = mappings.all_to_all_deferred(
        g2, p2, OUT[rank], IN[rank], bwd_defer={"role": "early", "carrier": carrier2}
    )
    mappings.wait_deferred_a2a(ot2)
    mappings.wait_deferred_a2a(op2)
    set_tensor_grad_fn_sequence_sr(ot2, torch.iinfo(torch.int).max)  # late fires FIRST
    import logging as _logging

    records = []
    handler = _logging.Handler()
    handler.emit = lambda r: records.append(r.getMessage())
    mappings.logger.addHandler(handler)
    (ot2.sum() + op2.sum()).backward()
    mappings.logger.removeHandler(handler)
    check("sec3 case B: late-first => early waited inline with a loud warning",
          any("early reverse ran AFTER the late sibling" in m for m in records))
    check("sec3 case B: values still correct (tokens)",
          torch.allclose(t2.grad, _reference_reverse_a2a(g, torch.ones(sum(OUT[rank]), 4), IN[rank], OUT[rank])))
    check("sec3 case B: values still correct (probs)",
          torch.allclose(p2.grad, _reference_reverse_a2a(g2, torch.ones(sum(OUT[rank]), 4), IN[rank], OUT[rank])))

    # --- case C: late role with an empty carrier => loud missing-handle warning ---
    t3 = torch.randn(sum(IN[rank]), 4, generator=gen, requires_grad=True)
    carrier3 = {}
    ot3 = mappings.all_to_all_deferred(
        g, t3, OUT[rank], IN[rank], bwd_defer={"role": "late", "carrier": carrier3}
    )
    mappings.wait_deferred_a2a(ot3)
    records.clear()
    handler2 = _logging.Handler()
    handler2.emit = lambda r: records.append(r.getMessage())
    mappings.logger.addHandler(handler2)
    ot3.sum().backward()
    mappings.logger.removeHandler(handler2)
    check("sec3 case C: missing early handle => loud warning",
          any("found no early work" in m for m in records))

    dist.destroy_process_group()


def test_3_bwd_defer_protocol():
    print("\n== sec3: bwd_defer early/late carrier protocol (2-proc gloo) ==")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_sec3_worker, args=(2, os.path.join(tmp, "s")), nprocs=2, join=True)


# ---------------------------------------------------------------------------
# sec 4 — gate + arm predicate
# ---------------------------------------------------------------------------

def test_4_gate_and_arm_predicate():
    print("\n== sec4: gate + arm predicate ==")
    td = token_dispatcher
    old = os.environ.pop("BT_MOE_PROBS_BWD_REORDER", None)
    try:
        check("sec4: gate off by default", td._probs_bwd_reorder_enabled() is False)
        os.environ["BT_MOE_PROBS_BWD_REORDER"] = "1"
        check("sec4: gate on with env=1", td._probs_bwd_reorder_enabled() is True)
    finally:
        if old is None:
            os.environ.pop("BT_MOE_PROBS_BWD_REORDER", None)
        else:
            os.environ["BT_MOE_PROBS_BWD_REORDER"] = old

    a = td._probs_bwd_reorder_armable
    ok = dict(probs_a2a_comm_present=True, w2_pipeline=False, drop_and_pad=False,
              tp_size=1, num_local_experts=16, moe_permute_fusion=True)
    check("sec4: golden config armable", a(**ok) is None)
    check("sec4: refuses without the W1 comm",
          a(**{**ok, "probs_a2a_comm_present": False}) is not None)
    check("sec4: refuses under W2", a(**{**ok, "w2_pipeline": True}) is not None)
    check("sec4: refuses drop_and_pad", a(**{**ok, "drop_and_pad": True}) is not None)
    check("sec4: refuses TP>1", a(**{**ok, "tp_size": 2}) is not None)
    check("sec4: refuses single-expert", a(**{**ok, "num_local_experts": 1}) is not None)
    check("sec4: refuses unfused sort", a(**{**ok, "moe_permute_fusion": False}) is not None)


# ---------------------------------------------------------------------------
# sec 5 — source guards (AST): the wiring forms that must not drift
# ---------------------------------------------------------------------------

def test_5_source_guards():
    print("\n== sec5: source guards (AST) ==")
    import ast
    import inspect

    # (a) the sort Function never takes a host sync (.tolist on the split path)
    fn_src = inspect.getsource(moe_utils._SortProbsEarlyBwd)
    check("sec5: _SortProbsEarlyBwd has no .tolist() (no new host sync)",
          ".tolist()" not in fn_src)
    check("sec5: _SortProbsEarlyBwd drives the fused fwd + bwd ops",
          "fused_sort_chunks_by_index_with_row_id_map" in fn_src
          and "fused_sort_chunks_by_index_bwd" in fn_src)

    # (b) dispatch_postprocess: the split is gated; the else branch keeps the
    # fused with-probs call (gate-off byte-identical path). Several dispatcher
    # classes define dispatch_postprocess — target the one carrying the
    # option-6 wiring (the alltoall dispatcher).
    src_path = os.path.join(_find_mcore(), "megatron", "core", "transformer", "moe", "token_dispatcher.py")
    src = open(src_path).read()
    with open(src_path) as f:
        tree = ast.parse(f.read())
    dp = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "dispatch_postprocess"
        and "_probs_bwd_reorder_active" in ast.get_source_segment(src, n)
    ]
    check("sec5: exactly one dispatch_postprocess carries the option-6 wiring", len(dp) == 1)
    if dp:
        seg = ast.get_source_segment(src, dp[0])
        check("sec5: dispatch_postprocess splits the sort under the gate",
              "sort_probs_chunks_early_bwd" in seg)
        check("sec5: the fused with-probs call survives in the else branch",
              seg.count("sort_chunks_by_idxs") >= 2 and "probs=global_probs" in seg)
        check("sec5: the probs-sort node is seq-bumped",
              "set_tensor_grad_fn_sequence_sr(global_probs" in seg)

    # (c) token_dispatch: bwd_defer only under the option-6 gate (target the
    # alltoall dispatcher's method — the one with the W1 branch).
    tdc = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "token_dispatch"
        and "_probs_a2a_comm" in ast.get_source_segment(src, n)
    ]
    check("sec5: exactly one token_dispatch carries the W1 branch", len(tdc) == 1)
    if tdc:
        seg = ast.get_source_segment(src, tdc[0])
        check("sec5: token_dispatch wires the early/late carrier under the gate",
              "bwd_defer" in seg and "_probs_bwd_reorder_active" in seg)

    # (d) mappings: the default (no-carrier) backward path still waits inline.
    bwd_src = inspect.getsource(mappings._AllToAllDeferredWait.backward)
    check("sec5: mappings default backward unchanged (bd is None => work.wait())",
          "if bd is None:" in bwd_src and "work.wait()" in bwd_src)
    check("sec5: mappings carrier protocol present (early stash / late pop)",
          'bd["carrier"]["early_work"] = work' in bwd_src and "late_done" in bwd_src)


def main():
    test_1_sort_function_semantics()
    test_2_engine_order_mechanism()
    test_4_gate_and_arm_predicate()
    test_5_source_guards()
    test_3_bwd_defer_protocol()  # last: spawns dist workers
    if FAILURES:
        raise SystemExit(f"{len(FAILURES)} FAILURES: {FAILURES}")
    print("OK — all option-6 CPU checks passed")


if __name__ == "__main__":
    main()
