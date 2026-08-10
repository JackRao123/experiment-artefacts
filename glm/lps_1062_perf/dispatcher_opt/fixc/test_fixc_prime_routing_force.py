#!/usr/bin/env python3
"""Mac-CPU test for FIX C-prime — BT_MOE_ROUTING_REPLAY_FORCE (replay routing
forced to the first-pass decision).

Drives the REAL TopKRouter (vendored router.py) through the REAL checkpoint
machinery (tensor_parallel.checkpoint -> CheckpointFunction) plus the REAL FIX
C pass-marker frames (recompute._wrap_checkpoint_chunk_pass), on Mac CPU:

  1. gate OFF: code inert — no stash, no force, counters zero; with a
     simulated replay-logit perturbation the free replay's routing_map
     DIVERGES from the first pass (the control that proves the test can see
     the ARM-1 failure class).
  2. gate ON (with BT_MOE_DISPATCH_REPLAY_CACHE=1 so frames exist): the
     replay's routing_map is bitwise the first pass's DESPITE the
     perturbation (the forced selection); probs stay grad-connected (input
     grad + router weight grad both flow); counters {stashes:1, forces:1,
     misses:0}; the in-flight byte counter returns to 0 after the fetch
     (eviction by one-shot pop).
  3. padding rows: a saved map with an all-False row never produces an
     all-(-inf) masked row (no NaN in probs; the row's map stays all-False).
  4. fallbacks: replay with an empty stash (miss) and a shape-mismatched saved
     map both route freely (status quo) with the counters incremented.
  5. verify mode (BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY=1): a consistent replay
     passes; a corrupted saved map raises RuntimeError (per-token set
     equality asserted).
  6. loss-terms fallback: moe_z_loss_coeff set -> no forcing, one WARNING,
     latch persists.
  7. source guards: TopKRouter.forward calls _routing_force_replay_saved_map;
     the gate helper reads BT_MOE_ROUTING_REPLAY_FORCE.

The replay-logit perturbation simulates the ARM-1 mechanism (grad-mode kernel
divergence upstream of the router): the monkeypatched gating adds eps only
when torch.is_grad_enabled() (False in the checkpoint first pass, True in the
replay) — truthful at this plain-function level.

Runs on Mac CPU. BT_TEST_MCORE_PATH overrides the tree walk-up.

Usage: python test_fixc_prime_routing_force.py
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

import megatron.core.recompute as recompute  # noqa: E402
import megatron.core.tensor_parallel.random as tp_random  # noqa: E402
from megatron.core import tensor_parallel  # noqa: E402
from megatron.core.packed_seq_params import PackedSeqParams  # noqa: E402
from megatron.core.transformer.moe.router import TopKRouter  # noqa: E402
import megatron.core.transformer.moe.router as router_mod  # noqa: E402

FAILURES = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name)
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

HIDDEN = 16
EXPERTS = 8
TOPK = 2
LAYER = 3


class _G:
    def size(self):
        return 1

    def rank(self):
        return 0


def _make_config(**overrides):
    cfg = types.SimpleNamespace(
        num_moe_experts=EXPERTS,
        moe_router_topk=TOPK,
        moe_router_load_balancing_type="none",
        moe_router_score_function="softmax",
        moe_router_enable_expert_bias=False,
        moe_aux_loss_coeff=0,
        moe_enable_routing_replay=False,
        moe_z_loss_coeff=None,
        hidden_size=HIDDEN,
        moe_router_dtype=None,
        moe_router_pre_softmax=False,
        moe_router_num_groups=None,
        moe_router_group_topk=None,
        moe_router_topk_scaling_factor=None,
        moe_router_fusion=False,
        moe_router_force_load_balancing=False,
        moe_router_force_biased=None,
        moe_expert_capacity_factor=None,
        moe_pad_expert_input_to_capacity=False,
        moe_token_drop_policy="probs",
        moe_router_num_layers_per_virtual_pipeline_stage=None,
        add_bias_linear=False,
        moe_router_padding_for_quantization=False,
        moe_router_padding_free=False,
        calculate_per_token_loss=False,
        moe_router_num_layers=None,
        num_layers=1,
        mtp_num_layers=None,
        perform_initialization=False,
        params_dtype=torch.float32,
        sequence_parallel=False,
        moe_input_jitter_eps=None,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_router(seed=0, layer=LAYER, **overrides):
    torch.cuda.current_device = lambda: "cpu"  # CPU-only: gating's weight move no-ops
    cfg = _make_config(**overrides)
    pg = types.SimpleNamespace(tp=_G(), cp=_G(), tp_cp=_G(), tp_dp_cp=_G())
    router = TopKRouter(cfg, pg)
    router.set_layer_number(layer)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        router.weight.data = torch.randn(EXPERTS, HIDDEN, generator=g)
    return router


def _install_rng_stubs():
    tp_random._get_all_rng_states = lambda: (torch.get_rng_state(), None, None)

    def _set_all(cpu_state, cuda_state, tracker_state):
        torch.set_rng_state(cpu_state)

    tp_random._set_all_rng_states = _set_all


def _reset_force_state():
    router_mod._ROUTING_FORCE_GATE_LOGGED[0] = False
    router_mod._ROUTING_FORCE_ARMED_LOGGED[0] = False
    router_mod._ROUTING_FORCE_HIT_LOGGED[0] = False
    router_mod._ROUTING_FORCE_LOSS_DISABLED[0] = False
    for k in router_mod._ROUTING_FORCE_STATS:
        router_mod._ROUTING_FORCE_STATS[k] = 0
    router_mod._ROUTING_FORCE_WINDOW[0] = 0
    router_mod._ROUTING_FORCE_WINDOW[1] = 0
    router_mod._ROUTING_FORCE_BYTES[0] = 0
    router_mod._ROUTING_FORCE_BYTES_PEAK[0] = 0


def _perturbable_gating(router, eps):
    """Simulate grad-mode kernel divergence: add eps to the logits only when
    grad is enabled (the replay), never in the no-grad first pass."""
    orig = router.gating

    def gating(x):
        out = orig(x)
        if torch.is_grad_enabled():
            return out + eps
        return out

    router.gating = gating


def _run_router_through_checkpoint(router, psp, x):
    """The production wiring: marker-wrap the chunk, then checkpoint it.

    Returns (first_pass_map, replay_map, probs_out): the checkpoint's RETURNED
    map is the first pass's; the replay's map is captured by a routing() spy
    (the replay's outputs are internal to the backward otherwise).
    """
    seen = {}  # pass-kind -> routing_map
    orig_routing = router.routing

    def routing_spy(logits, padding_mask=None):
        probs, routing_map = orig_routing(logits, padding_mask=padding_mask)
        seen["replay" if torch.is_grad_enabled() else "first"] = routing_map.detach().clone()
        return probs, routing_map

    router.routing = routing_spy

    def chunk(h):
        probs, routing_map = router(h)
        return probs, routing_map.to(torch.float32)

    wrapped = recompute._wrap_checkpoint_chunk_pass(chunk, psp)
    out = tensor_parallel.checkpoint(wrapped, False, x)
    probs_out, map_out = out
    probs_out.sum().backward()
    router.routing = orig_routing
    return map_out.to(torch.bool), seen.get("replay"), probs_out


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def case_gate_off_control():
    print("\n== case 1: gate OFF — inert; free replay diverges under perturbation (control) ==")
    os.environ.pop("BT_MOE_ROUTING_REPLAY_FORCE", None)
    os.environ.pop("BT_MOE_DISPATCH_REPLAY_CACHE", None)
    _reset_force_state()
    router = _make_router(seed=1)
    # Large per-logit perturbation in the replay: guaranteed boundary flips.
    _perturbable_gating(router, torch.randn(1, 1, EXPERTS) * 2.0)
    psp = PackedSeqParams()
    x = torch.randn(32, 1, HIDDEN, generator=torch.Generator().manual_seed(42), requires_grad=True)

    first_map, replay_map, _ = _run_router_through_checkpoint(
        router, psp, x.detach().clone().requires_grad_(True)
    )
    check("gate off: no stash on the carrier", not hasattr(psp, router_mod._ROUTING_FORCE_STASH_ATTR))
    check("gate off: counters zero", all(v == 0 for v in router_mod._ROUTING_FORCE_STATS.values()))
    check("control: the replay produced a map", replay_map is not None)
    check(
        "gate off (control): the perturbed replay's map diverges from the first pass's",
        replay_map is not None and not torch.equal(replay_map, first_map),
    )
    return first_map


def case_forced_replay(reference_map):
    print("\n== case 2: gate ON — replay forced to the first-pass map despite perturbation ==")
    os.environ["BT_MOE_ROUTING_REPLAY_FORCE"] = "1"
    os.environ["BT_MOE_DISPATCH_REPLAY_CACHE"] = "1"  # frames exist only with the base gate
    _reset_force_state()
    router = _make_router(seed=1)
    _perturbable_gating(router, torch.randn(1, 1, EXPERTS) * 2.0)
    psp = PackedSeqParams()
    x = torch.randn(32, 1, HIDDEN, generator=torch.Generator().manual_seed(42), requires_grad=True)

    x_in = x.detach().clone().requires_grad_(True)
    first_map, replay_map, _ = _run_router_through_checkpoint(router, psp, x_in)
    check(
        "gate on: replay's routing_map bitwise == first pass's (forced)",
        replay_map is not None and torch.equal(replay_map, reference_map),
    )
    check("gate on: input grad flows (probs grad-connected to hidden)",
          x_in.grad is not None and x_in.grad.abs().sum() > 0)
    check("gate on: router weight grad flows (activation-grad path preserved)",
          router.weight.grad is not None and router.weight.grad.abs().sum() > 0)
    check(
        "gate on: counters {stashes:1, forces:1, misses:0}",
        router_mod._ROUTING_FORCE_STATS["stashes"] == 1
        and router_mod._ROUTING_FORCE_STATS["forces"] == 1
        and router_mod._ROUTING_FORCE_STATS["misses"] == 0,
    )
    check("gate on: in-flight bytes back to 0 after the one-shot pop",
          router_mod._ROUTING_FORCE_BYTES[0] == 0)


def case_padding_rows():
    print("\n== case 3: padding rows — no all-(-inf) NaN row ==")
    os.environ["BT_MOE_ROUTING_REPLAY_FORCE"] = "1"
    os.environ["BT_MOE_DISPATCH_REPLAY_CACHE"] = "1"
    _reset_force_state()
    router = _make_router(seed=2)
    psp = PackedSeqParams()
    x = torch.randn(8, 1, HIDDEN, requires_grad=True)

    # First pass via a real replay-frame push: stash a map with an all-False row.
    with torch.no_grad():
        _, saved_map = router(x.detach())
    saved_map = saved_map.clone()
    saved_map[4] = False  # a padding row
    stash = {id(router): saved_map}
    setattr(psp, router_mod._ROUTING_FORCE_STASH_ATTR, stash)

    # Replay frame: push manually (the real frame machinery, no checkpoint).
    frame = recompute.CheckpointPassFrame(True, psp)
    recompute._replay_pass_stack().append(frame)
    try:
        probs, routing_map = router(x)
    finally:
        recompute._replay_pass_stack().pop()
    check("padding: returned map IS the saved map", routing_map is saved_map)
    check("padding: padded row's map stays all-False", not routing_map[4].any().item())
    check("padding: no NaN in probs (the all-(-inf) row guard works)",
          not torch.isnan(probs).any().item())
    check("padding: force counted", router_mod._ROUTING_FORCE_STATS["forces"] == 1)


def case_fallbacks():
    print("\n== case 4: fallbacks — empty stash (miss) + shape mismatch ==")
    os.environ["BT_MOE_ROUTING_REPLAY_FORCE"] = "1"
    os.environ["BT_MOE_DISPATCH_REPLAY_CACHE"] = "1"
    _reset_force_state()
    router = _make_router(seed=3)
    psp = PackedSeqParams()
    x = torch.randn(8, 1, HIDDEN, requires_grad=True)

    # miss: empty stash
    frame = recompute.CheckpointPassFrame(True, psp)
    recompute._replay_pass_stack().append(frame)
    try:
        router(x)
    finally:
        recompute._replay_pass_stack().pop()
    check("miss: routed freely, counter incremented", router_mod._ROUTING_FORCE_STATS["misses"] == 1)

    # shape mismatch: saved map with a wrong shape
    bad = torch.zeros(4, EXPERTS, dtype=torch.bool)  # logits are [8, 8]
    setattr(psp, router_mod._ROUTING_FORCE_STASH_ATTR, {id(router): bad})
    recompute._replay_pass_stack().append(recompute.CheckpointPassFrame(True, psp))
    try:
        router(x)
    finally:
        recompute._replay_pass_stack().pop()
    check("shape mismatch: routed freely, counter incremented",
          router_mod._ROUTING_FORCE_STATS["shape_mismatches"] == 1)


def case_verify_mode():
    print("\n== case 5: verify mode — set-equality assert ==")
    os.environ["BT_MOE_ROUTING_REPLAY_FORCE"] = "1"
    os.environ["BT_MOE_DISPATCH_REPLAY_CACHE"] = "1"
    os.environ["BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY"] = "1"
    _reset_force_state()
    router = _make_router(seed=4)
    psp = PackedSeqParams()
    x = torch.randn(8, 1, HIDDEN, requires_grad=True)

    with torch.no_grad():
        _, saved_map = router(x.detach())

    # consistent replay: verify passes
    setattr(psp, router_mod._ROUTING_FORCE_STASH_ATTR, {id(router): saved_map})
    recompute._replay_pass_stack().append(recompute.CheckpointPassFrame(True, psp))
    try:
        router(x)
    finally:
        recompute._replay_pass_stack().pop()
    check("verify: consistent replay passes", router_mod._ROUTING_FORCE_STATS["verify_asserts"] == 1)

    # corrupted saved map: flip one row's set -> the masked topk reproduces the
    # SAVED set (the mask forces it), so corrupting the saved map changes the
    # RETURNED map, not the assert — the assert compares recomputed(masked) vs
    # saved. To exercise the raise, corrupt AFTER masking is impossible from
    # outside; instead verify the assert fires when the recomputed map is
    # perturbed: patch routing to return a different map.
    orig_routing = router.routing

    def bad_routing(logits, padding_mask=None):
        probs, rm = orig_routing(logits, padding_mask=padding_mask)
        rm = rm.clone()
        rm[0] = ~rm[0]  # corrupt the recomputed map
        return probs, rm

    router.routing = bad_routing
    setattr(psp, router_mod._ROUTING_FORCE_STASH_ATTR, {id(router): saved_map})
    raised = False
    recompute._replay_pass_stack().append(recompute.CheckpointPassFrame(True, psp))
    try:
        router(x)
    except RuntimeError as e:
        raised = "BT_MOE_ROUTING_REPLAY_FORCE verify" in str(e)
    finally:
        recompute._replay_pass_stack().pop()
        router.routing = orig_routing
    check("verify: recomputed-vs-saved mismatch raises RuntimeError", raised)
    os.environ.pop("BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY", None)


def case_loss_terms_fallback():
    print("\n== case 6: z-loss/aux-loss fallback (one WARNING, latched) ==")
    os.environ["BT_MOE_ROUTING_REPLAY_FORCE"] = "1"
    os.environ["BT_MOE_DISPATCH_REPLAY_CACHE"] = "1"
    _reset_force_state()
    router = _make_router(seed=5, layer=1, moe_z_loss_coeff=0.01)
    psp = PackedSeqParams()
    x = torch.randn(8, 1, HIDDEN, requires_grad=True)

    frame = recompute.CheckpointPassFrame(True, psp)
    recompute._replay_pass_stack().append(frame)
    try:
        router(x)
    finally:
        recompute._replay_pass_stack().pop()
    check("z-loss on: no force, no miss (disabled before the fetch)", 
          router_mod._ROUTING_FORCE_STATS["forces"] == 0 and router_mod._ROUTING_FORCE_STATS["misses"] == 0)
    check("z-loss on: latch set", router_mod._ROUTING_FORCE_LOSS_DISABLED[0] is True)


def case_mtp_layer_number_collision():
    print("\n== case 7: MTP layer-number collision — stash keyed by id(router) ==")
    os.environ["BT_MOE_ROUTING_REPLAY_FORCE"] = "1"
    os.environ["BT_MOE_DISPATCH_REPLAY_CACHE"] = "1"
    _reset_force_state()
    # Two router INSTANCES sharing one layer_number (the MTP router reuses a
    # main-stack layer number): layer_number keying would cross-contaminate
    # (B's stash overwrites A's; A's replay pops B's map). id(router) keying
    # keeps them distinct — the router object is identical across a
    # microbatch's first pass and replay.
    routerA = _make_router(seed=11, layer=7)
    routerB = _make_router(seed=22, layer=7)
    psp = PackedSeqParams()
    x = torch.randn(32, 1, HIDDEN, generator=torch.Generator().manual_seed(7))

    seen = {"A": {}, "B": {}}
    for tag, r in (("A", routerA), ("B", routerB)):
        orig = r.routing

        def make_spy(orig, bag):
            def spy(logits, padding_mask=None):
                probs, rm = orig(logits, padding_mask=padding_mask)
                bag["replay" if torch.is_grad_enabled() else "first"] = rm.detach().clone()
                return probs, rm
            return spy

        r.routing = make_spy(orig, seen[tag])

    def chunk(h):
        pA, mA = routerA(h)
        pB, mB = routerB(h)
        return (pA.sum() + pB.sum()), mA.to(torch.float32) + mB.to(torch.float32)

    wrapped = recompute._wrap_checkpoint_chunk_pass(chunk, psp)
    x_in = x.detach().clone().requires_grad_(True)
    out = tensor_parallel.checkpoint(wrapped, False, x_in)
    out[0].backward()
    check("MTP-collision: the two routers' first-pass maps differ (real signal)",
          not torch.equal(seen["A"]["first"], seen["B"]["first"]))
    check("MTP-collision: router A's replay == router A's first pass",
          torch.equal(seen["A"]["replay"], seen["A"]["first"]))
    check("MTP-collision: router B's replay == router B's first pass",
          torch.equal(seen["B"]["replay"], seen["B"]["first"]))
    check("MTP-collision: counters {stashes:2, forces:2, misses:0}",
          router_mod._ROUTING_FORCE_STATS["stashes"] == 2
          and router_mod._ROUTING_FORCE_STATS["forces"] == 2
          and router_mod._ROUTING_FORCE_STATS["misses"] == 0)


def case_source_guards():
    print("\n== case 8: source guards ==")
    import inspect

    fwd_src = inspect.getsource(TopKRouter.forward)
    check("TopKRouter.forward calls _routing_force_replay_saved_map",
          "_routing_force_replay_saved_map" in fwd_src)
    check("TopKRouter.forward stashes in the non-forced path",
          "_routing_force_stash" in fwd_src)
    gate_src = inspect.getsource(router_mod._routing_force_enabled)
    check("gate reads BT_MOE_ROUTING_REPLAY_FORCE", "BT_MOE_ROUTING_REPLAY_FORCE" in gate_src)
    fetch_src = inspect.getsource(router_mod._routing_force_replay_saved_map)
    stash_src = inspect.getsource(router_mod._routing_force_stash)
    check("stash key is id(router), never layer_number (MTP collision guard)",
          "id(router)" in fetch_src and "id(router)" in stash_src
          and "router.layer_number" not in fetch_src and "router.layer_number" not in stash_src)


def main():
    _install_rng_stubs()
    reference_map = case_gate_off_control()
    case_forced_replay(reference_map)
    case_padding_rows()
    case_fallbacks()
    case_verify_mode()
    case_loss_terms_fallback()
    case_mtp_layer_number_collision()
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
