#!/usr/bin/env python3
"""STOPPED-VARIANT PRESERVE — W1-v2 paired dispatch node (BT_MOE_PROBS_A2A_COMM_V2).

Disposition (fibonacci, 2026-08-09): ARCHIVED-UNSHIPPED. hilbert's
launch-vs-kernel-start cut resolved the pre-registered ship-gate NEGATIVE:
the probs reverse's LAUNCH is already ~24 ms early on an idle stream; the
kernel waits ~41 ms for the PROBS GRAD itself (produced post-fc2-dgrad, deep
in the layer's backward chain). The late grad is the probs' — exactly the
readiness-drift case under which the paired node would chain the tokens
reverse to the late grad (correct but potentially slower than v1). The real
lever (backward-chain reorder: emit d(probs) right after fc2-dgrad) is worth
only ~1–1.5 s/step and is subsumed by W3's lookahead — parked as option 6 in
../W2V2_DECISION_stub.md (W3-absence fallback only).

This file preserves the stopped variant's tests (they were green on the
built variant) so they can be re-run if the stub un-parks: they skip cleanly
when `all_to_all_deferred_pair` is absent from the vendored mappings (the
ship tree). The probes that produced the seq-order evidence are
w1v2_probe.py / w1v2_probe2.py in this same directory.

Usage: python test_w1v2_paired_dispatch_STOPPED.py
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

if not hasattr(mappings, "all_to_all_deferred_pair"):
    print(
        "SKIP — all_to_all_deferred_pair not in the vendored tree (W1-v2 is "
        "archived-unshipped; this suite is preserved for a stub un-park)"
    )
    raise SystemExit(0)

FAILURES = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name, flush=True)
    if not cond:
        FAILURES.append(name)


IN_SPLITS = {0: [3, 5], 1: [2, 4]}
OUT_SPLITS = {0: [3, 2], 1: [5, 4]}
H = 4


def _reference_a2a(group, input_, out_splits, in_splits):
    out = input_.new_empty([sum(out_splits)] + list(input_.size()[1:]))
    dist.all_to_all_single(
        out, input_.contiguous(), output_split_sizes=out_splits,
        input_split_sizes=in_splits, group=group,
    )
    return out


def _worker(rank, world_size, store_path):
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
    ep_group = dist.group.WORLD
    probs_group = dist.new_group(dist.get_process_group_ranks(ep_group))
    g = torch.Generator().manual_seed(600 + rank)
    tokens = torch.randn(sum(IN_SPLITS[rank]), H, generator=g)
    probs = torch.randn(sum(IN_SPLITS[rank]), generator=g)

    # forward: pair == reference
    out_t, out_p = mappings.all_to_all_deferred_pair(
        ep_group, probs_group, tokens, probs, OUT_SPLITS[rank], IN_SPLITS[rank]
    )
    mappings.wait_deferred_a2a(out_t)
    mappings.wait_deferred_a2a(out_p)
    check("pair fwd: tokens == reference", torch.equal(out_t, _reference_a2a(ep_group, tokens, OUT_SPLITS[rank], IN_SPLITS[rank])))
    check("pair fwd: probs == reference", torch.equal(out_p, _reference_a2a(ep_group, probs, OUT_SPLITS[rank], IN_SPLITS[rank])))

    # backward: pair == reverse-A2A references
    gg = torch.Generator().manual_seed(700 + rank)
    grad_t = torch.randn(sum(OUT_SPLITS[rank]), H, generator=gg)
    grad_p = torch.randn(sum(OUT_SPLITS[rank]), generator=gg)
    t2 = tokens.clone().requires_grad_(True)
    p2 = probs.clone().requires_grad_(True)
    out_t2, out_p2 = mappings.all_to_all_deferred_pair(
        ep_group, probs_group, t2, p2, OUT_SPLITS[rank], IN_SPLITS[rank]
    )
    mappings.wait_deferred_a2a(out_t2)
    mappings.wait_deferred_a2a(out_p2)
    torch.autograd.backward([out_t2, out_p2], [grad_t, grad_p])
    check("pair bwd: tokens grad == reference", torch.equal(t2.grad, _reference_a2a(ep_group, grad_t, IN_SPLITS[rank], OUT_SPLITS[rank])))
    check("pair bwd: probs grad == reference", torch.equal(p2.grad, _reference_a2a(ep_group, grad_p, IN_SPLITS[rank], OUT_SPLITS[rank])))

    dist.barrier()
    dist.destroy_process_group()
    if FAILURES:
        raise SystemExit(f"rank {rank}: FAILURES {FAILURES}")
    print(f"rank {rank}: stopped-variant checks passed", flush=True)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(2, os.path.join(tmp, "s")), nprocs=2, join=True)
    print("OK — stopped-variant W1-v2 checks passed (only reachable if un-parked)")


if __name__ == "__main__":
    main()
