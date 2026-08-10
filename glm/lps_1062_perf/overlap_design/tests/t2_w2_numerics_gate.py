#!/usr/bin/env python3
"""T2 on-box numerics gate for W2 — BT_MOE_A2A_PIPELINE (the V1 gate).

Proves the chunked MoE A2A pipeline (K=2 expert groups, list-form A2A over
per-peer views) is BITWISE identical to the monolithic alltoall path on the
real stack: one MoELayer (TEGroupedMLP, moe_grouped_gemm, moe_permute_fusion,
FP8 e4m3 blockwise — the GLM-5.2 recipe), EP = world size, full-recompute
checkpoint wrapper (so the chunked path runs in the no-grad first pass AND
the grad-enabled replay), expert weights frozen (the golden LoRA config).

For each routing case, the SAME weights/inputs/routing are run with W2 off vs
W2 on (in-process A/B), and the layer output and input grad must be
torch.equal:

  1. balanced — random top-8 routing;
  2. imbalance — ~90% of selections to expert 0 (a2a sizes up to 3.5x skew);
  3. zero (peer, group) — every rank sends nothing to dest 3's group-1
     experts (global experts 56..63): exercises the list-A2A zero-count path
     (R2) on both the send side (all ranks) and the receive side (rank 3);
  4. zero whole expert — expert 42 gets zero rows GLOBALLY: exercises the
     zero-row expert GEMM and the Fp8Padding boundary logic;
  5. FIX C composition — cases 1 and 3 re-run with
     BT_MOE_DISPATCH_REPLAY_CACHE=1 under the checkpoint-pass marker, so the
     replay restores the cached W2 count matrices (the BUG-3 path).

Also covers the W1+W2 combination (probs on the second communicator) in every
case: run A = W1 off/W2 off, run B = W1 on/W2 on, and the W2-on runs are
additionally compared against run A.

TE-version notes (second-review item 4):
* Fp8Padding with a zero count pads 0 -> 0 (0 is a multiple of any
  align_size), so a zero-row expert contributes an empty problem with no
  padding rows — verified by the zero_expert case here. If a TE version
  changes that boundary behavior, case 4 fails first.
* An M=0 grouped-GEMM problem must not contribute to the FP8 amax: the
  blockwise recipe computes per-block scales statelessly per call, so empty
  problems are inert by construction; any TE regression there shows up here
  as a bitwise mismatch on the zero_expert case, and MULTI-STEP amax/history
  drift is covered by the 20-step loss canary (the named covering gate —
  see W2_PATCH_NOTES.md), not by this single-step script.
* The use_fixc branches assert the replay-cache stats (hits>0, misses==0)
  after the W2-on run: a silent FIX-C fallback would still pass torch.equal
  — the exact v1 inert-gate trap (second-review item 6).

Usage (16 ranks = golden EP16; 8 also works):
  torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d \
      --rdzv_endpoint=$ADDR t2_w2_numerics_gate.py
Env: BT_TEST_SEQ_LEN (default 8192), BT_TEST_HIDDEN (default 2048 — smaller
than GLM's 6144 to keep the gate fast; set 6144 for the full-fidelity run).
"""

import os
import sys

import torch
import torch.distributed as dist

# vendored mcore: BT_TEST_MCORE_PATH or the standard checkout layout
_MC = os.environ.get(
    "BT_TEST_MCORE_PATH",
    os.path.expanduser(
        "~/Documents/trainers/server/vendor/megatron-bridge/3rdparty/Megatron-LM"
    ),
)
if os.path.isdir(_MC):
    sys.path.insert(0, _MC)

import megatron.core.parallel_state as mpu  # noqa: E402
from megatron.core.enums import Fp8Recipe  # noqa: E402
from megatron.core.extensions.transformer_engine import (  # noqa: E402
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)
from megatron.core.fp8_utils import get_fp8_context  # noqa: E402
from megatron.core.process_groups_config import ProcessGroupCollection  # noqa: E402
from megatron.core.tensor_parallel import (  # noqa: E402
    checkpoint as tp_checkpoint,
    get_cuda_rng_tracker,
)
from megatron.core.transformer.moe.experts import (  # noqa: E402
    GroupedMLPSubmodules,
    TEGroupedMLP,
)
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules  # noqa: E402
from megatron.core.transformer.moe.router import TopKRouter  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402

try:
    from megatron.core.extensions.transformer_engine import te_checkpoint

    HAVE_TE_CKPT = True
except Exception:
    te_checkpoint = None
    HAVE_TE_CKPT = False

FAILURES = []


def check(name, cond):
    if dist.get_rank() == 0:
        print(("PASS" if cond else "FAIL"), "-", name, flush=True)
    if not cond:
        FAILURES.append(name)


NUM_EXPERTS = 256
TOPK = 8
K = 2


def build_config(hidden):
    return TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=8,
        moe_ffn_hidden_size=2048,
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=TOPK,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_permute_fusion=True,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_shared_expert_intermediate_size=None,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        bf16=True,
        params_dtype=torch.bfloat16,
        fp8="e4m3",
        fp8_recipe="blockwise",
        add_bias_linear=False,
        use_cpu_initialization=True,
        # the gate's validation surface
        moe_pad_expert_input_to_capacity=False,
        cuda_graph_impl="none",
    )


def build_layer(config, pg):
    def experts_builder(num_local_experts, cfg, /, *, pg_collection, name=None):
        return TEGroupedMLP(
            num_local_experts,
            cfg,
            GroupedMLPSubmodules(
                linear_fc1=TEColumnParallelGroupedLinear,
                linear_fc2=TERowParallelGroupedLinear,
            ),
            pg_collection=pg_collection,
            name=name,
        )

    layer = MoELayer(
        config,
        submodules=MoESubmodules(
            experts=experts_builder, shared_experts=None, router=TopKRouter
        ),
        layer_number=1,
        pg_collection=pg,
    )
    layer = layer.cuda().bfloat16()
    # golden config: experts frozen (attention-only LoRA) — required for W2 v1
    for p in layer.experts.parameters():
        p.requires_grad_(False)
    return layer


def craft_routing(case, T, device, seed):
    """(probs [T,E] sparse f32, routing_map [T,E] bool) — identical across ranks."""
    g = torch.Generator().manual_seed(seed)
    routing = torch.zeros(T, NUM_EXPERTS, dtype=torch.bool)
    for t in range(T):
        if case == "imbalance":
            idx = [0] * (TOPK - 1) + [int(torch.randint(0, NUM_EXPERTS, (1,), generator=g))]
        else:
            idx = torch.randperm(NUM_EXPERTS, generator=g)[:TOPK].tolist()
        routing[t, idx] = True
    if case == "zero_expert":
        routing[:, 42] = False
    if case == "zero_peer_group":
        # dest rank 3, group 1 (L = num_local_experts/K local experts per group)
        le = NUM_EXPERTS // dist.get_world_size()
        l = le // K
        routing[:, 3 * le + l : 3 * le + 2 * l] = False
    probs = torch.rand(T, NUM_EXPERTS, generator=g) * routing
    return probs.cuda(device), routing.cuda(device)


def run_once(layer, config, hidden_states, probs, routing_map, use_checkpoint, use_fixc_marker):
    """One fwd+bwd of the layer with a crafted routing; returns (out, in_grad)."""
    layer.router.forward = lambda *a, **k: (probs, routing_map)

    def fwd(h):
        return layer(h)[0]

    if use_fixc_marker:
        from megatron.core.recompute import _wrap_checkpoint_chunk_pass

        fwd = _wrap_checkpoint_chunk_pass(fwd, None)

    grad_out = torch.randn_like(hidden_states)
    hidden_states = hidden_states.detach().requires_grad_(True)
    with get_fp8_context(config, 0):
        if use_checkpoint:
            if config.fp8 and HAVE_TE_CKPT:
                out = te_checkpoint(
                    fwd, False, get_cuda_rng_tracker, layer.tp_group, hidden_states
                )
            else:
                out = tp_checkpoint(fwd, False, hidden_states)
        else:
            out = fwd(hidden_states)
    if isinstance(out, tuple):
        out = out[0]
    out.backward(grad_out)
    return out.detach().clone(), hidden_states.grad.detach().clone()


def set_w2(layer, enabled):
    d = layer.token_dispatcher
    if enabled:
        d.w2_try_enable_pipeline(layer.experts)
        layer._w2_frozen_checked = False  # re-run the first-forward check
        assert d._w2_config is not None, "W2 failed to arm (see WARNING lines)"
    else:
        d._w2_config = None


def set_w1(layer, enabled):
    d = layer.token_dispatcher
    if enabled:
        if d._probs_a2a_comm is None:
            d._probs_a2a_comm = dist.new_group(dist.get_process_group_ranks(d.ep_group))
    else:
        d._probs_a2a_comm = None


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    assert NUM_EXPERTS % world == 0
    os.environ.setdefault("BT_MOE_A2A_PIPELINE", "2")

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=world,
        expert_tensor_parallel_size=1,
    )
    pg = ProcessGroupCollection.use_mpu_process_groups()

    hidden = int(os.environ.get("BT_TEST_HIDDEN", "2048"))
    seq = int(os.environ.get("BT_TEST_SEQ_LEN", "8192"))
    config = build_config(hidden)
    torch.manual_seed(1234)  # identical weights across the two layers' runs
    layer = build_layer(config, pg)
    dist.barrier()

    cases = [
        ("balanced", dict(seed=11)),
        ("imbalance", dict(seed=22)),
        ("zero_peer_group", dict(seed=33)),
        ("zero_expert", dict(seed=44)),
    ]
    # BT_T2_SKIP_FIXC=1 skips the FIX-C composition variants entirely (use when
    # FIX C is blocked/parked — the W2 gate then runs C-free).
    fixc_cases = [] if os.environ.get("BT_T2_SKIP_FIXC", "0") == "1" else ["balanced", "zero_peer_group"]

    for case, kw in cases:
        for use_ckpt in (False, True):
            for use_fixc in ([False, True] if (use_ckpt and case in fixc_cases) else [False]):
                if use_fixc:
                    os.environ["BT_MOE_DISPATCH_REPLAY_CACHE"] = "1"
                probs, routing_map = craft_routing(case, seq, "cuda", kw["seed"])
                g = torch.Generator().manual_seed(kw["seed"] + 1000)
                hidden_states = torch.randn(
                    seq, 1, hidden, generator=g, dtype=torch.bfloat16
                ).cuda()

                # Run A: W1 off, W2 off (status quo)
                set_w1(layer, False)
                set_w2(layer, False)
                out_a, grad_a = run_once(
                    layer, config, hidden_states, probs, routing_map, use_ckpt, use_fixc
                )
                # Run B: W1 on, W2 on
                set_w1(layer, True)
                set_w2(layer, True)
                from megatron.core.transformer.moe import token_dispatcher as _td

                fixc_before = dict(_td._REPLAY_STATS) if use_fixc else None
                out_b, grad_b = run_once(
                    layer, config, hidden_states, probs, routing_map, use_ckpt, use_fixc
                )
                # Run C: W1 off, W2 on (probs on the EP comm, issued first)
                set_w1(layer, False)
                out_c, grad_c = run_once(
                    layer, config, hidden_states, probs, routing_map, use_ckpt, use_fixc
                )

                tag = f"{case} ckpt={int(use_ckpt)} fixc={int(use_fixc)}"
                check(f"{tag}: W2 output == monolithic output", torch.equal(out_a, out_b))
                check(f"{tag}: W2 input grad == monolithic", torch.equal(grad_a, grad_b))
                check(f"{tag}: W2(no-W1) output == monolithic", torch.equal(out_a, out_c))
                check(f"{tag}: W2(no-W1) input grad == monolithic", torch.equal(grad_a, grad_c))
                if use_fixc:
                    # Second-review item 6: the replay cache must have HIT on
                    # the W2-on replay (a silent fallback would still pass
                    # torch.equal — the v1 inert-gate trap).
                    fixc_after = dict(_td._REPLAY_STATS)
                    check(
                        f"{tag}: FIX-C replay cache hit on W2-on run "
                        f"(hits {fixc_before['hits']} -> {fixc_after['hits']})",
                        fixc_after["hits"] > fixc_before["hits"],
                    )
                    check(
                        f"{tag}: FIX-C no misses on W2-on run "
                        f"(misses {fixc_before['misses']} -> {fixc_after['misses']})",
                        fixc_after["misses"] == fixc_before["misses"],
                    )
                    os.environ.pop("BT_MOE_DISPATCH_REPLAY_CACHE", None)

    dist.barrier()
    ok = not FAILURES
    if rank == 0:
        print("=" * 60)
        print("ALL PASS" if ok else f"{len(FAILURES)} FAILURES: {FAILURES}", flush=True)
    dist.destroy_process_group()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
