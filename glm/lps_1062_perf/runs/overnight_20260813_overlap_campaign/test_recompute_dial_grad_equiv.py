# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Grad-equivalence + LoRA-trap tests for the per-layer recompute dial (LPS-1062).

Spec: trainers `experiment_artefacts/glm/lps_1062_perf/runs/
overnight_20260813_overlap_campaign/DIAL_GRAD_EQUIV_SPEC.md` (T1-T7).
Interface under test (frozen 2026-08-13, branch jackrao/lps-1062-recompute-dial,
mcore tip 06393114b): `moe_ep_overlap_checkpoint_num_layers` +
`build_checkpointed_layer_callables(layer)` in fine_grained_callables.py.

T1-T3 drive the opaque callable STANDALONE (mock node, no executor — jacobi's
recipe). T4-T7 construct the real TransformerModelChunkSchedulePlan at PP1.

Requires CUDA. NOTE: the overlap flag is rejected at EP1
(transformer_config.py:2624 "only supported with expert model parallelism") —
run under torchrun with EP2 (2 GPUs):

    DIAL_TEST_EP=2 <venv>/python -m torch.distributed.run --nproc_per_node=2 \
        <venv>/pytest tests/unit_tests/a2a_overlap/test_recompute_dial_grad_equiv.py -x -q

(Each rank runs the suite independently on its own EP-sharded experts; all
asserts are per-rank, matching the existing a2a_overlap suite convention.)
"""

import inspect
import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core import tensor_parallel
from megatron.core.models.common.model_chunk_schedule_plan import (
    TransformerModelChunkSchedulePlan,
)
from megatron.core.models.gpt.fine_grained_callables import (
    PreProcessNode,
    TransformerLayerNode,
    build_checkpointed_layer_callables,
)
from megatron.core.pipeline_parallel.utils import NoopScheduleNode, set_streams
from megatron.core.transformer.module import float16_to_fp32
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.a2a_overlap.utils import (
    compare_captures,
    deterministic_mode,
)
from tests.unit_tests.test_utilities import Utils

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_SEQ_LEN = 32
_DIAL_EP = int(os.environ.get("DIAL_TEST_EP", "1"))


def _dial_config(num_layers=2, dial_k=None, dropout=0.0, moe_layer_freq=None):
    """Tiny MLA-MoE config with the overlap flag on and the dial set.

    EP1 default (grad-equivalence of the checkpoint mechanism is EP-orthogonal);
    override with DIAL_TEST_EP. LoRA regime is emulated by freezing, see
    _freeze_to_adapter_only. Values mirror utils.get_test_config (which cannot
    be reused: its hardcoded EP4 collides with extra_kwargs).
    """
    from megatron.core.transformer.transformer_config import MLATransformerConfig

    kwargs = dict(
        attention_backend="unfused",
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=_DIAL_EP,
        deterministic_mode=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        num_layers=num_layers,
        hidden_size=512,
        add_bias_linear=False,
        num_attention_heads=128,
        ffn_hidden_size=512,
        kv_channels=128,
        hidden_dropout=dropout,
        attention_dropout=dropout,
        multi_latent_attention=True,
        qk_head_dim=64,
        qk_pos_emb_head_dim=64,  # q_head_dim = 128: the MLA default (128+64=192)
        # trips flash-attn cute's "Must use 2CTA for hdim 192" assert on sm100
        num_moe_experts=4,
        moe_grouped_gemm=True,
        moe_router_dtype="fp32",
        moe_token_dispatcher_type="alltoall",  # the flag requires alltoall/flex (:2626)
        overlap_moe_expert_parallel_comm=True,
        recompute_granularity=None,  # the dial path never uses the block machinery
    )
    if moe_layer_freq is not None:
        kwargs["moe_layer_freq"] = moe_layer_freq
    if dial_k is not None:
        kwargs["moe_ep_overlap_checkpoint_num_layers"] = dial_k
    return MLATransformerConfig(**kwargs)


def _build_model(config):
    """Tiny GPTModel (2 MoE layers unless overridden), per the a2a crib pattern."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    from megatron.core.models.gpt.gpt_model import GPTModel

    ids = [i for i in range(_SEQ_LEN)]
    data = {
        "input_ids": torch.tensor(ids, dtype=torch.int64).repeat((1, 1)).cuda(),
        "labels": torch.tensor(ids, dtype=torch.int64).repeat((1, 1)).cuda(),
        "position_ids": torch.tensor([i for i in range(_SEQ_LEN)], dtype=torch.int64)
        .repeat((1, 1))
        .cuda(),
        "attention_mask": torch.ones((1, 1, _SEQ_LEN, _SEQ_LEN), dtype=bool).cuda(),
    }
    spec = get_gpt_decoder_block_spec(config=config, use_transformer_engine=True)
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=128,
        pre_process=True,
        post_process=True,
        max_sequence_length=300,
    )
    return model, data


def _freeze_to_adapter_only(model, layer_indices=None):
    """Emulate the mission's adapter-only LoRA regime.

    Freeze EVERYTHING (incl. embedding — the frozen-embedding chain is what
    makes checkpoint inputs arrive requires_grad=False), then unfreeze one
    small param per selected MoE layer as the adapter stand-in: the router
    weight (its grad flows through routing, so it is nonzero iff backward
    fires into the layer). `layer_indices=None` selects all MoE layers
    (plan-level tests); the standalone callable tests pass [0] because only
    layer 0 is driven — an unfrozen param on a layer that never runs would
    read as a false trap. Both arms of every comparison share the same freeze
    pattern, so the math cancels; the trap detector is sharp because the trap
    produces EXACTLY None/zero grads.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    adapters = []
    for i, layer in enumerate(model.decoder.layers):
        if layer_indices is not None and i not in layer_indices:
            continue
        if hasattr(layer.mlp, "router") and hasattr(layer.mlp.router, "weight"):
            layer.mlp.router.weight.requires_grad_(True)
            adapters.append(layer.mlp.router.weight)
    assert adapters, "no adapter stand-in found (router weights) — fixture broken"
    return adapters


def _mock_node(model, is_last_layer=False, rotary_pos_emb=None):
    """jacobi's standalone-drive recipe: the minimal node the callable reads."""
    return SimpleNamespace(
        chunk_state=SimpleNamespace(
            attention_mask=None,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            packed_seq_params=None,
            sequence_len_offset=None,
            padding_mask=None,
            model=SimpleNamespace(
                decoder=SimpleNamespace(final_layernorm=None)
            ),
        ),
        is_last_layer=is_last_layer,
    )


def _rotary_for(model, data):
    """Best-effort real rotary embedding from the model, else None."""
    rot = getattr(model, "rotary_pos_emb", None)
    if rot is None:
        return None
    try:
        return rot(data["position_ids"])
    except Exception:
        return None


def _layer_forward(layer, chunk_state, hidden):
    """Mirror of the opaque callable's custom_forward call exactly."""
    output, _ = layer(
        hidden_states=hidden,
        attention_mask=chunk_state.attention_mask,
        rotary_pos_emb=chunk_state.rotary_pos_emb,
        rotary_pos_cos=chunk_state.rotary_pos_cos,
        rotary_pos_sin=chunk_state.rotary_pos_sin,
        packed_seq_params=chunk_state.packed_seq_params,
        sequence_len_offset=chunk_state.sequence_len_offset,
        padding_mask=chunk_state.padding_mask,
    )
    return output


def _fresh_input(model):
    hidden = 512
    base = (
        torch.randn(_SEQ_LEN, 1, hidden, dtype=torch.bfloat16, device="cuda") * 0.1
    )
    return base  # requires_grad=False: the frozen-embedding regime


def _adapter_grads(adapters):
    return [a.grad for a in adapters]


def _force_block_input_grad(module, args, kwargs):
    """Mirror of the bridge PEFT patch's effect (megatron-bridge
    peft/recompute.py:96-105): force the block input to require grad so the
    checkpointed layers' backward fires under adapter-only LoRA. A raw-mcore
    test fixture needs this explicitly — the mission's stock path gets the
    force from the bridge patch, which raw mcore does not include (measured:
    without it, the stock reference's loss comes out requires_grad=False).

    NOTE: GPTModel calls the decoder with KEYWORD args (gpt_model.py:572) —
    register with with_kwargs=True and mutate kwargs['hidden_states'].
    """
    hs = kwargs.get("hidden_states", args[0] if args else None)
    if hs is not None and hs.is_floating_point() and not hs.requires_grad:
        kwargs["hidden_states"] = hs.detach().requires_grad_(True)
        return (args, kwargs)
    return None


def _assert_adapters_nonzero(adapters, where):
    for i, a in enumerate(adapters):
        assert a.grad is not None, f"{where}: adapter {i} grad is None (the trap fired)"
        assert a.grad.abs().sum() > 0, f"{where}: adapter {i} grad is exactly zero (the trap fired)"


# ---------------------------------------------------------------------------
# T1-T3: standalone callable drive (no executor)
# ---------------------------------------------------------------------------


class TestDialCallableStandalone:
    """T1-T3: build_checkpointed_layer_callables driven via a mock node."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=_DIAL_EP,
        )
        set_streams()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _run_dial_and_reference(self, dropout=0.0, disable_flip=False, monkeypatch=None):
        """One seed, one model, one input: dial callable vs stock checkpoint.

        Returns (out_dial, out_ref, adapters, hidden_dial, hidden_ref).
        """
        with deterministic_mode():
            config = _dial_config(num_layers=2, dial_k=1, dropout=dropout)
            model, data = _build_model(config)
            model.cuda()
            adapters = _freeze_to_adapter_only(model, layer_indices=[0])
            layer = model.decoder.layers[0]  # first MoE layer (K=1 target)
            rotary = _rotary_for(model, data)
            base = _fresh_input(model)

            # ---- dial path: the opaque callable, framework-exact input handling
            fwd_funcs, backward_dw = build_checkpointed_layer_callables(layer)
            assert backward_dw == {}, "dial contract: empty dw map"
            ckpt_fn = fwd_funcs[0]
            assert all(f is None for f in fwd_funcs[1:]), "dial contract: slots 1-4 None"
            node = _mock_node(model, rotary_pos_emb=rotary)

            if disable_flip:
                # T2 arm A: stub out the in-place requires_grad_ flip. Scoped to
                # the dial call only. (The framework-mimicry line uses the
                # property SETTER, not the patched method.)
                assert monkeypatch is not None
                monkeypatch.setattr(
                    torch.Tensor, "requires_grad_", lambda self, *a, **k: self
                )

            # ScheduleNode._forward mimicry (pipeline_parallel/utils.py:205-213):
            hidden_dial = base.detach()
            hidden_dial.requires_grad = base.requires_grad  # False: frozen regime
            out_dial = ckpt_fn(node, hidden_dial)

            if disable_flip:
                monkeypatch.undo()

            # ---- reference path: stock checkpoint primitive, stock block behavior
            # (the block path forces requires_grad=True on the block input via
            # make_viewless_tensor — so the reference input requires grad).
            hidden_ref = base.detach().requires_grad_(True)
            out_ref = tensor_parallel.checkpoint(
                lambda h: _layer_forward(layer, node.chunk_state, h), False, hidden_ref
            )
            return out_dial, out_ref, adapters, hidden_dial, hidden_ref, model

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE")
    def test_t1_lora_trap_grad_equivalence(self):
        """T1 PRIMARY: adapter grads nonzero + equal; input grad exists + equal."""
        out_dial, out_ref, adapters, hidden_dial, hidden_ref, model = (
            self._run_dial_and_reference()
        )

        # forward: bitwise
        assert torch.equal(
            out_dial.view(torch.int16), out_ref.view(torch.int16)
        ), "T1: dial vs stock checkpoint forward not bitwise-equal"

        loss_dial = float16_to_fp32(out_dial).sum()
        loss_dial.backward()
        _assert_adapters_nonzero(adapters, "T1 dial path")
        dial_grads = [a.grad.clone() for a in adapters]
        dial_input_grad = hidden_dial.grad
        assert dial_input_grad is not None, "T1: no input grad on the dial path"

        model.zero_grad()
        loss_ref = float16_to_fp32(out_ref).sum()
        loss_ref.backward()
        _assert_adapters_nonzero(adapters, "T1 reference path")

        for i, (g_d, g_r) in enumerate(zip(dial_grads, _adapter_grads(adapters))):
            max_diff = (g_d - g_r).abs().max().item()
            assert torch.allclose(
                g_d, g_r, rtol=1e-5, atol=1e-8
            ), f"T1: adapter {i} grad mismatch, max abs diff {max_diff}"
            assert max_diff == 0.0 or max_diff < 1e-6, (
                f"T1: adapter {i} grad not bitwise (max diff {max_diff}) — "
                "investigate before accepting"
            )
        assert torch.allclose(
            dial_input_grad, hidden_ref.grad, rtol=1e-5, atol=1e-8
        ), "T1: input grad at the opaque boundary mismatch"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE")
    def test_t2_negative_control(self, monkeypatch):
        """T2: with the flip stubbed out, the trap MUST fire (else test broken).

        Arm A (callable): output must come out requires_grad=False (plain
        autograd.Function semantics), backward must refuse, adapter grads stay
        None. Arm B (source guard): the PreProcessNode grad-root fix hunk must
        be present in the installed source (the pre-existing landmine's
        detector — Aug-9 source-guard pattern).
        """
        out_dial, _, adapters, _, _, _ = self._run_dial_and_reference(
            disable_flip=True, monkeypatch=monkeypatch
        )
        assert not out_dial.requires_grad, (
            "T2 arm A BROKEN TEST: with the requires_grad_ flip stubbed out and a "
            "non-grad input, the checkpoint output still requires grad — the trap "
            "is not load-bearing in this construction; fix the test, not the patch"
        )
        with pytest.raises(RuntimeError):
            float16_to_fp32(out_dial).sum().backward()
        for i, a in enumerate(adapters):
            assert a.grad is None or a.grad.abs().sum() == 0, (
                f"T2 arm A: adapter {i} grad present despite the trap — test broken"
            )

        # Arm B: source guard for the PreProcessNode grad-root fix (b907b6153).
        src = inspect.getsource(PreProcessNode)
        assert "requires_grad_(True)" in src, (
            "T2 arm B: PreProcessNode grad-root fix (b907b6153) NOT present in the "
            "installed tree — the first stage-0 backward will root at a non-grad "
            "tensor (RuntimeError). Run the branch tip."
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE")
    def test_t3_rng_fork_restore(self):
        """T3: dropout active -> the checkpoint's recompute must reproduce the
        forward's RNG mask. Sharp form: dial callable vs EAGER ground truth
        from the same RNG state.

        The process-global RNG tracker is shared across arms, so sequential
        arms WITHOUT a restore see different dropout masks (one-model and
        two-model structures both suffer — measured). Snapshot/restore around
        each arm is the only correct construction. NOTE: tracker.get_states()
        copies the dict but shares the state tensors (docstring) — safe iff the
        tracker replaces rather than mutates states (standard mcore); if this
        test still fails, that sharing is the first suspect.
        """
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

        def _snap():
            return (
                torch.get_rng_state(),
                torch.cuda.get_rng_state(),
                get_cuda_rng_tracker().get_states(),
            )

        def _restore(s):
            torch.set_rng_state(s[0])
            torch.cuda.set_rng_state(s[1])
            get_cuda_rng_tracker().set_states(s[2])

        with deterministic_mode():
            config = _dial_config(num_layers=2, dial_k=1, dropout=0.1)
            model, data = _build_model(config)
            model.cuda()
            adapters = _freeze_to_adapter_only(model, layer_indices=[0])
            layer = model.decoder.layers[0]
            rotary = _rotary_for(model, data)
            node = _mock_node(model, rotary_pos_emb=rotary)
            base = _fresh_input(model)

            snap = _snap()
            fn, _ = build_checkpointed_layer_callables(layer)
            hidden_d = base.detach()
            hidden_d.requires_grad = base.requires_grad
            out_dial = fn[0](node, hidden_d)
            float16_to_fp32(out_dial).sum().backward()
            grads_dial = [a.grad.clone() for a in adapters]

            _restore(snap)
            model.zero_grad()
            hidden_e = base.detach().requires_grad_(True)
            out_eager = _layer_forward(layer, node.chunk_state, hidden_e)
            float16_to_fp32(out_eager).sum().backward()
            grads_eager = [a.grad.clone() for a in adapters]

            assert torch.equal(
                out_dial.view(torch.int16), out_eager.view(torch.int16)
            ), "T3: checkpointed vs eager forward differ with active RNG"
            for i, (g_d, g_e) in enumerate(zip(grads_dial, grads_eager)):
                assert torch.equal(g_d.view(torch.int16), g_e.view(torch.int16)), (
                    f"T3: adapter {i} grad not bitwise vs eager ground truth — "
                    "checkpoint recompute RNG divergence (risk §6.2)"
                )


# ---------------------------------------------------------------------------
# T4-T7: real plan construction at PP1
# ---------------------------------------------------------------------------


class TestDialPlan:
    """T4-T7: TransformerModelChunkSchedulePlan with the dial active."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=_DIAL_EP,
        )
        set_streams()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE")
    def test_t4_k0_equals_no_dial(self):
        """T4: K=0 is a legal no-op — plan identical to the no-dial plan."""
        with deterministic_mode():
            cfg_a = _dial_config(num_layers=2, dial_k=0)
            model_a, data_a = _build_model(cfg_a)
            _set_random_seed(123, data_parallel_random_init=False)  # identical re-init
            cfg_b = _dial_config(num_layers=2, dial_k=None)
            model_b, data_b = _build_model(cfg_b)
            model_a.cuda()
            model_b.cuda()
            # same seed inside deterministic_mode -> identical weights
            for pa, pb in zip(model_a.parameters(), model_b.parameters()):
                assert torch.equal(pa, pb), "T4 fixture: models diverged at init"

            plan_a = model_a.build_schedule_plan(**data_a)
            plan_b = model_b.build_schedule_plan(**data_b)
            for i in range(plan_a.num_layers()):
                la, lb = plan_a.get_layer(i), plan_b.get_layer(i)
                # both fully eager: no NoopScheduleNode in the comm slots
                for slot in ("mlp", "moe_dispatch", "moe_combine"):
                    assert not isinstance(getattr(la, slot), NoopScheduleNode), (
                        f"T4: K=0 layer {i} slot {slot} is a no-op — dial active at K=0?"
                    )
                    assert not isinstance(getattr(lb, slot), NoopScheduleNode)

            # numeric equivalence of the two plans
            out_a = TransformerModelChunkSchedulePlan.run(plan_a, None)
            TransformerModelChunkSchedulePlan.run(
                None, plan_a, b_grad=torch.ones_like(out_a)
            )
            out_b = TransformerModelChunkSchedulePlan.run(plan_b, None)
            TransformerModelChunkSchedulePlan.run(
                None, plan_b, b_grad=torch.ones_like(out_b)
            )
            assert torch.equal(
                out_a.view(torch.int16), out_b.view(torch.int16)
            ), "T4: K=0 plan output differs from no-dial plan"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE")
    def test_t5_k_equals_l_vs_stock_full_recompute(self):
        """T5: all-opaque plan vs stock block-level full recompute (flag OFF)."""
        with deterministic_mode():
            cfg_dial = _dial_config(num_layers=2, dial_k=2)
            model_dial, data = _build_model(cfg_dial)
            cfg_ref = _dial_config(num_layers=2, dial_k=None)
            cfg_ref.overlap_moe_expert_parallel_comm = False
            cfg_ref.recompute_granularity = "full"
            cfg_ref.recompute_method = "uniform"
            cfg_ref.recompute_num_layers = 1
            _set_random_seed(123, data_parallel_random_init=False)  # identical re-init
            model_ref, _ = _build_model(cfg_ref)
            model_dial.cuda()
            model_ref.cuda()
            for pa, pb in zip(model_dial.parameters(), model_ref.parameters()):
                assert torch.equal(pa, pb), "T5 fixture: models diverged at init"
            adapters_dial = _freeze_to_adapter_only(model_dial)
            adapters_ref = _freeze_to_adapter_only(model_ref)
            # The stock reference needs the bridge-patch-equivalent grad-root
            # force on the block input (raw mcore lacks the bridge's PEFT
            # patch; without it the reference's own checkpointed backward never
            # fires under this all-frozen fixture — measured failure).
            model_ref.decoder.register_forward_pre_hook(
                _force_block_input_grad, with_kwargs=True
            )

            # dial path: single-plan forward then backward on the SAME plan
            # (crib pattern: run(f_plan, None) then run(None, plan, b_grad))
            plan = model_dial.build_schedule_plan(**data)
            out_dial = TransformerModelChunkSchedulePlan.run(plan, None)
            TransformerModelChunkSchedulePlan.run(
                None, plan, b_grad=torch.ones_like(out_dial)
            )

            # reference path: plain forward under stock full recompute
            loss_ref = model_ref.forward(**data)
            loss_ref = float16_to_fp32(loss_ref)
            loss_ref.backward(torch.ones_like(loss_ref))

            _assert_adapters_nonzero(adapters_ref, "T5 reference")
            _assert_adapters_nonzero(adapters_dial, "T5 dial (trap at plan level)")

            # outputs bitwise (same checkpoint primitive per layer, same RNG)
            assert torch.equal(
                out_dial.view(torch.int16), loss_ref.view(torch.int16)
            ), "T5: all-opaque plan output differs from stock full-recompute"
            # adapter grads bitwise
            for i, (a_d, a_r) in enumerate(zip(adapters_dial, adapters_ref)):
                assert torch.equal(
                    a_d.grad.view(torch.int16), a_r.grad.view(torch.int16)
                ), f"T5: adapter {i} grad not bitwise vs stock full recompute"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE")
    def test_t6_node_contract(self):
        """T6: the opaque layer's node contract, per the frozen build."""
        with deterministic_mode():
            config = _dial_config(num_layers=2, dial_k=1)
            model, data = _build_model(config)
            model.cuda()
            plan = model.build_schedule_plan(**data)

            opaque = plan.get_layer(0)  # K=1 -> layer-local index 0 (MoE) opaque
            eager = plan.get_layer(1)

            # opaque: real attn node on the comp stream; everything else no-op
            assert isinstance(opaque.attn, TransformerLayerNode)
            for slot in ("mlp", "moe_dispatch", "moe_combine"):
                assert isinstance(getattr(opaque, slot), NoopScheduleNode), (
                    f"T6: opaque layer slot {slot} must be NoopScheduleNode"
                )
            # the schedule calls mlp.backward_dw() unconditionally -> no-op exists
            opaque.mlp.backward_dw()  # must not raise
            # NOTE: delay_wgrad_compute=False is forced by the plan builder in
            # extra_args at construction (code-verified on the branch); the
            # observable contract from here is the empty dw map + no-op-safe
            # backward_dw calls, asserted above.

            # eager layer: full five-node decomposition intact
            assert isinstance(eager.attn, TransformerLayerNode)
            assert not isinstance(eager.mlp, NoopScheduleNode)
            assert not isinstance(eager.moe_dispatch, NoopScheduleNode)
            assert not isinstance(eager.moe_combine, NoopScheduleNode)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE")
    def test_t7_dense_exclusion_and_k_semantics(self):
        """T7: dense layers are excluded from the dial; K counts stage-local index."""
        with deterministic_mode():
            # layers: [dense, moe, moe] via moe_layer_freq list
            config = _dial_config(num_layers=3, dial_k=1, moe_layer_freq=[0, 1, 1])
            model, data = _build_model(config)
            model.cuda()
            plan = model.build_schedule_plan(**data)

            # layer 0: dense -> never dial-checkpointed even though index 0 < K=1
            assert not isinstance(plan.get_layer(0).mlp, NoopScheduleNode), (
                "T7: dense layer was dial-checkpointed — MoE-only semantics broken"
            )
            # layer 1: MoE at stage-local index 1, NOT < K=1 -> stays eager
            assert not isinstance(plan.get_layer(1).mlp, NoopScheduleNode), (
                "T7: MoE layer at index 1 checkpointed at K=1 — K is a stage-local "
                "INDEX bound, not a count of MoE layers (dense prefix consumes index)"
            )

            # K=2: index 1 < 2 -> the first MoE layer goes opaque
            _set_random_seed(123, data_parallel_random_init=False)  # identical re-init
            config2 = _dial_config(num_layers=3, dial_k=2, moe_layer_freq=[0, 1, 1])
            model2, data2 = _build_model(config2)
            model2.cuda()
            plan2 = model2.build_schedule_plan(**data2)
            assert not isinstance(plan2.get_layer(0).mlp, NoopScheduleNode)
            assert isinstance(plan2.get_layer(1).mlp, NoopScheduleNode), (
                "T7: K=2 should checkpoint the MoE layer at stage-local index 1"
            )
            assert not isinstance(plan2.get_layer(2).mlp, NoopScheduleNode)
