#!/usr/bin/env python3
"""Repro: output-discarding recompute x fine-grained activation offload.

LPS-1062, turing 2026-08-21. One GPU, no model, no 131k context, no
distributed init, seconds per run. Written because the guided config
(recompute_modules=[core_attn, layernorm, moe_act, mla_up_proj] with
offload_modules=[attn_proj, expert_fc1]) killed every first-stage rank during
startup warmup with:

    RuntimeError: Trying to backward through the graph a second time (or
    directly access saved tensors after they have already been freed).
      at megatron/core/tensor_parallel/random.py:771, in
         CheckpineWithoutOutput._recompute  ->  inputs = self.ctx.saved_tensors

and bisecting that by trainer boots costs ~20 minutes per bit.

WHY THESE TWO CLASSES COLLIDE
-----------------------------
`CheckpointWithoutOutput` (used by the recompute names moe_act, layernorm,
mla_up_proj, gdn_norm_out) discards its region's OUTPUT in forward and
regenerates it in backward by re-running the region. To re-run it needs the
region's INPUTS, and `_recompute` reads them from `self.ctx.saved_tensors`.

The offload installs `torch.autograd.graph.saved_tensors_hooks`, which replaces
every tensor autograd saves inside an offload window with a tag and then frees
the device storage (`forced_released_tensors` -> untyped_storage().resize_(0)).

So if a CheckpointWithoutOutput region's saved inputs fall inside an offload
window, `_recompute` asks for tensors that are gone.

Evidence this is a defect and not a misconfiguration: the sibling method
`CheckpointWithoutOutputFunction.backward` in the same class deliberately reads
`ctx.inputs` instead, commented "to avoid double-reloading the inputs in CPU
offloading scenario". One path was made offload-aware; `_recompute` was not.

THE ORDERING MIRRORED HERE
--------------------------
From absorbed_mla.py, the real GLM-5.2 attention path:

    :659  self.qkv_up_checkpoint = CheckpointWithoutOutput(...)      # mla_up_proj
    :663  q_absorbed, kv = self.qkv_up_checkpoint.checkpoint(fn, ...)
    ...
    :902  self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
    :925  attn_proj_manager = off_interface(self.offload_attn_proj, core_attn_out,
                                            "attn_proj")

Note both attach to the SAME tensor, core_attn_out.

USAGE
-----
    NVTE_CPU_OFFLOAD_V1=1 CUDA_VISIBLE_DEVICES=<free gpu> \
        <worker-venv python> offload_recompute_interop_repro.py

Exits non-zero if the 2x2 matrix does not come out as expected, so it can gate
a boot.
"""

from __future__ import annotations

import os
import sys
import traceback

N, D = 4096, 1024  # 4.2M elements -> comfortably over min_offloaded_tensor_size


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("FAIL: no CUDA device")
        return 2

    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
          f"NVTE_CPU_OFFLOAD_V1={os.environ.get('NVTE_CPU_OFFLOAD_V1')}")

    from megatron.core.pipeline_parallel import fine_grained_activation_offload as fo
    from megatron.core.tensor_parallel.random import CheckpointWithoutOutput

    off_interface = fo.FineGrainedActivationOffloadingInterface

    def fresh_manager(cap=None):
        """A manager + one forward chunk, as init_chunk_handler does per microbatch."""
        fo.PipelineOffloadManager.reset_instance()
        fo.PipelineOffloadManager()
        off_interface.init_chunk_handler(
            0,      # pp_rank
            1,      # vp_size
            0,      # vp_stage
            1024,   # min_offloaded_tensor_size (elements)
            0,      # delta_offload_bytes_across_pp_ranks
            1.0,    # activation_offload_fraction
            max_inflight_offloads=cap,
        )

    W1 = torch.randn(D, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    W2 = torch.randn(D, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def region(a):
        """Stand-in for a cheap output-discarding region (mla_up_proj / layernorm)."""
        return torch.nn.functional.layer_norm(a.float(), (D,)).to(torch.bfloat16) * 1.5

    def scenario(use_recompute: bool, use_offload: bool, ckpt_inside_window: bool):
        """Mirror absorbed_mla's ordering.

        h  ~ q_absorbed/kv_compressed : the DISCARDED checkpoint output
        z  ~ core_attn_out            : downstream tensor that saves h, and the
                                        tensor both the recompute hook and the
                                        offload window attach to
        ckpt_inside_window: whether the checkpoint's own save_for_backward
            happens inside an offload window. This is the precondition under
            test -- only then are the region's saved INPUTS intercepted and
            freed, which is what _recompute later cannot find.
        """
        fresh_manager()
        x = torch.randn(N, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        ckpt = CheckpointWithoutOutput(fp8=False) if use_recompute else None

        def make_h():
            return ckpt.checkpoint(region, x) if use_recompute else region(x)

        if ckpt_inside_window and use_offload:
            # An enclosing group, as a real layer forward has around its parts.
            outer = off_interface(True, x, "attn_norm")
            with outer as x_in:
                h = ckpt.checkpoint(region, x_in) if use_recompute else region(x_in)
                z = h @ W1
            z = outer.group_offload(z, forced_released_tensors=[])
        else:
            h = make_h()
            z = h @ W1

        # absorbed_mla:902 -- hook on the DOWNSTREAM tensor, before the window
        if use_recompute:
            ckpt.discard_output_and_register_recompute(z)

        # absorbed_mla:925 -- attn_proj window over that same tensor
        mgr = off_interface(use_offload, z, "attn_proj")
        with mgr as z2:
            y = z2 @ W2
        y = mgr.group_offload(y, forced_released_tensors=[])

        y.float().sum().backward()

        # Prove the mechanism engaged. An "OK" from a case that silently
        # offloaded nothing tells us nothing at all -- the same failure mode as
        # an arm wearing the baseline's label. commits>0 means bulk_offload_group
        # really ran; this needs BT_OFFLOAD_VALVE_TELEMETRY=1 and the turing
        # counter fix (Megatron-LM 2c3d720cf), without which uncapped runs
        # report nothing.
        tele = fo.PipelineOffloadManager.get_instance().get_valve_telemetry()
        commits = sum(v[0] for v in tele.values())
        assert x.grad is not None, "no gradient reached x"
        return commits, tele

    CASES = [
        ("A recompute only                         ", True,  False, False),
        ("B offload only                           ", False, True,  False),
        ("C both, checkpoint OUTSIDE offload window", True,  True,  False),
        ("D both, checkpoint INSIDE  offload window", True,  True,  True),
    ]

    results = {}
    for label, rec, off, inside in CASES:
        try:
            commits, tele = scenario(rec, off, inside)
            vacuous = off and commits == 0
            results[label.strip()] = ("VACUOUS", f"offload requested but commits={commits}", "") if vacuous else None
            tag = "VACUO" if vacuous else "OK   "
            print(f"{tag} {label} | offload commits={commits} {dict(tele) if tele else ''}")
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            where = ""
            for line in tb.splitlines():
                if "random.py" in line and ", line " in line:
                    where = line.strip()
            results[label.strip()] = (type(e).__name__, str(e).splitlines()[0][:150], where)
            print(f"FAIL  {label}: {type(e).__name__}: {str(e).splitlines()[0][:110]}")
            if where:
                print(f"          {where}")

    print("\n--- verdict ---")
    a = results["A recompute only"]
    b = results["B offload only"]
    c = results["C both, checkpoint OUTSIDE offload window"]
    d = results["D both, checkpoint INSIDE  offload window"]

    if a is not None or b is not None:
        print("SCAFFOLDING WRONG: a single mechanism failed on its own, so this "
              "toy does not yet model either mechanism correctly.")
        return 1

    TARGET = "Trying to backward through the graph a second time"
    def is_target(r):
        return r is not None and TARGET in r[1]

    if is_target(d) and c is None:
        print("REPRODUCED, and the PRECONDITION is identified: the collision "
              "happens only when the output-discarding checkpoint's own "
              "save_for_backward runs INSIDE an offload window, so its saved "
              "inputs are intercepted and freed before _recompute asks for them.")
        print(f"  site: {d[2]}")
        print("  => a config is safe iff no recompute_modules region using "
              "CheckpointWithoutOutput (moe_act, layernorm, mla_up_proj, "
              "gdn_norm_out) sits inside an active offload window.")
        return 0
    if is_target(c) or is_target(d):
        print("REPRODUCED the trainer error, but not cleanly separated by the "
              "window precondition. Inspect the per-case results above.")
        return 0
    print("NOT REPRODUCED: both combinations passed. The trainer failure needs "
          "something this toy still does not model -- most likely pipeline "
          "chunk bookkeeping across microbatches, or a different module pair.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
