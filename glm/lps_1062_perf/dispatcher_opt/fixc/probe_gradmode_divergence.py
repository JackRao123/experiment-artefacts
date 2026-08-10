#!/usr/bin/env python3
"""On-box probe for the ARM-1 FIX-C verify failure: does grad mode change
kernel/engine selection (and therefore values) in the GLM-5.2 compute chain?

Hypothesis (b) — REFUTED for GEMMs on-box (2026-08-09): vanilla torch linear,
TE Linear bf16, TE Linear FP8 e4m3, and a 4-deep FP8 stack are ALL
torch.equal across no_grad vs enable_grad. Leading candidate after the
refutation: the cuDNN DSA ATTENTION forward — attention fwd kernels
legitimately select different engine variants under grad vs no-grad
(training mode saves softmax stats / different tiling), so the no-grad first
pass vs the grad-enabled replay can differ at ULP in the attention OUTPUT ->
hidden drift -> router logits -> boundary topk flips (the verify failure
signature: num_tokens_per_local_expert_dev mismatch at routing_shape
(8192, 256); the flip sim below maps ULP magnitude to the expected flipped-row
count).

Runs the SAME input through each candidate twice IN ONE PROCESS, no_grad vs
enable_grad, and prints torch.equal + max-abs-diff:

  1. vanilla torch linear (the router's gating GEMM class);
  2. TE Linear, bf16, no FP8;
  3. TE Linear, FP8 e4m3 blockwise (the ship recipe);
  4. a 4-deep TE Linear FP8 stack (accumulated drift, upstream-chain proxy);
  5. THE KILL SHOT: run_fused_absorbed_sparse_attention forward under no_grad
     vs enable_grad on identical inputs (the one op class the v1 probe
     skipped — the cuDNN DSA attention forward);
  6. router-flip simulation: ULP-level noise on 256-expert logits -> how many
     topk-8 boundary flips result.

Run on the box with the trainer's python env (torch + transformer_engine +
the vendored mcore + the cuDNN DSA backend):

  # LD_LIBRARY_PATH must include the cuDNN/cuDNN-frontend libs (the trainer
  # launcher sets it; e.g. source the bench env or
  # export LD_LIBRARY_PATH=/path/to/cudnn/lib:$LD_LIBRARY_PATH) — without it
  # the fused backend import fails and case 5 cannot run.
  python probe_gradmode_divergence.py
"""

import sys

import torch

if not torch.cuda.is_available():
    sys.stderr.write(
        "ERROR: this probe needs a CUDA box (the trainer image). On Mac CPU there is"
        " nothing to measure — the grad-mode kernel-selection question is GPU-only.\n"
    )
    sys.exit(2)

print(f"torch {torch.__version__}, cuda {torch.version.cuda}, device {torch.cuda.get_device_name(0)}")


def probe(name, fn, x):
    with torch.no_grad():
        a = fn(x)
    with torch.enable_grad():
        b = fn(x)
    eq = torch.equal(a, b)
    md = (a - b).abs().max().item() if not eq else 0.0
    print(f"{name:44s} equal={eq}  max|diff|={md:.3e}")
    return eq


# 1. vanilla linear (router gating class)
w = torch.randn(256, 2048, dtype=torch.bfloat16, device="cuda")
x = torch.randn(8192, 2048, dtype=torch.bfloat16, device="cuda")
probe("vanilla F.linear bf16 (router class)", lambda t: torch.nn.functional.linear(t, w), x)

# 2/3/4. TE Linear
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Float8BlockScaling

    x_t = torch.randn(8192, 2048, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    lin = te.Linear(2048, 2048, bias=False, params_dtype=torch.bfloat16).cuda()
    probe("TE Linear bf16 (no fp8)", lin, x_t)

    try:
        recipe = Float8BlockScaling()  # GLM-5.2 blockwise recipe (TE 2.16.0 name)
    except Exception:
        recipe = DelayedScaling()
    import transformer_engine.pytorch as _tep

    def fp8_call(module, t):
        with _tep.fp8_autocast(enabled=True, fp8_recipe=recipe):
            return module(t)

    lin2 = te.Linear(2048, 2048, bias=False, params_dtype=torch.bfloat16).cuda()
    probe("TE Linear FP8 e4m3", lambda t: fp8_call(lin2, t), x_t)

    stack = torch.nn.ModuleList(
        [te.Linear(2048, 2048, bias=False, params_dtype=torch.bfloat16).cuda() for _ in range(4)]
    )

    def run_stack(t):
        h = t
        for m in stack:
            h = fp8_call(m, h)
        return h

    probe("4-deep TE Linear FP8 stack", run_stack, x_t)
except ImportError as e:
    print(f"transformer_engine not importable here ({e}) — run on the trainer image")

# 5. THE KILL SHOT: the cuDNN DSA attention forward across grad modes.
# Absorbed-MLA shapes: query [sq, b, heads, 576], key [skv, b, 1, 576]
# (kv_lora 512 + rope 64), v_channels 512, topk_indices [b, sq, topk].
try:
    import os
    import sys

    _MC = os.environ.get(
        "BT_TEST_MCORE_PATH",
        os.path.expanduser(
            "~/Documents/trainers/server/vendor/megatron-bridge/3rdparty/Megatron-LM"
        ),
    )
    if os.path.isdir(_MC):
        sys.path.insert(0, _MC)
    from megatron.core.transformer.experimental_attention_variant.dsa_cudnn_kernels import (
        run_fused_absorbed_sparse_attention,
    )

    sq, skv, heads, width, v_channels, topk = 2048, 2048, 64, 576, 512, 128
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(sq, 1, heads, width, generator=g, device="cuda",
                    dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(skv, 1, 1, width, generator=g, device="cuda", dtype=torch.bfloat16)
    topk_indices = torch.randint(0, skv, (1, sq, topk), generator=g, device="cuda")
    topk_length = torch.full((1, sq), topk, dtype=torch.int32, device="cuda")

    def attn(t):
        return run_fused_absorbed_sparse_attention(
            t, k, topk_indices, 1.0, v_channels, topk_length=topk_length
        )

    with torch.no_grad():
        out_a = attn(q)
    with torch.enable_grad():
        out_b = attn(q)
    if out_a is None or out_b is None:
        print("cuDNN DSA attention fwd: backend DECLINED the shapes (returned None) —"
              " check LD_LIBRARY_PATH / backend availability")
    else:
        eq = torch.equal(out_a, out_b)
        md = (out_a - out_b).abs().max().item() if not eq else 0.0
        print(f"{'cuDNN DSA attention fwd (KILL SHOT)':44s} equal={eq}  max|diff|={md:.3e}")
except ImportError as e:
    print(f"mcore DSA kernels not importable ({e}) — run on the trainer image")

# 6. router-flip simulation: how many topk-8 boundary flips from ULP drift?
g = torch.Generator(device="cuda").manual_seed(0)
logits = torch.randn(8192, 256, generator=g, device="cuda")
top_ref = logits.topk(8, dim=1).indices
for eps in (1e-6, 1e-4, 1e-2):
    noisy = logits + eps * torch.randn(logits.shape, generator=g, device="cuda")
    top_noisy = noisy.topk(8, dim=1).indices
    flips = (top_ref.unsqueeze(2) != top_noisy.unsqueeze(1)).all(dim=2).any(dim=1).sum().item()
    print(f"logits + N(0,{eps:g}): rows with >=1 flipped expert = {flips}/8192")

print("\nRead: equal=False on the cuDNN DSA attention row CONFIRMS the"
      " attention-engine grad-mode mechanism (the GEMM rows were all clean"
      " on-box); the flip simulation maps the ULP magnitude to the expected"
      " integer-count mismatch size (single-digit-to-tens flipped rows"
      " predicted for ULP-scale drift).")
