"""Op-level numerics of the expert activation: unfused bf16 chain vs fused SwiGLU vs fp64.

Shapes mimic one expert chunk of GLM-5.3 (tokens x 2*ffn) with per-token probs.
Reports, for the forward output and the input gradient, the error of each path
against an fp64 reference, and the ratio of the two errors. If the unfused bf16
chain (silu, mul, probs-mul, each rounded to bf16) carries more rounding noise
than the fused kernel (fp32 internal math, one final rounding), a lower gradient
norm under the fused kernel is the removed noise, not a lost signal.
"""

import argparse

import torch
import torch.nn.functional as F
from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl


def unfused(x, probs):
    x_glu, x_lin = torch.chunk(x, 2, dim=-1)
    y = F.silu(x_glu) * x_lin
    return (y * probs).to(y.dtype)


def ref64(x, probs):
    x_glu, x_lin = torch.chunk(x.double(), 2, dim=-1)
    return F.silu(x_glu) * x_lin * probs.double()


def err(name, got, ref):
    d = (got.double() - ref).abs()
    rel = d.norm() / ref.norm()
    print(f"  {name:12s} max|d|={d.max().item():.3e} mean|d|={d.mean().item():.3e} rel_l2={rel.item():.3e}")
    return rel.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=65536)
    ap.add_argument("--ffn", type=int, default=2048)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device(args.device)
    x = (torch.randn(args.tokens, 2 * args.ffn, device=dev) * 1.0).to(torch.bfloat16)
    probs = torch.rand(args.tokens, 1, device=dev, dtype=torch.float32) * 0.5
    g = (torch.randn(args.tokens, args.ffn, device=dev) * 1e-2).to(torch.bfloat16)

    print("forward output:")
    ref = ref64(x, probs)
    xu = x.clone().requires_grad_(True)
    pu = probs.clone().requires_grad_(True)
    yu = unfused(xu, pu)
    xf = x.clone().requires_grad_(True)
    pf = probs.clone().requires_grad_(True)
    yf = weighted_bias_swiglu_impl(xf, None, pf, False, None)
    print(f"  dtypes: unfused {yu.dtype} fused {yf.dtype}")
    ru = err("unfused bf16", yu, ref)
    rf = err("fused", yf, ref)
    print(f"  unfused/fused error ratio (rel_l2): {ru / rf:.2f}")

    print("input gradient (d/dx):")
    x64 = x.double().requires_grad_(True)
    p64 = probs.double().requires_grad_(True)
    ref64(x64, p64).backward(g.double())
    yu.backward(g)
    yf.backward(g)
    gu = err("unfused bf16", xu.grad, x64.grad)
    gf = err("fused", xf.grad, x64.grad)
    print(f"  unfused/fused error ratio (rel_l2): {gu / gf:.2f}")
    print(f"  grad norms: fp64 {x64.grad.norm().item():.6e}  unfused {xu.grad.double().norm().item():.6e}  fused {xf.grad.double().norm().item():.6e}")
    print("probs gradient:")
    err("unfused bf16", pu.grad, p64.grad)
    err("fused", pf.grad, p64.grad)


if __name__ == "__main__":
    main()
