"""Standalone probe for fix 3: LM-head GEMM numerics and speed on one B300.

Compares, for bf16-valued inputs (hidden [4096, 6144] and a vocab shard of the
head weight [V, 6144]):
  fp32   : hidden.float() @ W.float().T             (what chunked_lm_head.py does today;
                                                     lands in cutlass3x_sm100_simt_sgemm)
  bf16o32: torch.mm(hidden, W.T, out_dtype=float32)  (bf16 tensor cores, fp32 accumulate,
                                                     fp32 output, no output rounding)
  bf16   : hidden @ W.T -> bf16                      (the plain module path, output rounded to bf16;
                                                     shown only to calibrate the error scale)
and the backward GEMM grad_hidden = grad_logits(fp32) @ W:
  fp32        : SIMT fp32
  bf16 single : mm(grad.bfloat16(), W, out_dtype=fp32)             (grad rounded to bf16)
  bf16 split  : mm(hi, W, o32) + mm(lo, W, o32), grad = hi + lo    (near-fp32 grad)
Reports max/mean abs and relative error vs an fp64 reference, and per-call time.
"""

import argparse
import time

import torch


def bench(fn, iters=10):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


def err(name, got, ref):
    d = (got.double() - ref).abs()
    rel = (d / ref.abs().clamp_min(1e-6)).max().item()
    print(
        f"  {name:14s} max|d|={d.max().item():.3e} mean|d|={d.mean().item():.3e} "
        f"max_rel={rel:.3e}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=6144)
    ap.add_argument("--vocab", type=int, default=151552)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device(args.device)
    # bf16-valued inputs, as in the trainer (hidden and weight are bf16 tensors).
    h = (torch.randn(args.rows, args.hidden, device=dev) * 0.5).to(torch.bfloat16)
    w = (torch.randn(args.vocab, args.hidden, device=dev) * 0.02).to(torch.bfloat16)
    g = torch.randn(args.rows, args.vocab, device=dev, dtype=torch.float32) * 1e-3

    print(f"torch {torch.__version__} allow_tf32={torch.backends.cuda.matmul.allow_tf32}")
    print(f"shapes: hidden {tuple(h.shape)} weight {tuple(w.shape)}")

    # ---- forward: logits = h @ w.T
    ref = h.double() @ w.double().T
    f32 = lambda: h.float() @ w.float().T  # noqa: E731
    b16o32 = lambda: torch.mm(h, w.T, out_dtype=torch.float32)  # noqa: E731
    b16 = lambda: (h @ w.T)  # noqa: E731
    print("forward numerics vs fp64:")
    err("fp32(simt)", f32(), ref)
    err("bf16->fp32out", b16o32(), ref)
    err("bf16 out", b16(), ref)
    d = (f32().double() - b16o32().double()).abs()
    print(f"  fp32 vs bf16o32 directly: max|d|={d.max().item():.3e} mean|d|={d.mean().item():.3e}")
    print("forward time per call (ms):")
    print(f"  fp32(simt)     {bench(f32):8.2f}   (+ upcast of W each call: {bench(lambda: w.float()):.2f})")
    print(f"  bf16->fp32out  {bench(b16o32):8.2f}")
    print(f"  bf16 out       {bench(b16):8.2f}")

    # ---- backward: grad_h = g @ w  ([rows, V] @ [V, H])
    refb = g.double() @ w.double()
    f32b = lambda: g @ w.float()  # noqa: E731
    single = lambda: torch.mm(g.to(torch.bfloat16), w, out_dtype=torch.float32)  # noqa: E731

    def split():
        hi = g.to(torch.bfloat16)
        lo = (g - hi.float()).to(torch.bfloat16)
        return torch.mm(hi, w, out_dtype=torch.float32) + torch.mm(lo, w, out_dtype=torch.float32)

    print("backward numerics vs fp64 (grad_hidden, fp32 before the trainer casts it to bf16):")
    err("fp32(simt)", f32b(), refb)
    err("bf16 single", single(), refb)
    err("bf16 split", split(), refb)
    # what the trainer actually consumes: bf16(grad_hidden)
    print("backward numerics after the bf16 cast the trainer applies (grad flows into bf16 hidden):")
    err("fp32(simt)", f32b().to(torch.bfloat16), refb)
    err("bf16 single", single().to(torch.bfloat16), refb)
    err("bf16 split", split().to(torch.bfloat16), refb)
    same = (f32b().to(torch.bfloat16) == split().to(torch.bfloat16)).double().mean().item()
    same1 = (f32b().to(torch.bfloat16) == single().to(torch.bfloat16)).double().mean().item()
    print(f"  fraction of bf16 grad elements bitwise equal to fp32 path: split={same:.4f} single={same1:.4f}")
    print("backward time per call (ms):")
    print(f"  fp32(simt)     {bench(f32b):8.2f}")
    print(f"  bf16 single    {bench(single):8.2f}")
    print(f"  bf16 split     {bench(split):8.2f}")


if __name__ == "__main__":
    main()
