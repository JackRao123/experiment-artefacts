#!/usr/bin/env python3
"""On-box numerical parity check for the TF32 fp32-LM-head patch (LPS-1062).

Run inside the server venv on a GPU node:
    python3 test_tf32_head_parity.py

Verifies, for GLM-5.2 head shapes (H=6144, V=154880, chunk=4096), that the
TF32 tensor-core GEMM over bf16-upcast operands matches the FP32 SIMT GEMM to
within fp32 accumulation-order noise, using a float64 reference as ground
truth for both forward logits and dgrad.
"""

import torch

H, V, N = 6144, 154_880, 4096

def main() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(1062)
    dev = "cuda"
    # bf16-valued operands, upcast to fp32 — exactly what _project_logits sees.
    x = torch.randn(N, H, device=dev, dtype=torch.bfloat16).float().requires_grad_(True)
    w = torch.randn(V, H, device=dev, dtype=torch.bfloat16).float()
    g = torch.randn(N, V, device=dev, dtype=torch.bfloat16).float()

    ref64 = (x.double() @ w.double().t())

    torch.backends.cuda.matmul.allow_tf32 = False
    y_fp32 = torch.nn.functional.linear(x, w)
    torch.backends.cuda.matmul.allow_tf32 = True
    y_tf32 = torch.nn.functional.linear(x, w)
    torch.backends.cuda.matmul.allow_tf32 = False

    def err(a, r):
        d = (a.double() - r).abs()
        rel = (d / r.abs().clamp(min=1e-6)).max().item()
        return d.max().item(), rel

    a_fp32, r_fp32 = err(y_fp32, ref64)
    a_tf32, r_tf32 = err(y_tf32, ref64)
    print(f"fwd  abs/rel err vs fp64: fp32-simt {a_fp32:.3e}/{r_fp32:.3e}  tf32 {a_tf32:.3e}/{r_tf32:.3e}")

    # dgrad parity
    dref64 = g.double() @ w.double()
    torch.backends.cuda.matmul.allow_tf32 = False
    d_fp32 = g @ w
    torch.backends.cuda.matmul.allow_tf32 = True
    d_tf32 = g @ w
    torch.backends.cuda.matmul.allow_tf32 = False
    a2_fp32, r2_fp32 = err(d_fp32, dref64)
    a2_tf32, r2_tf32 = err(d_tf32, dref64)
    print(f"dgrad abs/rel err vs fp64: fp32-simt {a2_fp32:.3e}/{r2_fp32:.3e}  tf32 {a2_tf32:.3e}/{r2_tf32:.3e}")

    # TF32 must be no more than ~4x the fp32 SIMT accumulation noise.
    assert a_tf32 <= 4 * max(a_fp32, 1e-6), (a_tf32, a_fp32)
    assert a2_tf32 <= 4 * max(a2_fp32, 1e-6), (a2_tf32, a2_fp32)

    # Timing
    for name, flag in (("fp32-simt", False), ("tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = flag
        for _ in range(3):
            torch.nn.functional.linear(x, w)
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
        t0.record()
        for _ in range(10):
            torch.nn.functional.linear(x, w)
        t1.record(); torch.cuda.synchronize()
        print(f"{name}: {t0.elapsed_time(t1)/10:.2f} ms per [{N}x{H}]@[{V}x{H}]^T")
    torch.backends.cuda.matmul.allow_tf32 = False
    print("PARITY-OK")

if __name__ == "__main__":
    main()
