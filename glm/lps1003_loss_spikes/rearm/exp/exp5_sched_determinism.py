#!/usr/bin/env python3
"""EXP5: How deterministic is hardware block scheduling for a fill-sized grid?

Kernel: every block records (smid, start %globaltimer). Grid = the real F2
size (~935k blocks of 128 threads). Repeat 15x on an idle GPU; measure:
  - dispatch-order stability: Spearman corr of block start-times across runs
  - wave structure: resident block count (blocks whose [start, start+dur]
    overlap), wave width in blocks and in buffer rows
  - start-time monotonicity vs blockIdx (do low blocks really start first?)
"""
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
__device__ __forceinline__ long long gt() {
    long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ __forceinline__ int smid() {
    int s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s;
}
__global__ void stamper(long long* start, int* sm, float* buf, long long n) {
    long long b = blockIdx.x;
    if (threadIdx.x == 0) { start[b] = gt(); sm[b] = smid(); }
    long long i = b * (long long)blockDim.x * 4 + threadIdx.x * 4;
    #pragma unroll
    for (int k = 0; k < 4; k++) if (i + k < n) buf[i + k] = -1.0f/0.0f;
}
void launch_stamper(torch::Tensor start, torch::Tensor sm, torch::Tensor buf, long long sp) {
    long long n = buf.numel();
    long long nb = (n + 511) / 512;
    stamper<<<(int)nb, 128, 0, (cudaStream_t)sp>>>(
        start.data_ptr<long long>(), sm.data_ptr<int>(), buf.data_ptr<float>(), n);
}
"""
cpp = "void launch_stamper(torch::Tensor, torch::Tensor, torch::Tensor, long long);"
mod = load_inline(name="sched5", cpp_sources=cpp, cuda_sources=cuda_src,
                  functions=["launch_stamper"], verbose=False, extra_cuda_cflags=["-O3"])

dev = torch.device("cuda:0")
props = torch.cuda.get_device_properties(0)
N = 15914 * 60192 // 2           # F2-sized: 478,947,744 elems
NB = (N + 511) // 512
print(f"GPU {props.name} SMs={props.multi_processor_count} grid={NB:,} blocks")

buf = torch.empty(N, dtype=torch.float32, device=dev)
start = torch.zeros(NB, dtype=torch.int64, device=dev)
sm = torch.zeros(NB, dtype=torch.int32, device=dev)
s = torch.cuda.Stream()

runs = []
for r in range(8):
    torch.cuda.synchronize()
    with torch.cuda.stream(s):
        mod.launch_stamper(start, sm, buf, s.cuda_stream)
    torch.cuda.synchronize()
    runs.append(start.cpu().clone())

import numpy as np
r0 = runs[0].numpy().astype(np.float64)
o0 = np.argsort(np.argsort(r0))
print("\ndispatch-order stability vs run0 (Spearman rho of start times):")
for i, rr in enumerate(runs[1:], 1):
    ri = rr.numpy().astype(np.float64)
    oi = np.argsort(np.argsort(ri))
    rho = np.corrcoef(o0, oi)[0, 1]
    print(f"  run{i}: rho={rho:.6f}")

t = r0 - r0.min()
q = np.quantile
print(f"\nrun0: duration {t.max()/1e6:.3f} ms")
# monotonicity: start time vs blockIdx
bi = np.arange(NB, dtype=np.float64)
rho_mono = np.corrcoef(bi, t)[0, 1]
print(f"start-time vs blockIdx Pearson r = {rho_mono:.6f}  (1.0 = perfectly ascending)")
# wave width: how many blocks start within the same microsecond
order = np.argsort(t)
ts = t[order]
for w_us in (1, 5, 20):
    import collections
    c = collections.Counter((ts // (w_us * 1000)).astype(np.int64))
    sizes = np.array(sorted(c.values()))
    print(f"blocks starting per {w_us}us bucket: median={int(np.median(sizes))} p90={int(np.quantile(sizes,0.9))}")
# highest-address block start vs lowest
print(f"first block start={t[0]/1e3:.1f}us last block start={t[-1]/1e3:.1f}us "
      f"(block 0 vs block {NB-1})")
frontier_spread = np.abs(t[order][:, None]) # placeholder
# how far can the 'filled frontier' be non-monotone: max over addresses of (start of any later block - start of this block)
rev = np.maximum.accumulate(t[::-1])[::-1]  # max start time at >= this block
lead = rev - t  # how much later some higher block starts
print(f"non-monotonicity: p50={np.median(lead)/1e3:.1f}us p99={np.quantile(lead,0.99)/1e3:.1f}us max={lead.max()/1e3:.1f}us")
