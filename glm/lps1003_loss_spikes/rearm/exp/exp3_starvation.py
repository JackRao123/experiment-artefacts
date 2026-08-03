#!/usr/bin/env python3
"""EXP3: Does a persistent kernel starve a later-launched small kernel on
another stream?  (The claim: F2 could not run until score kernel S finished.)

Persistent kernel P: one CTA per SM, 384 threads (like the DSA kernel),
configurable dynamic smem, spins for ~5ms wall. Fill-like kernel F: many
small 128-thread blocks stamping %globaltimer. Both record per-block
(smid, start_gt, end_gt). F launched on a second stream 200us after P.

Observed fact wanted: min(F.start) vs max(P.end) — did ANY F block execute
before P finished?  Sweep P's dynamic smem 0 / 100KB / 200KB to control
co-residency headroom.
"""
import time
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
__device__ __forceinline__ long long gt() {
    long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ __forceinline__ int smid() {
    int s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s;
}
extern __shared__ char smem[];
__global__ void persistent_spin(long long spin_ns, long long* start, long long* end, int* sm, int use_smem) {
    if (threadIdx.x == 0) { start[blockIdx.x] = gt(); sm[blockIdx.x] = smid(); }
    long long t0 = gt();
    while (gt() - t0 < spin_ns) { }
    if (threadIdx.x == 0) end[blockIdx.x] = gt();
    if (use_smem && threadIdx.x == 0) ((volatile char*)smem)[0] = 1; // keep smem alive
}
__global__ void filler(float* buf, long long n, long long* start, int* sm) {
    if (threadIdx.x == 0) { start[blockIdx.x] = gt(); sm[blockIdx.x] = smid(); }
    long long i = (long long)blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    #pragma unroll
    for (int k = 0; k < 4; k++) if (i + k < n) buf[i + k] = -1.0f/0.0f;
}
void launch_persistent(long long spin_ns, torch::Tensor start, torch::Tensor end,
                       torch::Tensor sm, int nblocks, int smem_bytes, long long sp) {
    if (smem_bytes > 48*1024)
        cudaFuncSetAttribute(persistent_spin, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    persistent_spin<<<nblocks, 384, smem_bytes, (cudaStream_t)sp>>>(
        spin_ns, start.data_ptr<long long>(), end.data_ptr<long long>(), sm.data_ptr<int>(), smem_bytes > 0 ? 1 : 0);
}
void launch_filler(torch::Tensor buf, torch::Tensor start, torch::Tensor sm, long long sp) {
    long long n = buf.numel();
    int nb = (int)((n + 511) / 512);
    filler<<<nb, 128, 0, (cudaStream_t)sp>>>(buf.data_ptr<float>(), n,
        start.data_ptr<long long>(), sm.data_ptr<int>());
}
"""
cpp_src = ("void launch_persistent(long long, torch::Tensor, torch::Tensor, torch::Tensor, int, int, long long);\n"
           "void launch_filler(torch::Tensor, torch::Tensor, torch::Tensor, long long);")
mod = load_inline(name="starv", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["launch_persistent", "launch_filler"], verbose=False)

dev = torch.device("cuda:0")
props = torch.cuda.get_device_properties(0)
NSM = props.multi_processor_count
print(f"GPU {props.name}, {NSM} SMs, smem/SM={props.shared_memory_per_multiprocessor if hasattr(props,'shared_memory_per_multiprocessor') else 'n/a'}")

SPIN_NS = 5_000_000  # 5 ms
buf = torch.empty(200_000_000 // 4, dtype=torch.float32, device=dev)  # ~50MB fill, ~0.05ms
NB_F = (buf.numel() + 511) // 512

sP = torch.cuda.Stream(); sF = torch.cuda.Stream()
for smem_kb in (0, 100, 200):
    p_start = torch.zeros(NSM, dtype=torch.int64, device=dev)
    p_end = torch.zeros(NSM, dtype=torch.int64, device=dev)
    p_sm = torch.zeros(NSM, dtype=torch.int32, device=dev)
    f_start = torch.zeros(NB_F, dtype=torch.int64, device=dev)
    f_sm = torch.zeros(NB_F, dtype=torch.int32, device=dev)
    torch.cuda.synchronize()
    with torch.cuda.stream(sP):
        mod.launch_persistent(SPIN_NS, p_start, p_end, p_sm, NSM, smem_kb * 1024, sP.cuda_stream)
    time.sleep(0.0002)
    with torch.cuda.stream(sF):
        mod.launch_filler(buf, f_start, f_sm, sF.cuda_stream)
    torch.cuda.synchronize()
    ps, pe, fs = p_start.cpu(), p_end.cpu(), f_start.cpu()
    p0, p1 = int(ps.min()), int(pe.max())
    f0, f1 = int(fs.min()), int(fs.max())
    frac_during = float(((fs > p0) & (fs < p1)).float().mean())
    print(f"\nsmem={smem_kb}KB: P window {p0}..{p1} ({(p1-p0)/1e6:.2f}ms)  "
          f"F starts {f0}..{f1}")
    print(f"  F blocks starting DURING P: {frac_during*100:.1f}%   "
          f"F first block vs P end: {'BEFORE (co-ran)' if f0 < p1 else 'AFTER (starved)'} "
          f"delta={(f0-p1)/1e3:.1f}us")
