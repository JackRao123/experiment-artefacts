#!/usr/bin/env python3
"""EXP2: Direct device-side observation of the fill's split boundary.

Elements N/2-1 and N/2 sit in the SAME 512-element chunk, adjacent threads of
the SAME WARP of the same block if the fill were ONE kernel — their stores
issue in the same instruction window (sub-microsecond skew, typically same
cycle for a vectorized store pair). If the fill is TWO kernels split at N/2,
element N/2-1 is the last element of kernel 1 and N/2 the first of kernel 2:
a device-side spy can then observe a dwell state "N/2-1 filled, N/2 not"
lasting the K1-tail -> K2-wave0 gap (micro-seconds), and never for a single
kernel.

Spy kernel: 1 block on a second stream, volatile-polls both elements +
%globaltimer, logs state transitions. We report dwell time in state
A_filled&&!B_filled and the reverse.
"""
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ long long gt() {
    long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}

__global__ void spy_kernel(const volatile float* buf, long long idxA, long long idxB,
                           float sentinel, int* states, long long* times, int max_n,
                           volatile int* stop, int* n_out) {
    int i = 0;
    while (i < max_n && !(*stop)) {
        float a = buf[idxA];
        float b = buf[idxB];
        int st = ((a != sentinel) ? 1 : 0) | ((b != sentinel) ? 2 : 0);
        states[i] = st;
        times[i] = gt();
        i++;
        if (st == 3) {  // both filled: keep sampling a short tail then exit
            for (int j = 0; j < 64 && i < max_n; j++, i++) { states[i] = 3; times[i] = gt(); }
            break;
        }
    }
    *n_out = i;
}

void launch_spy(torch::Tensor buf_flat, long long idxA, long long idxB, float sentinel,
                torch::Tensor states, torch::Tensor times, torch::Tensor stop,
                torch::Tensor n_out, long long stream_ptr) {
    cudaStream_t s = (cudaStream_t)stream_ptr;
    spy_kernel<<<1, 1, 0, s>>>(
        (const volatile float*)buf_flat.data_ptr<float>(), idxA, idxB, sentinel,
        states.data_ptr<int>(), times.data_ptr<long long>(),
        (int)states.numel(), (volatile int*)stop.data_ptr<int>(), n_out.data_ptr<int>());
}
"""
cpp_src = "void launch_spy(torch::Tensor, long long, long long, float, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, long long);"

mod = load_inline(name="spy", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["launch_spy"], verbose=False)

dev = torch.device("cuda:0")
R, C = 15914, 60192
N = R * C
half = N // 2
SENT = 1.0
print(f"buffer [{R},{C}] N={N:,} half={half:,} = row {half / C}")

buf = torch.empty(R, C, dtype=torch.float32, device=dev)
flat = buf.view(-1)
MAXS = 2_000_000
states = torch.zeros(MAXS, dtype=torch.int32, device=dev)
times = torch.zeros(MAXS, dtype=torch.int64, device=dev)
stop = torch.zeros(1, dtype=torch.int32, device=dev)
n_out = torch.zeros(1, dtype=torch.int32, device=dev)

s_fill = torch.cuda.Stream()
s_spy = torch.cuda.Stream()

import collections
dwellAB = []   # ns dwell in state 1 (A filled, B not)  -> split signature
dwellBA = []   # ns dwell in state 2 (B filled, A not)  -> would REFUTE
TRIALS = 20
for t in range(TRIALS):
    buf.fill_(SENT)
    torch.cuda.synchronize()
    stop.zero_(); n_out.zero_()
    with torch.cuda.stream(s_spy):
        mod.launch_spy(flat, half - 1, half, SENT, states, times, stop, n_out,
                       s_spy.cuda_stream)
    with torch.cuda.stream(s_fill):
        buf.fill_(float("-inf"))
    s_fill.synchronize()
    stop.fill_(1)
    s_spy.synchronize()
    n = int(n_out.item())
    st = states[:n].cpu().numpy()
    tm = times[:n].cpu().numpy()
    # dwell per state
    d = collections.defaultdict(int)
    runs = []
    i = 0
    while i < n:
        j = i
        while j < n and st[j] == st[i]:
            j += 1
        dur = int(tm[min(j, n - 1)] - tm[i])
        d[int(st[i])] += dur
        runs.append((int(st[i]), dur, j - i))
        i = j
    seq = [r[0] for r in runs]
    print(f"trial {t}: samples={n} state_seq={seq} "
          f"dwell(ns) 0={d[0]:,} 1={d[1]:,} 2={d[2]:,} 3={d[3]:,}")
    if d[1]: dwellAB.append(d[1])
    if d[2]: dwellBA.append(d[2])

print(f"\nRESULT: state(A=elem N/2-1 filled, B=elem N/2 not) observed in "
      f"{len(dwellAB)}/{TRIALS} trials, dwell ns: {sorted(dwellAB)}")
print(f"REVERSE state (B filled, A not) observed in {len(dwellBA)}/{TRIALS} "
      f"trials, dwell ns: {sorted(dwellBA)}")
print("Interpretation: sustained state-1 dwell => two kernels split at N/2; "
      "state-2 or neither => refutes split-at-half.")
