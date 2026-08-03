#!/usr/bin/env python3
"""EXP4: Reproduce the LPS-1003 destroyer/partial mask family with LEGAL
unsynchronized streams, using rank0's real geometry.

Producer = persistent kernel, one CTA/SM work-queue over 128-row m-tiles of
the real segments; per-tile spin proportional to real causal key width; then
stores rowid to cols [0, w(row)). This mimics the DSA indexer's decomposition
(CLC persistent, per-tile compute time ~ key width, stores at tile end).

Fill = the actual torch fill_(-inf) on the [15914, 60192] fp32 buffer (which
torch splits into two kernels F1/F2 at row 7957 — verified in exp1/2).

Streams: fill enqueued on stream A, producer on stream B, host delay swept.
No cross-stream sync — the legal-stream equivalent of the prod reorder.

Predictions if the theory is right:
  - some trials: rows [7957, 15913] ALL -inf (destroyer, lo exactly 7957)
  - some trials: partial masks with lo > 7957 concentrated where the
    longest tiles store last
  - some trials clean
  - lo NEVER < 7957 when only F2 loses; full wipe only if both fills lose
  - control (same stream / event-synced): always clean
"""
import time, sys
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
__device__ __forceinline__ long long gt() {
    long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
// tiles: [T,4] int64 = (row0, nrows, width_max, spin_ns)
__global__ void producer(float* buf, long long C, const long long* tiles, int T,
                         const long long* row_width, int* next, long long* t_store) {
    __shared__ int my;
    while (true) {
        if (threadIdx.x == 0) my = atomicAdd(next, 1);
        __syncthreads();
        int t = my;
        if (t >= T) return;
        long long row0 = tiles[t*4], nrows = tiles[t*4+1], spin = tiles[t*4+3];
        long long t0 = gt();
        while (gt() - t0 < spin) { }             // "compute"
        // store phase: rows of this tile, value = row id
        for (long long r = row0; r < row0 + nrows; r++) {
            long long w = row_width[r];
            for (long long c = threadIdx.x; c < w; c += blockDim.x)
                buf[r*C + c] = (float)r;
        }
        if (threadIdx.x == 0) t_store[t] = gt();
        __syncthreads();
    }
}
void launch_producer(torch::Tensor buf, torch::Tensor tiles, torch::Tensor row_width,
                     torch::Tensor next, torch::Tensor t_store, int nblocks, long long sp) {
    long long C = buf.size(1);
    producer<<<nblocks, 384, 0, (cudaStream_t)sp>>>(
        buf.data_ptr<float>(), C, tiles.data_ptr<long long>(), (int)tiles.size(0),
        row_width.data_ptr<long long>(), next.data_ptr<int>(), t_store.data_ptr<long long>());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "producer launch failed");
}
"""
cpp_src = "void launch_producer(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, long long);"
mod = load_inline(name="destro", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["launch_producer"], verbose=False,
                  extra_cuda_cflags=["-O3"])

dev = torch.device("cuda:0")
props = torch.cuda.get_device_properties(0)
NSM = props.multi_processor_count

# real rank0 geometry
cu_q = [0, 1294, 2588, 3528, 4468, 5121, 5774, 6155, 6536, 8417, 10298, 11381, 12464, 14189, 15914]
offs = [0, 40114, 0, 29140, 0, 20243, 0, 11811, 0, 58311, 0, 33573, 0, 53475]
R, C, RATIO = 15914, 60192, 4
HALF_ROW = (R * C // 2) // C
assert HALF_ROW == 7957

row_width = torch.zeros(R, dtype=torch.int64)
tiles = []
NS_PER_KEYCOL = 100.0  # ns per key column: longest tile (60192 cols) ~ 6ms   # scale: longest tile ~ (60192 cols)*128rows -> aim total ~6ms
for s in range(14):
    r0, r1, off = cu_q[s], cu_q[s + 1], offs[s]
    for l in range(r0, r1):
        row_width[l] = max(1, min(C, (off + (l - r0)) // RATIO + 1))
    t = r0
    while t < r1:
        nr = min(128, r1 - t)
        wmax = int(row_width[t + nr - 1])
        spin_ns = int(wmax * NS_PER_KEYCOL)  # ~ proportional to tile key width
        tiles.append((t, nr, wmax, spin_ns))
        t += nr
tiles_t = torch.tensor(tiles, dtype=torch.int64, device=dev)
row_width_d = row_width.to(dev)
total_spin = sum(t[3] for t in tiles) / NSM * 4  # rough serial estimate
print(f"{len(tiles)} tiles, {NSM} SMs, est producer ms={sum(t[3] for t in tiles)/max(1,len(tiles))*len(tiles)/NSM/1e6:.1f}")

buf = torch.empty(R, C, dtype=torch.float32, device=dev)
nxt = torch.zeros(1, dtype=torch.int32, device=dev)
t_store = torch.zeros(len(tiles), dtype=torch.int64, device=dev)
sA = torch.cuda.Stream(); sB = torch.cuda.Stream()

def run_trial(delay_us, ordered):
    buf.fill_(7.0)  # neutral pre-state
    torch.cuda.synchronize()
    nxt.zero_()
    with torch.cuda.stream(sA):
        buf.fill_(float("-inf"))
    if ordered:
        sB.wait_stream(sA)
    elif delay_us:
        time.sleep(delay_us / 1e6)
    with torch.cuda.stream(sB):
        mod.launch_producer(buf, tiles_t, row_width_d, nxt, t_store, NSM, sB.cuda_stream)
    torch.cuda.synchronize()
    col0 = buf[:, 0]
    bad = torch.isinf(col0) & (col0 < 0)
    idx = bad.nonzero().flatten()
    if len(idx) == 0:
        return None
    return int(idx[0]), int(idx[-1]), int(len(idx))

import collections
print("\n--- CONTROL (event-ordered): 10 trials")
ctrl = [run_trial(0, True) for _ in range(10)]
print("bad masks:", [c for c in ctrl if c] or "none (all clean)")

print("\n--- RACE (no sync), sweep host delay")
fam = collections.Counter()
examples = collections.defaultdict(list)
for delay in (0, 50, 100, 200, 400, 700, 1000, 1500):
    for _ in range(12):
        r = run_trial(delay, False)
        if r is None:
            fam["clean"] += 1
        else:
            lo, hi, n = r
            if lo == 7957 and n >= (R - 7957) - 64:
                k = "DESTROYER lo=7957"
            elif lo >= 7957:
                k = "partial (lo>7957)"
            elif lo < 7957 and n > R - 100:
                k = "full wipe"
            else:
                k = "OTHER lo<7957"
            fam[k] += 1
            if len(examples[k]) < 6:
                examples[k].append((delay, lo, hi, n))
print(dict(fam))
for k, v in examples.items():
    print(f"  {k}: (delay_us, lo, hi, n) = {v}")
