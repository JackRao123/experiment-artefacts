import time, torch, collections
from torch.utils.cpp_extension import load_inline
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
__device__ __forceinline__ int64_t gt() {
    int64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__global__ void producer(float* buf, int64_t C, const int64_t* tiles, int T,
                         const int64_t* row_width, int* next, int64_t* t_begin, int64_t* t_store) {
    __shared__ int my;
    while (true) {
        if (threadIdx.x == 0) my = atomicAdd(next, 1);
        __syncthreads();
        int t = my;
        if (t >= T) return;
        int64_t row0 = tiles[t*4], nrows = tiles[t*4+1], spin = tiles[t*4+3];
        if (threadIdx.x == 0) t_begin[t] = gt();
        int64_t t0 = gt();
        while (gt() - t0 < spin) { }
        for (int64_t r = row0; r < row0 + nrows; r++) {
            int64_t w = row_width[r];
            for (int64_t c = threadIdx.x; c < w; c += blockDim.x)
                buf[r*C + c] = (float)r;
        }
        if (threadIdx.x == 0) t_store[t] = gt();
        __syncthreads();
    }
}
__global__ void stamp(int64_t* slot) { *slot = gt(); }
void launch_producer(torch::Tensor buf, torch::Tensor tiles, torch::Tensor row_width,
                     torch::Tensor next, torch::Tensor t_begin, torch::Tensor t_store, int nblocks, int64_t sp) {
    int64_t C = buf.size(1);
    producer<<<nblocks, 384, 0, (cudaStream_t)sp>>>(
        buf.data_ptr<float>(), C, tiles.data_ptr<int64_t>(), (int)tiles.size(0),
        row_width.data_ptr<int64_t>(), next.data_ptr<int>(), t_begin.data_ptr<int64_t>(), t_store.data_ptr<int64_t>());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "producer launch failed");
}
void launch_stamp(torch::Tensor slot, int64_t sp) {
    stamp<<<1,1,0,(cudaStream_t)sp>>>(slot.data_ptr<int64_t>());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "stamp launch failed");
}
"""
cpp_src = "void launch_producer(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int64_t);\nvoid launch_stamp(torch::Tensor, int64_t);"
mod = load_inline(name="destro2", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["launch_producer", "launch_stamp"], verbose=False, extra_cuda_cflags=["-O3"])
dev = torch.device("cuda:0")
NSM = torch.cuda.get_device_properties(0).multi_processor_count
cu_q = [0, 1294, 2588, 3528, 4468, 5121, 5774, 6155, 6536, 8417, 10298, 11381, 12464, 14189, 15914]
offs = [0, 40114, 0, 29140, 0, 20243, 0, 11811, 0, 58311, 0, 33573, 0, 53475]
R, C, RATIO = 15914, 60192, 4
row_width = torch.zeros(R, dtype=torch.int64)
tiles = []
for s in range(14):
    r0, r1, off = cu_q[s], cu_q[s+1], offs[s]
    for l in range(r0, r1):
        row_width[l] = max(1, min(C, (off + (l - r0)) // RATIO + 1))
    t = r0
    while t < r1:
        nr = min(128, r1 - t)
        tiles.append((t, nr, int(row_width[t+nr-1]), int(row_width[t+nr-1]) * 100))
        t += nr
T = len(tiles)
tiles_t = torch.tensor(tiles, dtype=torch.int64, device=dev)
row_width_d = row_width.to(dev)
buf = torch.empty(R, C, dtype=torch.float32, device=dev)
nxt = torch.zeros(1, dtype=torch.int32, device=dev)
t_begin = torch.zeros(T, dtype=torch.int64, device=dev)
t_store = torch.zeros(T, dtype=torch.int64, device=dev)
f0s = torch.zeros(1, dtype=torch.int64, device=dev); f1s = torch.zeros(1, dtype=torch.int64, device=dev)
sA = torch.cuda.Stream(); sB = torch.cuda.Stream()
seg_of = lambda r: max(i for i in range(14) if cu_q[i] <= r)
for trial in range(3):
    buf.fill_(7.0); torch.cuda.synchronize(); nxt.zero_()
    with torch.cuda.stream(sA):
        mod.launch_stamp(f0s, sA.cuda_stream)
        buf.fill_(float("-inf"))
        mod.launch_stamp(f1s, sA.cuda_stream)
    with torch.cuda.stream(sB):
        mod.launch_producer(buf, tiles_t, row_width_d, nxt, t_begin, t_store, NSM, sB.cuda_stream)
    torch.cuda.synchronize()
    t0 = int(f0s.item()); t1 = int(f1s.item())
    col0 = buf[:, 0]
    bad = (torch.isinf(col0) & (col0 < 0)).cpu()
    tb = (t_begin.cpu() - t0) / 1e3; ts = (t_store.cpu() - t0) / 1e3
    print(f"\ntrial {trial}: fill window 0..{(t1-t0)/1e3:.0f}us  (F1 ~0-{(t1-t0)/2e3:.0f}, F2 ~{(t1-t0)/2e3:.0f}-{(t1-t0)/1e3:.0f})")
    persg = collections.defaultdict(lambda: [1e18, -1e18, 0, 0])
    for i, (r0_, nr, wmax, spin) in enumerate(tiles):
        s = seg_of(r0_)
        e = persg[s]
        e[0] = min(e[0], float(ts[i])); e[1] = max(e[1], float(ts[i]))
        e[2] += int(bad[r0_:r0_+nr].sum()); e[3] += nr
    for s in sorted(persg):
        lo_, hi_, nbad, ntot = persg[s]
        print(f"  seg{s:2d} rows[{cu_q[s]},{cu_q[s+1]}) store {lo_:7.0f}..{hi_:7.0f}us bad {nbad}/{ntot}")
