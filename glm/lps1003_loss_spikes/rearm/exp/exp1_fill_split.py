#!/usr/bin/env python3
"""EXP1: Does torch fill_ split into two kernels at the real trainer shapes?

Facts to observe: for each shape, the exact list of CUDA kernels launched by
out.fill_(-inf), with grid/block dims, via kineto. No inference.
"""
import torch
from torch.profiler import profile, ProfilerActivity

assert torch.cuda.is_available()
dev = torch.device("cuda:0")
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name} SMs={props.multi_processor_count} torch={torch.__version__}")

SHAPES = [
    ("rank0 (fired, lo=7957)", (15914, 60192)),
    ("rank14 (fired, 0.4% over int32)", (15914, 33858)),
    ("rank15 (immune, 5% under int32)", (15914, 31977)),
    ("synthetic just-over (2^29+1 elems)", (2**29 + 4,)),
    ("synthetic just-under (2^29-4 elems)", (2**29 - 4,)),
]

for name, shape in SHAPES:
    out = torch.empty(*shape, dtype=torch.float32, device=dev)
    nbytes = out.numel() * 4
    out.fill_(0.0)  # warm up allocator/JIT
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out.fill_(float("-inf"))
        torch.cuda.synchronize()
    kernels = [e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
    print(f"\n=== {name} shape={shape} numel={out.numel():,} bytes={nbytes:,} "
          f"(int32max={2**31-1:,}, over={nbytes > 2**31-1})")
    # use raw events for per-launch grid dims
    launches = []
    for evt in prof.events():
        if evt.device_type == torch.autograd.DeviceType.CUDA and "elementwise" in evt.name.lower() or "fill" in evt.name.lower():
            launches.append(evt)
    seen = 0
    for evt in prof.events():
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.name and "memset" not in evt.name.lower():
            seen += 1
            print(f"  CUDA kernel #{seen}: {evt.name[:80]}  dur={evt.device_time:.0f}us")
    # kineto grid dims live in the events' shapes/attributes; also dump table
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
    del out
    torch.cuda.empty_cache()
