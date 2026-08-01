#!/usr/bin/env python3
"""Fill (nearly) all free memory on every local GPU with 0xFF bytes (fp16 NaN
pattern / huge garbage), sync, release, exit. Run on EACH node before booting
the trainer to simulate landing on a node whose memory holds hostile garbage
(LPS-1003 uninit-read repro).
"""
import torch

for d in range(torch.cuda.device_count()):
    torch.cuda.set_device(d)
    free, total = torch.cuda.mem_get_info(d)
    blocks = []
    remaining = free
    # leave ~2 GiB headroom; allocate in 4 GiB blocks to avoid fragmentation stalls
    budget = remaining - (2 << 30)
    block = 4 << 30
    while budget > 0:
        n = min(block, budget)
        try:
            t = torch.empty(n, dtype=torch.uint8, device=f"cuda:{d}")
            t.fill_(0xFF)
            blocks.append(t)
            budget -= n
        except torch.cuda.OutOfMemoryError:
            block //= 2
            if block < (256 << 20):
                break
    torch.cuda.synchronize(d)
    filled = sum(b.numel() for b in blocks)
    print(f"cuda:{d}: poisoned {filled / (1 << 30):.1f} GiB with 0xFF", flush=True)
    del blocks
    torch.cuda.empty_cache()
print("POISON COMPLETE", flush=True)
