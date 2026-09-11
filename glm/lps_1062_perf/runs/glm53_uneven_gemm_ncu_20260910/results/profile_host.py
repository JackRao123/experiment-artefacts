"""Locate host-side setup work; cProfile timings are not performance benchmarks."""
import cProfile
import csv
import pstats
from pathlib import Path

import torch
from expert_gemm import ExpertGEMMPair

root = Path(__file__).parent
torch.set_num_threads(1)
torch.manual_seed(328)
with (root / "counts.csv").open() as stream:
    rows = sorted((r for r in csv.DictReader(stream) if r["config"] == "CP8EP1" and r["rank"] == "0"
                   and r["moe_layer"] == "1" and r["stage"] == "expert_input"), key=lambda r: int(r["expert"]))
pair = ExpertGEMMPair([int(r["tokens"]) for r in rows], 6144, 4096)
x = torch.randn(131072, 6144, device="cuda", dtype=torch.bfloat16, requires_grad=True)
for name, function in (("te", pair.te_forward), ("grouped", pair.grouped_forward)):
    for _ in range(5):
        y = function(x)
        torch.cuda.synchronize()
    profiler = cProfile.Profile()
    for _ in range(20):
        profiler.enable()
        y = function(x)
        profiler.disable()
        torch.cuda.synchronize()
    profiler.dump_stats(root / f"host-{name}.pstats")
    with (root / f"host-{name}.txt").open("w") as stream:
        pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(30)
        pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(20)
    print(name, "host profile saved", flush=True)
