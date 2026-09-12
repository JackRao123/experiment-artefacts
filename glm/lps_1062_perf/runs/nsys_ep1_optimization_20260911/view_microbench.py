"""CPU-only persistent-view binding benchmark using exact committed FSDP methods.

Scalar-sized toy buffers verify view/slot behavior without loading the trainer.
No inference about GPU GEMM speed or full-model TPS is made.
"""
import argparse
import ast
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace
from typing import Optional

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("core", type=Path)
parser.add_argument("--revision", required=True)
args = parser.parse_args()
relative = "megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py"
source = subprocess.check_output(["git", "-C", str(args.core), "show", f"{args.revision}:{relative}"], text=True)
tree = ast.parse(source)
klass = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "DataParallelBuffer")
names = {"fetch_bucket", "get_item_from_bucket"}
methods = [x for x in klass.body if isinstance(x, ast.FunctionDef) and x.name in names]


@dataclass
class Bucket:
    data: torch.Tensor


class FixedPoolAllocator:
    fsdp_param_groups = [SimpleNamespace(fsdp_unit_id=0)]
    fsdp_double_buffer_units = [0]


ns = {"torch":torch, "Optional":Optional, "Bucket":Bucket, "FixedPoolAllocator":FixedPoolAllocator,
      "MaxPoolAllocator":FixedPoolAllocator, "to_local_if_dtensor":lambda p:p,
      "is_float8tensor":lambda p:False}
exec(compile(ast.Module(body=methods, type_ignores=[]), relative, "exec"), ns)
Replay = type("Replay", (), {name:ns[name] for name in names})
result = {}
for count in (256, 768):
    r = Replay()
    r.dtype = torch.bfloat16
    r.device = torch.device("cpu")
    r.is_data_distributed = True
    r.bucket_id = 0
    r.temporary_bucket_allocator = FixedPoolAllocator()
    r.params = [torch.nn.Parameter(torch.zeros(4,4,dtype=r.dtype),requires_grad=False) for _ in range(count)]
    r.param_idx = {p:i for i,p in enumerate(r.params)}
    r.item_index_map = {i:SimpleNamespace(global_data_index=i*16,size=16) for i in range(count)}
    r.bucket_index = SimpleNamespace(global_data_index=0, size=count*16)
    buffers = [Bucket(torch.arange(count*16,dtype=torch.float32).to(r.dtype)), Bucket(torch.full((count*16,),3,dtype=r.dtype))]
    r.slot = 0
    r.allocate_bucket_storage = lambda **kwargs: buffers[r.slot]
    timings = {}
    for enabled in (False, True):
        r.ddp_config = SimpleNamespace(fsdp_cache_parameter_metadata=enabled)
        r._cached_param_views = None
        r._cached_param_views_key = None
        times = []
        for i in range(30):
            t = time.perf_counter()
            r.fetch_bucket(set_param_data=True)
            times.append((time.perf_counter()-t)*1000)
            assert all(torch.equal(p, buffers[r.slot].data[j*16:(j+1)*16].view(4,4)) for j,p in enumerate(r.params))
        timings[str(enabled)] = statistics.median(times[5:])
        # Pool-slot reassignment must rebuild every parameter view.
        r.slot = 1
        r.fetch_bucket(set_param_data=True)
        assert all(torch.equal(p, torch.full_like(p,3)) for p in r.params)
        r.slot = 0
        r.fetch_bucket(set_param_data=True)
        assert all(torch.equal(p,buffers[0].data[j*16:(j+1)*16].view(4,4)) for j,p in enumerate(r.params))
    result[count] = timings
print(json.dumps({"source_revision":args.revision,"median_binding_ms":result,"slot_reassignment_verified":True},indent=2))
