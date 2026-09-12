"""CPU-only replay of the actual FSDP parameter-swap methods (no trainer/GPU).

Loads method ASTs from the specified checkout, preserving their implementation.
Toy scalar weights isolate metadata cost; this is not a model TPS measurement.
"""
import argparse
import ast
import json
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("core", type=Path)
parser.add_argument("--experts", type=int, default=256)
parser.add_argument("--revision", help="Read committed source without changing the running checkout")
args = parser.parse_args()
source = args.core / "megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py"
source_text = (subprocess.check_output(["git", "-C", str(args.core), "show",
               f"{args.revision}:{source.relative_to(args.core)}"], text=True)
               if args.revision else source.read_text())
tree = ast.parse(source_text)
klass = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "MegatronFSDP")
names = {"_replace_param_with_distributed_if_needed", "_replace_param_with_raw_if_needed",
         "_reestablish_shared_weights", "_replace_cached_parameters"}
methods = [x for x in klass.body if isinstance(x, ast.FunctionDef) and x.name in names]
helper = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "_replace_module_parameter")
ns = {"torch": torch}
exec(compile(ast.Module(body=[helper, *methods], type_ignores=[]), str(source), "exec"), ns)
Replay = type("Replay", (), {name: ns[name] for name in names})
model = torch.nn.Module()
model.decoder = torch.nn.Module()
model.decoder.layers = torch.nn.ModuleList()
for layer in range(75):
    block = torch.nn.Module()
    block.mlp = torch.nn.Module()
    block.mlp.experts = torch.nn.Module()
    for projection in ("linear_fc1", "linear_fc2"):
        linear = torch.nn.Module()
        for expert in range(args.experts):
            linear.register_parameter(f"weight{expert}", torch.nn.Parameter(torch.zeros(1), requires_grad=False))
        setattr(block.mlp.experts, projection, linear)
    model.decoder.layers.append(block)
raw = dict(model.named_parameters())
dist = {name: torch.nn.Parameter(torch.ones(1), requires_grad=False) for name in raw}
# Include a tied weight alias to verify the fast path preserves identity.
alias_owner = model.decoder.layers[-1].mlp.experts.linear_fc2
alias_owner.alias = next(iter(raw.values()))
replay = Replay()
replay.module = model
replay.raw_param = raw
replay.param_and_grad_buffer = SimpleNamespace(optimizer_named_parameters=list(dist.items()))
replay.is_param_fsdp_distributed = False
results = {}
for cached in (False, True):
    replay.ddp_config = SimpleNamespace(fsdp_cache_parameter_metadata=cached)
    measurements = []
    for i in range(7):
        a = time.perf_counter()
        replay._replace_param_with_distributed_if_needed()
        b = time.perf_counter()
        assert alias_owner.alias is next(iter(dist.values()))
        assert all(param is dist[name] for name, param in model.named_parameters())
        c = time.perf_counter()
        replay._replace_param_with_raw_if_needed()
        d = time.perf_counter()
        assert alias_owner.alias is next(iter(raw.values()))
        assert all(param is raw[name] for name, param in model.named_parameters())
        if i >= 2:
            measurements.append({"raw_to_dist_ms":(b-a)*1000, "dist_to_raw_ms":(d-c)*1000})
    results[str(cached)] = {key:statistics.median(x[key] for x in measurements) for key in measurements[0]}
print(json.dumps({"source":str(source), "revision":args.revision, "parameters":len(raw), "expert_layout":args.experts, "median_ms":results}, indent=2))
