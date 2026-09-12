"""Replay the exact FSDP orchestration methods with bucket-state recorders.

Checks deduplicated wait/release order and per-parameter FP8 processing order.
No CUDA or trainer required; not a kernel or throughput benchmark.
"""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

source = Path(sys.argv[1]) / "megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py"
tree = ast.parse(source.read_text())
names = ("all_gather_and_wait_parameters_ready", "release_module_parameters")
functions = [next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name) for name in names]


class Param:
    def __init__(self, idx):
        self.idx = idx


class Pipeline:
    def __init__(self):
        self.waits, self.releases, self.events = [], [], []
        self.ready, self.released = set(), set()

    def all_gather_params(self, **kwargs):
        pass

    def wait_bucket_ready(self, bucket, bwd):
        self.waits.append((bucket,bwd))
        if bucket not in self.ready:
            self.events.append(("ready",bucket,bwd))
        self.ready.add(bucket)

    def release_bucket(self, bucket, bwd, lazy):
        self.releases.append((bucket,bwd,lazy))
        if bucket not in self.released:
            self.events.append(("release",bucket,bwd,lazy))
        self.released.add(bucket)


params = [Param(i) for i in range(768)]
bucket_by_param = {p: (p.idx%3) for p in params}
module = SimpleNamespace(parameters=lambda:iter(params))
output = {}
for bwd in (False, True):
    for lazy in (False, True):
        variants = []
        for cached in (False, True):
            pipe = Pipeline()
            model = SimpleNamespace(
                all_gather_pipeline=pipe,
                param_and_grad_buffer=SimpleNamespace(param_to_param_group=bucket_by_param),
                data_parallel_sharding_strategy="optim_grads_params",
                ddp_config=SimpleNamespace(fsdp_cache_parameter_metadata=cached,
                    fsdp_param_gather_prefetch=True, keep_fp8_transpose_cache=False),
                dist_index=SimpleNamespace(use_hybrid_fsdp=False), suggested_AG_prefetch_size=100,
            )
            ns = {"PrefetchOrder":SimpleNamespace(FORWARD_PASS_ORDER=1), "self":model,
                  "is_float8tensor":lambda p: p.idx%7==0,
                  "fp8_create_transpose_cache":lambda p:pipe.events.append(("fp8",p.idx)),
                  "release_params_fp8_transpose_cache":lambda ps:pipe.events.append(("drop_fp8",tuple(p.idx for p in ps)))}
            exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),"exec"),ns)
            ns[names[0]](model,params,bwd=bwd)
            ns[names[1]](module,bwd,lazy)
            variants.append(pipe)
        assert variants[0].events == variants[1].events
        output[f"bwd={bwd},lazy={lazy}"] = {
            "uncached_wait_calls":len(variants[0].waits), "cached_wait_calls":len(variants[1].waits),
            "uncached_release_calls":len(variants[0].releases), "cached_release_calls":len(variants[1].releases),
            "state_transitions_and_fp8_order_match":True}
print(json.dumps(output,indent=2))
