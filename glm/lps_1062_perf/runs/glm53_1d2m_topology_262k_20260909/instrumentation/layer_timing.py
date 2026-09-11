"""CUDA-event timing of whole checkpoint blocks; no inter-block synchronization."""
import functools
import inspect
import json
import os
from pathlib import Path
import torch

def install():
    from megatron.core.tensor_parallel.random import CheckpointFunction
    from trainers_server_megatron_bridge.backend import MegatronBridgeBackend
    original_forward = CheckpointFunction.forward
    original_backward = CheckpointFunction.backward
    original_fb = MegatronBridgeBackend.forward_backward
    pending = []
    state = {"step":0}
    def event():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e
    @functools.wraps(original_forward)
    def forward(ctx, fn, distribute, *args):
        closure = inspect.getclosurevars(fn).nonlocals
        if fn.__name__ != "custom_forward" or "start" not in closure:
            return original_forward(ctx, fn, distribute, *args)
        layer = int(closure["start"])
        assert int(closure["end"]) == layer+1
        row = {"step":state["step"],"layer":layer,"rank":int(os.environ["RANK"])}
        ctx._timing_row = row
        row["f0"] = event()
        with torch.profiler.record_function(f"block_{layer}/forward"):
            result = original_forward(ctx, fn, distribute, *args)
        row["f1"] = event()
        pending.append(row)
        return result
    @functools.wraps(original_backward)
    def backward(ctx, *args):
        row = getattr(ctx, "_timing_row", None)
        if row is None: return original_backward(ctx,*args)
        fn = ctx.run_function
        def recompute(*values):
            row["r0"] = event()
            with torch.profiler.record_function(f"block_{row['layer']}/recompute"):
                output = fn(*values)
            row["r1"] = event()
            return output
        ctx.run_function = recompute
        row["b0"] = event()
        with torch.profiler.record_function(f"block_{row['layer']}/backward_including_recompute"):
            result = original_backward(ctx,*args)
        row["b1"] = event()
        ctx.run_function = fn
        return result
    @functools.wraps(original_fb)
    def fb(self, *args, **kwargs):
        state["step"] += 1
        result = original_fb(self,*args,**kwargs)
        if pending:
            torch.cuda.synchronize()
            rows = []
            for row in pending:
                out = {k:row[k] for k in ("step","layer","rank")}
                for label,start,end in (("forward_ms","f0","f1"),("recompute_ms","r0","r1"),("backward_including_recompute_ms","b0","b1"),("backward_after_recompute_ms","r1","b1")):
                    out[label] = row[start].elapsed_time(row[end])
                out["backward_excluding_recompute_ms"] = out["backward_including_recompute_ms"]-out["recompute_ms"]
                rows.append(out)
            folder = Path(os.environ["LAYER_TIMING_DIR"])
            folder.mkdir(parents=True,exist_ok=True)
            with (folder/f"rank{os.environ['RANK']}.jsonl").open("a") as f:
                for row in rows: f.write(json.dumps(row)+"\n")
            pending.clear()
        return result
    CheckpointFunction.forward = staticmethod(forward)
    CheckpointFunction.backward = staticmethod(backward)
    MegatronBridgeBackend.forward_backward = fb
