"""Experiment-only CUDA-event sublayer timers and cyclic-GC observations."""
import functools
import gc
import inspect
import json
import os
import time
import traceback
from pathlib import Path
import torch

def install():
    from megatron.core.tensor_parallel.random import CheckpointFunction
    from megatron.core.transformer.transformer_layer import TransformerLayer
    from trainers_server_megatron_bridge.backend import MegatronBridgeBackend
    original_f=CheckpointFunction.forward
    original_b=CheckpointFunction.backward
    original_fb=MegatronBridgeBackend.forward_backward
    state={"step":0,"row":None,"phase":"forward","in_fb":False,"freeze_owner":False,"frozen_count":gc.get_freeze_count()}
    pending=[]
    gc_events=[]
    gc_active={}
    folder=Path(os.environ["LAYER_TIMING_DIR"])
    rank=int(os.environ["RANK"])
    def event():
        e=torch.cuda.Event(enable_timing=True);e.record();return e
    def gc_callback(phase, info):
        if phase=="start":
            annotation=torch.profiler.record_function(f"python_gc/generation_{info['generation']}")
            annotation.__enter__()
            gc_active.update(start=time.perf_counter(),annotation=annotation,generation=info["generation"],step=state["step"],in_fb=state["in_fb"],frozen=state['frozen_count'],enabled=gc.isenabled(),stack=traceback.format_stack(limit=12) if info['generation']==2 else [])
        elif phase=="stop" and gc_active:
            elapsed=(time.perf_counter()-gc_active["start"])*1000
            gc_active["annotation"].__exit__(None,None,None)
            gc_events.append(dict(step=gc_active["step"],rank=rank,generation=gc_active["generation"],duration_ms=elapsed,collected=info["collected"],uncollectable=info["uncollectable"],in_fb=gc_active['in_fb'],frozen_count=gc_active['frozen'],gc_enabled=gc_active['enabled'],stack=gc_active['stack']))
            gc_active.clear()
    gc.callbacks.append(gc_callback)
    def wrap_part(name, original):
        @functools.wraps(original)
        def part(self,*args,**kwargs):
            row=state["row"]
            if row is None:return original(self,*args,**kwargs)
            phase=state["phase"];prefix=f"{phase}_{name}"
            row[prefix+"0"]=event()
            with torch.profiler.record_function(f"layer_{row['layer']}/{phase}/{name}"):
                result=original(self,*args,**kwargs)
            row[prefix+"1"]=event()
            value=result[0] if isinstance(result,tuple) else result
            if phase=="recompute" and torch.is_tensor(value) and value.requires_grad:
                def hook(grad):
                    row[name+"_bwd_start"]=event()
                    return grad
                value.register_hook(hook)
            return result
        return part
    TransformerLayer._forward_attention=wrap_part("attention",TransformerLayer._forward_attention)
    TransformerLayer._forward_mlp=wrap_part("mlp",TransformerLayer._forward_mlp)
    @functools.wraps(original_f)
    def forward(ctx,fn,distribute,*args):
        closure=inspect.getclosurevars(fn).nonlocals
        if fn.__name__!="custom_forward" or "start" not in closure:return original_f(ctx,fn,distribute,*args)
        row={"step":state["step"],"rank":rank,"layer":int(closure["start"])}
        assert int(closure["end"])==row["layer"]+1
        ctx._timing_row=row;row["f0"]=event()
        state.update(row=row,phase="forward")
        try:
            with torch.profiler.record_function(f"block_{row['layer']}/forward"):
                result=original_f(ctx,fn,distribute,*args)
        finally:state["row"]=None
        row["f1"]=event();pending.append(row)
        return result
    @functools.wraps(original_b)
    def backward(ctx,*args):
        row=getattr(ctx,"_timing_row",None)
        if row is None:return original_b(ctx,*args)
        fn=ctx.run_function
        def recompute(*values):
            row["r0"]=event();state.update(row=row,phase="recompute")
            try:
                with torch.profiler.record_function(f"block_{row['layer']}/recompute"):result=fn(*values)
            finally:state["row"]=None
            row["r1"]=event();return result
        ctx.run_function=recompute;row["b0"]=event()
        with torch.profiler.record_function(f"block_{row['layer']}/backward_including_recompute"):result=original_b(ctx,*args)
        row["b1"]=event();ctx.run_function=fn
        return result
    @functools.wraps(original_fb)
    def fb(self,*args,**kwargs):
        if state['step']==0:
            inventory=[]
            for model in self._model_list:
                for name,module in model.named_modules():
                    if name.endswith(('.mlp.experts.linear_fc1','.mlp.experts.linear_fc2')):
                        for i in range(module.num_gemms):
                            weight=getattr(module,f'weight{i}')
                            inventory.append(dict(module=name,expert=i,type=type(weight).__name__,dtype=str(weight.dtype),shape=list(weight.shape),quantized_payload=hasattr(weight,'_rowwise_data')))
            folder.mkdir(parents=True,exist_ok=True)
            (folder/f'weights.rank{rank}.json').write_text(json.dumps(inventory,indent=2)+'\n')
            if not inventory or any(w['quantized_payload'] or w['dtype']!='torch.bfloat16' for w in inventory):
                raise RuntimeError('Expected resident BF16 expert weights')
        owner=getattr(self,"_owns_gc_freeze",False)
        if owner!=state['freeze_owner']:
            # get_freeze_count walks the permanent generation; never call it
            # from each GC callback or layer measurement after freezing millions
            # of model/runtime objects. Cache this activation diagnostic once.
            state['frozen_count']=gc.get_freeze_count()
            state['freeze_owner']=owner
        state["step"]+=1;state["in_fb"]=True
        try:result=original_fb(self,*args,**kwargs)
        finally:state["in_fb"]=False
        torch.cuda.synchronize()
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/f"rank{rank}.jsonl").open("a") as f:
            for row in pending:
                out={k:row[k] for k in ("step","rank","layer")}
                out["gc_frozen_count_at_activation"]=state['frozen_count']
                out["gc_enabled"]=gc.isenabled()
                out["owns_gc_freeze"]=getattr(self,"_owns_gc_freeze",None)
                out["gc_freeze_pending"]=getattr(self,"_gc_freeze_pending",None)
                pairs={"forward_ms":("f0","f1"),"recompute_ms":("r0","r1"),"backward_including_recompute_ms":("b0","b1"),"mlp_backward_ms":("mlp_bwd_start","attention_bwd_start"),"attention_backward_ms":("attention_bwd_start","b1")}
                for phase in ("forward","recompute"):
                    for name in ("attention","mlp"):
                        prefix=f"{phase}_{name}"
                        pairs[prefix+"_ms"]=(prefix+"0",prefix+"1")
                for label,(a,b) in pairs.items():out[label]=row[a].elapsed_time(row[b])
                out["backward_excluding_recompute_ms"]=out["backward_including_recompute_ms"]-out["recompute_ms"]
                f.write(json.dumps(out)+"\n")
        with (folder/f"gc.rank{rank}.jsonl").open("a") as f:
            for row in gc_events:f.write(json.dumps(row)+"\n")
        pending.clear();gc_events.clear()
        return result
    CheckpointFunction.forward=staticmethod(forward)
    CheckpointFunction.backward=staticmethod(backward)
    MegatronBridgeBackend.forward_backward=fb
