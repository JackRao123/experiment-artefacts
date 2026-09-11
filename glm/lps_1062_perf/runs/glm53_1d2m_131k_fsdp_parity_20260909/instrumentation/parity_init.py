"""Identical constant LoRA A / zero B initialization across DDP and MFSDP."""
import json
import math
import os
from pathlib import Path
import torch


def install():
    from trainers_server_megatron_bridge import backend
    original=backend.get_model

    def get_model(*args,**kwargs):
        models=original(*args,**kwargs)
        records=[]
        with torch.no_grad():
            for model in models:
                fsdp=os.environ.get('PARITY_FSDP')=='1'
                params=model.module.param_and_grad_buffer.optimizer_named_parameters if fsdp else model.named_parameters()
                for name,p in params:
                    if not p.requires_grad:continue
                    if not '.adapter.' in name:raise ValueError(('unexpected trainable',name))
                    if name.endswith('linear_in.weight'):
                        value=1/math.sqrt(p.shape[-1])
                    elif name.endswith('linear_out.weight'):
                        value=0.
                    else:raise ValueError(('unexpected adapter',name))
                    local=p.to_local() if hasattr(p,'to_local') else p
                    # Round through BF16 before filling masters so both paths
                    # start from exactly the same representable model values.
                    value=float(torch.tensor(value,dtype=torch.bfloat16))
                    local.fill_(value)
                    records.append(dict(name=name,shape=list(p.shape),local_numel=local.numel(),value=value,dtype=str(local.dtype)))
                if fsdp:model.module.install_optimized_model_weights()
        assert records
        out=Path(os.environ['LAYER_TIMING_DIR']);out.mkdir(parents=True,exist_ok=True)
        (out/f'parity-init.rank{os.environ["RANK"]}.json').write_text(json.dumps(records,indent=2)+'\n')
        return models
    backend.get_model=get_model
