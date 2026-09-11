"""Experimental zero-copy grouped BF16 GEMMs for frozen, FSDP-packed experts.

Only TP1, full-recompute, frozen BF16 expert projections are supported. No
packing allocation occurs in forward. Backward reconstructs the view from the
module's currently unsharded parameters, not a stale FSDP allocation.
"""
import functools
import logging
import torch

logger=logging.getLogger(__name__)


def packed_view(module):
    weights=[getattr(module,f'weight{i}') for i in range(module.num_gemms)]
    first=weights[0];n,k=first.shape
    base=first.data_ptr();stride=n*k*first.element_size()
    for i,w in enumerate(weights):
        if w.requires_grad or w.dtype!=torch.bfloat16 or not w.is_contiguous() or w.data_ptr()!=base+i*stride:
            raise RuntimeError('Frozen grouped-MM requires contiguous unsharded BF16 expert weights')
    if first.untyped_storage().nbytes() < (first.storage_offset()+module.num_gemms*n*k)*first.element_size():
        raise RuntimeError('Expert weights do not share a sufficiently large backing allocation')
    return first.as_strided((module.num_gemms,n,k),(n*k,k,1))


class FrozenGroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, offsets, module):
        ctx.module=module
        ctx.save_for_backward(offsets)
        return torch.nn.functional.grouped_mm(inp,packed_view(module).transpose(1,2),offs=offsets)

    @staticmethod
    def backward(ctx, grad):
        if torch.is_grad_enabled():raise NotImplementedError('Experiment supports first-order SFT gradients only')
        offsets,=ctx.saved_tensors
        result=torch.nn.functional.grouped_mm(grad.contiguous(),packed_view(ctx.module),offs=offsets)
        return result,None,None


def install():
    from transformer_engine.pytorch import GroupedLinear
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
    original=GroupedLinear.forward

    @functools.wraps(original)
    def forward(module,inp,m_splits,*args,**kwargs):
        config=getattr(module,'config',None)
        if module.num_gemms<=1 or config is None or config.recompute_granularity!='full':
            return original(module,inp,m_splits,*args,**kwargs)
        if any(getattr(module,f'weight{i}').requires_grad for i in range(module.num_gemms)):
            return original(module,inp,m_splits,*args,**kwargs)
        if module.use_bias or module.tp_size!=1 or inp.ndim!=2 or inp.dtype!=torch.bfloat16 or FP8GlobalStateManager.is_fp8_enabled():
            raise RuntimeError('Unsupported configuration for frozen grouped-MM experiment')
        if torch.is_tensor(m_splits):m_splits=m_splits.tolist()
        offsets=torch.tensor(m_splits,device=inp.device,dtype=torch.int32).cumsum(0,dtype=torch.int32)
        if not getattr(module,'_experiment_grouped_logged',False):
            logger.warning('EXPERIMENT grouped-MM: E=%d K=%d N=%d, zero-copy FSDP weight view',module.num_gemms,module.in_features,module.out_features)
            module._experiment_grouped_logged=True
        out=FrozenGroupedMM.apply(inp,offsets,module)
        return (out,None) if module.return_bias else out
    GroupedLinear.forward=forward
