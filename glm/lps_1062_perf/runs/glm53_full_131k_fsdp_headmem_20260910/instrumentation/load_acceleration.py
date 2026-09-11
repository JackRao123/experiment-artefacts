"""Scoped import acceleration for the immutable GLM snapshot, not training."""
import collections
import contextlib
import logging
from pathlib import Path
import torch

logger=logging.getLogger(__name__)


@contextlib.contextmanager
def accelerate_hf_import(snapshot_path):
    from safetensors import safe_open
    from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource, StateDict
    from megatron.bridge.models.glm_moe_dsa.glm5_bridge import GLM5Bridge
    from megatron.bridge.models.conversion.quantization_utils import maybe_dequantize_fp8_blockwise

    snapshot=Path(snapshot_path)
    original_read=SafeTensorsStateSource.load_tensors
    original_getitem=StateDict.__getitem__
    original_get=StateDict.get
    original_contains=StateDict.__contains__
    original_dequant_descriptor=GLM5Bridge.__dict__['_maybe_dequant_fp8']
    handles=collections.OrderedDict()
    counts=dict(reads=0,opens=0,exact_lookups=0,gpu_dequants=0)

    def read(source,keys_to_load):
        if source.path!=snapshot:
            return original_read(source,keys_to_load)
        index=source.key_to_filename_map
        if not index or any(key not in index for key in keys_to_load):
            return original_read(source,keys_to_load)
        result={}
        for key in keys_to_load:
            path=snapshot/index[key]
            if path not in handles:
                opener=safe_open(path,framework='pt',device='cpu')
                handle=opener.__enter__()
                handles[path]=(opener,handle,frozenset(handle.keys()))
                counts['opens']+=1
                if len(handles)>8:
                    _,(old,_,_)=handles.popitem(last=False)
                    old.__exit__(None,None,None)
            handles.move_to_end(path)
            _,handle,keys=handles[path]
            if key not in keys:return original_read(source,keys_to_load)
            result[key]=handle.get_tensor(key)
            counts['reads']+=1
        return result

    def key_index(state):
        if isinstance(state.source,SafeTensorsStateSource) and state.source.path==snapshot:
            return state.source.key_to_filename_map
        return None

    def getitem(state,key):
        index=key_index(state)
        if index and isinstance(key,str) and not any(c in key for c in '*?['):
            counts['exact_lookups']+=1
            if key not in index:raise KeyError(f'Key not found: {key}')
            return state._load_tensors([key])[key]
        if index and isinstance(key,list) and all(isinstance(k,str) for k in key):
            missing=[k for k in key if k not in index]
            if missing:raise KeyError(f'Keys not found: {missing}')
            counts['exact_lookups']+=len(key)
            return state._load_tensors(key)
        return original_getitem(state,key)

    def get(state,key,default=None):
        index=key_index(state)
        if index and isinstance(key,str):
            counts['exact_lookups']+=1
            return state._load_tensors([key])[key] if key in index else default
        return original_get(state,key,default)

    def contains(state,key):
        index=key_index(state)
        if index and isinstance(key,str):
            counts['exact_lookups']+=1
            return key in index
        return original_contains(state,key)

    def dequant(weight,param_name,hf_state_dict):
        scale=hf_state_dict.get(param_name+'_scale_inv')
        if weight.dtype in (torch.float8_e4m3fn,torch.float8_e5m2):
            device=torch.device('cuda',torch.cuda.current_device())
            weight=weight.to(device)
            scale=scale.to(device) if scale is not None else None
            counts['gpu_dequants']+=1
        return maybe_dequantize_fp8_blockwise(weight,scale)

    SafeTensorsStateSource.load_tensors=read
    StateDict.__getitem__=getitem
    StateDict.get=get
    StateDict.__contains__=contains
    GLM5Bridge._maybe_dequant_fp8=staticmethod(dequant)
    try:
        yield counts
    finally:
        SafeTensorsStateSource.load_tensors=original_read
        StateDict.__getitem__=original_getitem
        StateDict.get=original_get
        StateDict.__contains__=original_contains
        GLM5Bridge._maybe_dequant_fp8=original_dequant_descriptor
        for opener,_,_ in handles.values():opener.__exit__(None,None,None)
        logger.warning('Scoped accelerated HF import: %s',counts)
