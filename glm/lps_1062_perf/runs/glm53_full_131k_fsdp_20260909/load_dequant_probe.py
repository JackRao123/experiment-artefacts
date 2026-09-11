"""Check CPU/GPU load-time FP8 dequantization on real immutable weights."""
import importlib.util
import json
import statistics
import time
from pathlib import Path
import torch
from safetensors import safe_open

root=Path('/root/.cache/team_artifacts/huggingface/hub/models--zai-org--GLM-5.3/snapshots/187fb9fff6319062325ff825627ef6db084d9bc6')
src=Path('/root/glm53-131k-fixes-20260909/trainers/server-megatron-bridge/vendor/megatron-bridge/src/megatron/bridge/models/conversion/quantization_utils.py')
spec=importlib.util.spec_from_file_location('dequant_probe_impl',src);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
keys=[next(k for k in index if '.experts.0.'+p+'.weight' in k and not k.endswith('_scale_inv')) for p in ('gate_proj','down_proj')]
results=[]
for key in keys:
    with safe_open(root/index[key],framework='pt',device='cpu') as f:w=f.get_tensor(key)
    with safe_open(root/index[key+'_scale_inv'],framework='pt',device='cpu') as f:s=f.get_tensor(key+'_scale_inv')
    cpu=[];gpu=[]
    for i in range(6):
        t=time.perf_counter();ref=module.maybe_dequantize_fp8_blockwise(w,s);cpu_elapsed=time.perf_counter()-t
        torch.cuda.synchronize();t=time.perf_counter()
        out=module.maybe_dequantize_fp8_blockwise(w.cuda(),s.cuda())
        torch.cuda.synchronize();gpu_elapsed=time.perf_counter()-t
        if i:cpu.append(cpu_elapsed);gpu.append(gpu_elapsed)
    same=torch.equal(ref,out.cpu())
    assert same
    results.append(dict(key=key,shape=list(w.shape),cpu_mean_s=statistics.mean(cpu),gpu_with_h2d_mean_s=statistics.mean(gpu),bitwise_equal=same,cpu_samples=cpu,gpu_samples=gpu))
print(json.dumps(results,indent=2))
