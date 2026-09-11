"""Check the scoped loader against the original on real weights and missing keys."""
import json
import torch
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource, StateDict
from megatron.bridge.models.glm_moe_dsa.glm5_bridge import GLM5Bridge
from load_acceleration import accelerate_hf_import

root='/root/.cache/team_artifacts/huggingface/hub/models--zai-org--GLM-5.3/snapshots/187fb9fff6319062325ff825627ef6db084d9bc6'
state=StateDict(SafeTensorsStateSource(root))
keys=[next(k for k in state if '.experts.0.'+part+'.weight' in k and not k.endswith('_scale_inv')) for part in ('gate_proj','up_proj','down_proj')]
refs={k:GLM5Bridge._maybe_dequant_fp8(state[k],k,state) for k in keys}
with accelerate_hf_import(root) as counts:
    for key in keys:
        got=GLM5Bridge._maybe_dequant_fp8(state[key],key,state)
        assert got.is_cuda and torch.equal(refs[key],got.cpu()),key
    assert state.get('a_key_that_is_not_in_the_checkpoint') is None
    assert set(state._match_keys('model.layers.10.mlp.experts.0.*'))
    recorded=dict(counts)
assert not GLM5Bridge._maybe_dequant_fp8(state[keys[0]],keys[0],state).is_cuda
print(json.dumps(dict(bitwise_equal=True,keys=keys,counts=recorded,original_methods_restored=True),indent=2))
