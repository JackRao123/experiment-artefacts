"""Capture exact source/dependency fingerprints without environment secrets."""
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

RUN=Path(__file__).resolve().parent
SRC=Path('/root/glm53-131k-fixes-20260909/trainers')
BRIDGE=SRC/'server-megatron-bridge/vendor/megatron-bridge'
CORE=BRIDGE/'3rdparty/Megatron-LM'
def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
repos={name:dict(head=git(path,'rev-parse','HEAD'),status=git(path,'status','--porcelain')) for name,path in [('trainers',SRC),('bridge',BRIDGE),('core',CORE)]}
files=[SRC/'server-megatron-bridge/src/trainers_server_megatron_bridge/fp32_lm_head.py',CORE/'megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py',RUN/'instrumentation/load_acceleration.py',SRC/'server-megatron-bridge/src/trainers_server_megatron_bridge/backend.py',SRC/'server-megatron-bridge/src/trainers_server_megatron_bridge/megatron_config.py',CORE/'megatron/core/utils.py',CORE/'megatron/core/transformer/moe/token_dispatcher.py',CORE/'megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py',RUN/'instrumentation/fsdp_experiment.py',RUN/'instrumentation/layer_timing.py',RUN/'lifecycle/run_trainer_node.sh']
fingerprints={str(p):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),git_blob=subprocess.check_output(['git','hash-object',str(p)],text=True).strip()) for p in files}
versions={name:importlib.metadata.version(name) for name in ('torch','transformer_engine','triton')}
print(json.dumps(dict(repos=repos,versions=versions,files=fingerprints,core_equivalent_commit='e6c86ea3b75674dfff3769c7c92141629864dc0c',kernel_mode='standard TE; grouped-MM prototype not enabled'),indent=2))
