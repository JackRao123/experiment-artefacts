"""Prepare CP4 with lower transient FP32 LM-head memory."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'glm53_full_131k_fsdp_noprefetch_20260910'
OLD='/root/glm53-full-131k-fsdp-noprefetch-20260910'
REMOTE='/root/glm53-full-131k-fsdp-headmem-20260910'
for name in ('operate.py','collect.py','profile_driver.py','analyze.py','comm_breakdown.py','trace_breakdown.py','capture_source.py','validate_results.py'):
    text=(PREV/name).read_text().replace(OLD,REMOTE).replace('glm53-full-fsdp-noprefetch-', 'glm53-full-fsdp-headmem-')
    if name=='capture_source.py':text=text.replace('files=[',"files=[SRC/'server-megatron-bridge/src/trainers_server_megatron_bridge/fp32_lm_head.py',")
    (ROOT/name).write_text(text)
for folder in ('instrumentation','lifecycle'):
    (ROOT/folder).mkdir(exist_ok=True)
    for src in (PREV/folder).iterdir():
        if src.suffix not in ('.py','.sh'):continue
        text=src.read_text().replace(OLD,REMOTE)
        if src.name=='run_trainer_node.sh':text=text.replace('export CUDA_DEVICE_MAX_CONNECTIONS=32','export BT_MEMORY_EFFICIENT_LM_HEAD=1\nexport CUDA_DEVICE_MAX_CONNECTIONS=32')
        if src.name=='fsdp_experiment.py':text=text.replace('def install():','def install():\n    from trainers_server_megatron_bridge import chunked_lm_head\n    chunked_lm_head.CHUNKED_LM_HEAD_SEQ_CHUNK = 2048\n    logger.warning("EXPERIMENT: fused FP32 head operations, 2048-token head chunks")')
        (ROOT/folder/src.name).write_text(text)
(ROOT/'cp4ep1').mkdir(exist_ok=True)
c=json.loads((PREV/'cp4ep1/trainer-config.json').read_text());c['checkpoint_dir']=f'{REMOTE}/cp4ep1/checkpoints'
(ROOT/'cp4ep1/trainer-config.json').write_text(json.dumps(c,indent=2)+'\n')
print(ROOT)
