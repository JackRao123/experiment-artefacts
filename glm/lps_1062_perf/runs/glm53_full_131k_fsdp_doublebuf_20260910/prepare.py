"""Prepare persistent FSDP buffers and accelerated immutable-checkpoint loading."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLDROOT=ROOT.parent/'glm53_full_131k_fsdp_20260909'
OLD='/root/glm53-full-131k-fsdp-20260909'
REMOTE='/root/glm53-full-131k-fsdp-doublebuf-20260910'
for name in ('operate.py','collect.py','profile_driver.py','analyze.py','trace_breakdown.py','comm_breakdown.py','capture_source.py'):
    text=(OLDROOT/name).read_text().replace(OLD,REMOTE).replace("glm53-full-fsdp-{topology}","glm53-full-fsdp-doublebuf-{topology}")
    if name=='capture_source.py':
        text=text.replace("'c5de29803ca1ac8caf606a72a3300115c1b0fda0'","'e6c86ea3b75674dfff3769c7c92141629864dc0c'")
        text=text.replace("files=[", "files=[CORE/'megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py',RUN/'instrumentation/load_acceleration.py',")
    (ROOT/name).write_text(text)
for folder in ('instrumentation','lifecycle'):
    (ROOT/folder).mkdir(exist_ok=True)
    for src in (OLDROOT/folder).iterdir():
        if src.suffix not in ('.py','.sh'):continue
        text=src.read_text().replace(OLD,REMOTE)
        if src.name=='fsdp_experiment.py':
            text=text.replace('import time','import time\nfrom load_acceleration import accelerate_hf_import')
            text=text.replace("cfg.ddp.keep_fp8_transpose_cache = False", "cfg.ddp.keep_fp8_transpose_cache = False\n        cfg.ddp.fsdp_double_buffer = True\n        cfg.ddp.megatron_fsdp_max_pool_double_buffer = True\n        cfg.ddp.fsdp_db_use_persist_buf_on_alloc_fail = False")
            text=text.replace('with torch.no_grad():','with torch.no_grad(), accelerate_hf_import(bridge.hf_pretrained.model_name_or_path):')
        if src.name=='stop_trainer.sh':text=text.replace('sleep 10','sleep 40')
        (ROOT/folder/src.name).write_text(text)
for cp in (8,4,2):
    topology=f'cp{cp}ep1';(ROOT/topology).mkdir(exist_ok=True)
    c=json.loads((OLDROOT/topology/'trainer-config.json').read_text())
    c['checkpoint_dir']=f'{REMOTE}/{topology}/checkpoints'
    (ROOT/topology/'trainer-config.json').write_text(json.dumps(c,indent=2)+'\n')
print(ROOT)
