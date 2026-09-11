"""Stage the next FSDP experiment independently of accepted unsharded controls."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT.parent / 'glm53_1d2m_131k_fsdp_cp8ep1_20260909'
REMOTE = '/root/glm53-131k-fsdp-cp8ep1-20260909'
OLD = '/root/glm53-131k-ep1-rootcause-20260909'
DEST.mkdir(exist_ok=True)
for name in ('operate.py','collect.py','profile_driver.py','trace_breakdown.py','comm_breakdown.py','analyze.py','validate.py'):
    (DEST/name).write_text((ROOT/name).read_text().replace(OLD, REMOTE))
for folder in ('instrumentation','lifecycle'):
    (DEST/folder).mkdir(exist_ok=True)
    for src in (ROOT/folder).iterdir():
        if src.suffix not in ('.py','.sh'): continue
        text=src.read_text().replace(OLD,REMOTE)
        if src.name == 'sitecustomize.py':
            text=text.replace('from layer_timing import install', 'from fsdp_experiment import install as install_fsdp\n        install_fsdp()\n        from layer_timing import install')
        if src.name == 'run_trainer_node.sh':
            text=text.replace('export LAYER_TIMING_DIR=', 'export CUDA_DEVICE_MAX_CONNECTIONS=32 BT_MULTI_ADAPTER_ENABLED=0\nexport LAYER_TIMING_DIR=')
        (DEST/folder/src.name).write_text(text)
(DEST/'cp8ep1').mkdir(exist_ok=True)
c=json.loads((ROOT/'cp8ep1/trainer-config.json').read_text())
c['checkpoint_dir']=f'{REMOTE}/cp8ep1/checkpoints'
(DEST/'cp8ep1/trainer-config.json').write_text(json.dumps(c,indent=2)+'\n')
(DEST/'README.md').write_text('IN PROGRESS: opt-in run-local Megatron FSDP, BF16, meta initialization then shard-aware HF import. CP8EP1, 1d2m, 131072 tokens. Single-adapter only; this does not claim multi-adapter swapping or save/load support. Full one-layer recompute and same profiling protocol. No tests changed.\n')
print(DEST)
