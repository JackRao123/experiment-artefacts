"""Prepare the lower-memory no-lookahead FSDP experiment."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'glm53_full_131k_fsdp_doublebuf_20260910'
OLD='/root/glm53-full-131k-fsdp-doublebuf-20260910'
REMOTE='/root/glm53-full-131k-fsdp-noprefetch-20260910'
for name in ('operate.py','collect.py','profile_driver.py','analyze.py','comm_breakdown.py','trace_breakdown.py','capture_source.py','validate_results.py'):
    text=(PREV/name).read_text().replace(OLD,REMOTE).replace('glm53-full-fsdp-doublebuf-', 'glm53-full-fsdp-noprefetch-')
    (ROOT/name).write_text(text)
for folder in ('instrumentation','lifecycle'):
    (ROOT/folder).mkdir(exist_ok=True)
    for src in (PREV/folder).iterdir():
        if src.suffix not in ('.py','.sh'):continue
        text=src.read_text().replace(OLD,REMOTE)
        if src.name=='fsdp_experiment.py':
            text=text.replace('def install():', 'def install():\n    from megatron.core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp import MegatronFSDP, PrefetchOrder\n    original_unshard = MegatronFSDP.all_gather_and_wait_parameters_ready\n    def unshard(self, params, prefetch=True, prefetch_order=PrefetchOrder.FORWARD_PASS_ORDER, wait_bucket_ready=True, bwd=False):\n        return original_unshard(self, params, prefetch=False, prefetch_order=prefetch_order, wait_bucket_ready=wait_bucket_ready, bwd=bwd)\n    MegatronFSDP.all_gather_and_wait_parameters_ready = unshard\n    logger.warning("EXPERIMENT: parameter lookahead disabled; required all-gathers remain")')
        if src.name=='layer_timing.py':
            text=text.replace("if state['step']==0:\n            inventory=[]", "if state['step']==0:\n            folder.mkdir(parents=True,exist_ok=True)\n            def oom(device, alloc_size, allocated, total):\n                torch.cuda.memory._dump_snapshot(str(folder/f'memory.rank{rank}.oom.pickle'))\n            torch._C._cuda_attach_out_of_memory_observer(oom)\n            inventory=[]")
        (ROOT/folder/src.name).write_text(text)
for cp in (4,2):
    topology=f'cp{cp}ep1';(ROOT/topology).mkdir(exist_ok=True)
    c=json.loads((PREV/topology/'trainer-config.json').read_text());c['checkpoint_dir']=f'{REMOTE}/{topology}/checkpoints'
    (ROOT/topology/'trainer-config.json').write_text(json.dumps(c,indent=2)+'\n')
print(ROOT)
