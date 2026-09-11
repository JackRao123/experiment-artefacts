"""Prepare (do not launch) full-model FSDP CP8 and CP4 experiments."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DEST=ROOT.parent/'glm53_full_131k_fsdp_20260909'
REMOTE='/root/glm53-full-131k-fsdp-20260909'
OLD='/root/glm53-131k-fsdp-cp8ep1-20260909'
DEST.mkdir(exist_ok=True)
for name in ('profile_driver.py','trace_breakdown.py','comm_breakdown.py','analyze.py'):
    (DEST/name).write_text((ROOT/name).read_text().replace(OLD,REMOTE))
for folder in ('instrumentation','lifecycle'):
    (DEST/folder).mkdir(exist_ok=True)
    for src in (ROOT/folder).iterdir():
        if src.suffix in ('.py','.sh'):
            (DEST/folder/src.name).write_text(src.read_text().replace(OLD,REMOTE))
for cp in (8,4,2):
    topology=f'cp{cp}ep1'
    (DEST/topology).mkdir(exist_ok=True)
    c=json.loads((ROOT/'cp8ep1/trainer-config.json').read_text())
    c['base_model']='/root/.cache/team_artifacts/huggingface/hub/models--zai-org--GLM-5.3/snapshots/187fb9fff6319062325ff825627ef6db084d9bc6'
    c['checkpoint_dir']=f'{REMOTE}/{topology}/checkpoints'
    c['context_parallel_size']=cp
    (DEST/topology/'trainer-config.json').write_text(json.dumps(c,indent=2)+'\n')
(DEST/'README.md').write_text('PREPARED, NOT RUN. Full GLM-5.3, BF16 expert storage, MFSDP parameter sharding, TP1/PP1/EP1/ETP1. Start CP8 on eight GPUs; if validated, relax CP. Sequence length remains 131072 per datum. Use one datum per data-parallel replica: CP8 D1, CP4 D2, CP2 D4, so no rank benchmarks an empty DP shard. Separate latency and tokens/GPU. Three warmups, five controls, memory and all-rank runtime profiles. Source and validation must be recorded before interpreting results.\n')
print(DEST)
