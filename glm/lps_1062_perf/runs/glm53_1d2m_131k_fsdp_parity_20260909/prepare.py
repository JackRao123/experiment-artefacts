"""Prepare matched-adapter numerical runs; not performance headline populations."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'glm53_1d2m_131k_fsdp_cp8ep1_20260909'
REMOTE='/root/glm53-131k-fsdp-parity-20260909'
OLD='/root/glm53-131k-fsdp-cp8ep1-20260909'
for folder in ('lifecycle','instrumentation'):
    (ROOT/folder).mkdir(exist_ok=True)
    for src in (PREV/folder).iterdir():
        if src.suffix not in ('.py','.sh'):continue
        text=src.read_text().replace(OLD,REMOTE)
        if src.name=='sitecustomize.py':
            text=text.replace('install_fsdp()', 'if os.environ.get("PARITY_FSDP")=="1": install_fsdp()\n        from parity_init import install as install_parity\n        install_parity()')
        (ROOT/folder/src.name).write_text(text)
driver=(PREV/'profile_driver.py').read_text().replace('"learning_rate": 1e-5','"learning_rate": 0.0')
(ROOT/'profile_driver.py').write_text(driver)
for label in ('ddp','fsdp'):
    (ROOT/label).mkdir(exist_ok=True)
    c=json.loads((PREV/'cp8ep1/trainer-config.json').read_text())
    c['checkpoint_dir']=f'{REMOTE}/{label}/checkpoints'
    (ROOT/label/'trainer-config.json').write_text(json.dumps(c,indent=2)+'\n')
print(ROOT)
