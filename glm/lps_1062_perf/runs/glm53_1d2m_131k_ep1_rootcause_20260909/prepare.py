"""Create run-local copies of the established protocol and devbox lifecycle."""
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'glm53_1d2m_131k_bf16_20260909'
REMOTE = '/root/glm53-131k-ep1-rootcause-20260909'
OLD_REMOTE = '/root/glm53-131k-bf16-20260909'

for name in ('profile_driver.py', 'collect.py', 'trace_breakdown.py', 'comm_breakdown.py', 'analyze.py', 'validate.py'):
    text = (PREVIOUS / name).read_text().replace(OLD_REMOTE, REMOTE)
    (ROOT / name).write_text(text)
for folder in ('instrumentation', 'lifecycle'):
    (ROOT / folder).mkdir(exist_ok=True)
    for src in (PREVIOUS / folder).iterdir():
        if src.suffix not in ('.py', '.sh'):
            continue
        (ROOT / folder / src.name).write_text(src.read_text().replace(OLD_REMOTE, REMOTE))
for topology in ('cp8ep1', 'cp8ep8', 'cp1ep1'):
    (ROOT / topology).mkdir(exist_ok=True)
    config = json.loads((PREVIOUS / topology / 'trainer-config.json').read_text())
    config['checkpoint_dir'] = f'{REMOTE}/{topology}/checkpoints'
    (ROOT / topology / 'trainer-config.json').write_text(json.dumps(config, indent=2) + '\n')
patch = subprocess.check_output(['git', '-C', '/Users/jackrao/Documents/mcore-wt-dsa-cp1', 'diff', '--', 'megatron/core/transformer/moe/token_dispatcher.py'], text=True)
(ROOT / 'singleton-identity-sort.patch').write_text(patch)
print(ROOT)
