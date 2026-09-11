"""Collect clean-branch reproduction data without importing trainer code."""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE = '/root/glm53-pr1355-repro-20260910'
parser = argparse.ArgumentParser()
parser.add_argument('variant')
args = parser.parse_args()
assert args.variant in ('debug-cp8', 'debug-cp4', 'debug-grouped-cp8', 'full-cp8', 'full-cp4')
out = ROOT / args.variant / 'result'
out.mkdir(parents=True, exist_ok=True)


def copy(pair):
    source, target = pair
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['scp', '-o', 'ControlPath=none', f'tj-32vj99q:{source}', str(target)], check=True)


copy((f'{REMOTE}/results/pr1355-{args.variant}.json', out / 'benchmark.json'))
benchmark = json.loads((out / 'benchmark.json').read_text())
jobs = [(f'{REMOTE}/{args.variant}/driver.log', out / 'driver.log'),
        (f'{REMOTE}/.devbox_up/trainer_srun.log', out / 'trainer.log')]
for kind in ('memory', 'runtime'):
    metadata = benchmark.get(kind + '_profile_stop')
    if metadata:
        for name in metadata['files']:
            assert Path(name).name == name
            jobs.append((metadata['local_path'] + '/' + name, out / kind / name))
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(copy, jobs))
controls = [w for w in benchmark['windows'] if w['phase'] == 'control']
assert controls and all(math.isfinite(w['loss']) and math.isfinite(w['grad_norm']) for w in controls)
mean = statistics.mean(w['fb_elapsed_s'] for w in controls)
summary = {
    'variant': args.variant,
    'controls': len(controls),
    'fb_mean_s': mean,
    'tps_per_gpu': benchmark['tokens_per_step'] / benchmark['num_gpus'] / mean,
    'peak_allocated_gib': max(w['peak_allocated_bytes'] for w in controls) / 2**30,
    'finite_loss_and_gradients': True,
    'artifacts': {},
}
for kind in ('memory', 'runtime'):
    metadata = benchmark.get(kind + '_profile_stop')
    if not metadata:
        continue
    files = [out / kind / name for name in metadata['files']]
    assert sum(p.stat().st_size for p in files) == metadata['size_bytes']
    for path in files:
        with path.open('rb') as stream:
            summary['artifacts'][str(path.relative_to(out))] = hashlib.file_digest(stream, 'sha256').hexdigest()
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k != 'artifacts'}, indent=2))
