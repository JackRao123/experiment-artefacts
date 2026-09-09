"""Retrieve only this run's profiler outputs and validate its protocol."""
import hashlib
import json
import statistics
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
REMOTE = '/root/glm53-main-262k-20260909'
HOST = 'tj-32vj99q'
LABEL = 'glm53-main-262k-cp8ep8-c5'


def copy(remote, local):
    subprocess.run(['scp', '-C', f'{HOST}:{remote}', str(local)], check=True)


def main():
    local = ROOT/f'{LABEL}.json'
    copy(f'{REMOTE}/results/{LABEL}.json', local)
    run = json.loads(local.read_text())
    assert Counter(w['phase'] for w in run['windows']) == {
        'warmup': 1, 'control': 5, 'memory_profile': 1, 'runtime_profile': 1
    }
    assert run['initial_status']['world_size'] == 8
    assert run['seq_len'] == 262144
    assert run['final_status']['step'] - run['initial_status']['step'] == 8
    manifest = {}
    for kind in ('runtime', 'memory'):
        info = run[f'{kind}_profile_stop']
        folder = ROOT/kind
        folder.mkdir(exist_ok=True)
        total = 0
        for name in info['files']:
            assert PurePosixPath(name).name == name
            path = folder/name
            copy(str(PurePosixPath(info['local_path'])/name), path)
            with path.open('rb') as stream:
                sha = hashlib.file_digest(stream, 'sha256').hexdigest()
            size = path.stat().st_size
            total += size
            manifest[str(path.relative_to(ROOT))] = {'bytes': size, 'sha256': sha}
        assert total == info['size_bytes'], (kind, total, info['size_bytes'])
    copy(f'{REMOTE}/driver.log', ROOT/'driver.log')
    copy(f'{REMOTE}/.devbox_up/trainer_srun.log', ROOT/'trainer_srun.log')
    controls = [w for w in run['windows'] if w['phase']=='control']
    timings = [w['fb_elapsed_s'] for w in controls]
    summary = {
        'main_commit': '33d19a3542c3d67553e9dc9d30381d2bdf6db31b',
        'control_count': 5, 'seq_len': 262144, 'gpus': 8,
        'fb_mean_s': statistics.mean(timings), 'fb_sample_sd_s': statistics.stdev(timings),
        'tps_per_gpu': 262144/statistics.mean(timings)/8,
        'peak_control_allocated_gib': max(w['peak_allocated_bytes'] for w in controls)/2**30,
        'peak_control_reserved_gib': max(w['peak_reserved_bytes'] for w in controls)/2**30,
        'artifacts': manifest,
    }
    (ROOT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
