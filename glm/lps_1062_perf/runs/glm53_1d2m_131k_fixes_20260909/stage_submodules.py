"""Initialize pinned submodules from the devbox's existing local clones."""
from pathlib import Path
import subprocess

def run(*args):
    return subprocess.check_output(args, text=True).strip()

def stage(old: Path, new: Path):
    if not (new / '.gitmodules').exists():
        return
    lines = run('git', '-C', str(new), 'config', '-f', '.gitmodules', '--get-regexp', r'submodule\..*\.path').splitlines()
    for line in lines:
        key, path = line.split(' ', 1)
        name = key[len('submodule.'):-len('.path')]
        run('git', '-C', str(new), 'config', f'submodule.{name}.url', str(old/path))
        run('git', '-c', 'protocol.file.allow=always', '-C', str(new), 'submodule', 'update', '--init', '--', path)
        stage(old/path, new/path)

if __name__ == '__main__':
    import sys
    stage(Path(sys.argv[1]), Path(sys.argv[2]))
