"""Repair copied venv console-script shebangs; packages/source stay unchanged."""
import shutil
from pathlib import Path

VENV = Path("/root/.devbox-venvs/server")
OLD = b"#!/root/.cache/user_artifacts/devboxes/32vj99q/trainers/server-megatron-bridge/.venv/bin/"
NEW = b"#!/root/.devbox-venvs/server/bin/"
BACKUP = Path(__file__).resolve().parent / "devbox_validation/old-launcher-shebangs"

if __name__ == "__main__":
    assert (VENV / "bin/python3").resolve().is_file()
    BACKUP.mkdir(exist_ok=True)
    count = 0
    for path in sorted((VENV / "bin").iterdir()):
        if path.is_symlink() or not path.is_file():
            continue
        with path.open("rb") as stream:
            first = stream.readline(4096)
        if not first.startswith(OLD):
            continue
        original = path.read_bytes()
        backup = BACKUP / path.name
        if backup.exists():
            raise FileExistsError(backup)
        shutil.copy2(path, backup)
        path.write_bytes(first.replace(OLD, NEW, 1) + original[len(first):])
        count += 1
    print(f"Repaired {count} launcher shebangs; originals preserved in {BACKUP}")
