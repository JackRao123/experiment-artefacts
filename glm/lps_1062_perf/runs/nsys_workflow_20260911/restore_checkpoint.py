"""Restore the exact benchmark checkpoint to node-local storage with HF/Xet."""
import json
import os
from pathlib import Path

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_XET_CACHE"] = "/root/glm53-checkpoints-local/xet"

from huggingface_hub import HfApi, snapshot_download

REPO = "zai-org/GLM-5.3"
REVISION = "187fb9fff6319062325ff825627ef6db084d9bc6"
CACHE = "/root/glm53-checkpoints-local/hub"

if __name__ == "__main__":
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION
    print(f"Restoring {REPO}@{REVISION}: {sum(f.size or 0 for f in info.siblings)} bytes", flush=True)
    snapshot = Path(snapshot_download(REPO, revision=REVISION, cache_dir=CACHE, max_workers=8))
    files = []
    for entry in info.siblings:
        path = snapshot / entry.rfilename
        if not path.is_file() or path.stat().st_size != entry.size:
            raise RuntimeError(f"Missing/incomplete file: {entry.rfilename}")
        files.append({"name": entry.rfilename, "size": entry.size})
    manifest = {"repo": REPO, "revision": REVISION, "snapshot": str(snapshot), "files": files}
    ready = snapshot.parent.parent.parent / "glm53-ready.json"
    temporary = ready.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(ready)
    print(f"READY: {snapshot}", flush=True)
