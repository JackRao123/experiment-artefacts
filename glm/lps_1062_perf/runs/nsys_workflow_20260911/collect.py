"""Retrieve only finalized nsys artifacts, verify SHA256, optionally analyze locally."""
import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("case")
parser.add_argument("--host", default="tj-q9exk9w")
parser.add_argument("--control-path")
parser.add_argument("--remote-root", default="/root/glm53-fsdp-nsys-131k-20260911")
parser.add_argument("--allow-partial", action="store_true")
parser.add_argument("--analyze", action="store_true")
args = parser.parse_args()
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.case):
    parser.error("case must be a simple directory name")
root = Path(__file__).resolve().parent
local = root / args.case
remote = args.remote_root.rstrip("/") + "/" + args.case
ssh = ["ssh", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=6"]
if args.control_path:
    ssh += ["-S", args.control_path]
payload = subprocess.check_output([*ssh, args.host, "cat " + shlex.quote(remote + "/benchmark.json")])
benchmark = json.loads(payload)
if not benchmark.get("capture_complete") and not args.allow_partial:
    raise RuntimeError("Capture is unfinished or failed; use --allow-partial only for explicitly qualified partial results")
artifacts = benchmark.get("artifacts", {})
if not artifacts:
    raise RuntimeError("No finalized artifact manifest. Older runs require explicit manual verification.")
allowed = {f"{phase}.{ext}" for phase in ("timing", "metrics") for ext in ("nsys-rep", "sqlite")}
if not artifacts.keys() <= allowed:
    raise RuntimeError("Unexpected artifact names in manifest")
local.mkdir(parents=True, exist_ok=True)
# Freeze the metadata used for this transfer rather than racing a second read.
(local / "benchmark.json").write_bytes(payload)
for name, info in artifacts.items():
    subprocess.run(["rsync", "-az" if name.endswith(".sqlite") else "-a", "--partial", "-e", shlex.join(ssh),
                    f"{args.host}:{remote}/{name}", str(local / name)], check=True)
    path = local / name
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Not a regular artifact: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != info["sha256"] or path.stat().st_size != info["bytes"]:
        raise RuntimeError(f"Artifact verification failed: {name}")
    print(f"Verified {name} ({info['bytes']} bytes)", flush=True)
if args.analyze:
    for name in artifacts:
        if name.endswith(".sqlite"):
            subprocess.run([sys.executable, str(root / "analyze.py"), str(local / name),
                            "--benchmark", str(local / "benchmark.json")], check=True)
