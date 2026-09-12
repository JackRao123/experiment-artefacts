"""Prepare an exact-config continuation for the existing Nsight runner."""
import argparse
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("case")
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent / "nsys_workflow_20260911"
source = root / args.case
target = root / f"{args.case}-validation"
if (target / "benchmark.json").exists():
    raise FileExistsError("Preserve the completed validation")
target.mkdir(exist_ok=True)
for name in ("trainer-config.json", "run_options.json"):
    if (target / name).exists() and (target / name).read_bytes() != (source / name).read_bytes():
        raise ValueError(f"Existing validation differs: {name}")
    shutil.copyfile(source / name, target / name)
print(target)
