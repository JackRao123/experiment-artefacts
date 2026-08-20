"""Verify every RECORD-hashed file in a venv (ENOSPC-casualty sweep)."""
import base64
import csv
import hashlib
import sys
from pathlib import Path

sp = Path(sys.argv[1])
skip = ("nvidia_cutlass_dsl",)  # post-surgery by design (cu13 wins overlaps)
bad, checked, missing = [], 0, 0
dist_infos = sorted(sp.glob("*.dist-info"))
for di in dist_infos:
    if any(di.name.lower().startswith(s) for s in skip):
        continue
    rec = di / "RECORD"
    if not rec.exists():
        continue
    for row in csv.reader(rec.open()):
        if len(row) < 2 or not row[1].startswith("sha256="):
            continue
        rel, hashval = row[0], row[1]
        if rel.startswith(".."):
            continue  # entry-point scripts: installer-rewritten shebangs
        p = sp / rel
        if not p.exists():
            bad.append((di.name, rel, "MISSING"))
            missing += 1
            continue
        if p.is_dir() or p.is_symlink():
            continue
        h = hashlib.sha256(p.read_bytes()).digest()
        want = base64.urlsafe_b64decode(hashval[7:] + "=" * (-len(hashval[7:]) % 4))
        if h != want:
            bad.append((di.name, rel, "HASH"))
        checked += 1
print(f"checked {checked} hashed files across {len(dist_infos)} dist-infos")
for pkg, rel, kind in bad[:60]:
    print(f"{kind}: {pkg}: {rel}")
print(f"TOTAL MISMATCHES: {len(bad)} (missing: {missing})")
