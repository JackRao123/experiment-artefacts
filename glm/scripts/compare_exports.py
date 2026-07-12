"""Compare CP1-vs-CP2 debug-model adapter exports.

Init weights are CP-invariant (CP ranks replicate parameters), so with the
same seed the step-0 exports must match bit-for-bit: same tensor names,
shapes, dtypes, and bytes. LoRA-B must be all zeros (fresh init sanity).
"""

import json
import sys
from pathlib import Path

from safetensors import safe_open

BASE = Path("/root/.cache/user_artifacts/glm_prof/export_test")


def adapter_dir(variant: str) -> Path:
    root = BASE / variant / "sampler_weights" / "export-smoke"
    if not root.is_dir():
        sys.exit(f"missing export dir: {root}")
    return root


def load(variant: str):
    d = adapter_dir(variant)
    st = next(iter(d.glob("*.safetensors")), None)
    if st is None:
        sys.exit(f"no safetensors under {d}: {[p.name for p in d.iterdir()]}")
    tensors = {}
    with safe_open(st, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    cfg_path = d / "adapter_config.json"
    cfg = json.load(open(cfg_path)) if cfg_path.is_file() else None
    return tensors, cfg, st.name


a, cfg_a, name_a = load("cp1")
b, cfg_b, name_b = load("cp2")
print(f"cp1: {len(a)} tensors ({name_a}); cp2: {len(b)} tensors ({name_b})")

ok = True
if set(a) != set(b):
    ok = False
    print("KEY MISMATCH")
    print("  only cp1:", sorted(set(a) - set(b))[:8])
    print("  only cp2:", sorted(set(b) - set(a))[:8])
else:
    mismatched = []
    for k in sorted(a):
        ta, tb = a[k], b[k]
        if ta.shape != tb.shape or ta.dtype != tb.dtype:
            mismatched.append((k, "shape/dtype", ta.shape, tb.shape, ta.dtype, tb.dtype))
        elif not (ta == tb).all():
            diff = (ta.float() - tb.float()).abs().max().item()
            mismatched.append((k, f"values (max abs diff {diff:.3e})"))
    if mismatched:
        ok = False
        print(f"TENSOR MISMATCHES ({len(mismatched)}):")
        for m in mismatched[:10]:
            print("  ", m)
    else:
        print(f"all {len(a)} tensors bit-identical across cp1/cp2")

lora_b = [k for k in a if "lora_B" in k or "lora_b" in k]
nonzero_b = [k for k in lora_b if a[k].abs().max().item() != 0.0]
print(f"lora_B tensors: {len(lora_b)}, nonzero at init: {len(nonzero_b)}")
if nonzero_b:
    ok = False
    print("  unexpected nonzero lora_B:", nonzero_b[:5])

if cfg_a != cfg_b:
    ok = False
    print("adapter_config.json differs:")
    keys = set(cfg_a or {}) | set(cfg_b or {})
    for k in sorted(keys):
        va, vb = (cfg_a or {}).get(k), (cfg_b or {}).get(k)
        if va != vb:
            print(f"  {k}: cp1={va!r} cp2={vb!r}")
else:
    print("adapter_config.json identical")

print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
