#!/usr/bin/env bash
# Build a PYTHONPATH-shadow copy of the vendored megatron package with
# env-gated fingerprint instrumentation (BT_FPRINT_DIR). Never touches the
# shared clone.
set -euo pipefail

SRC=/root/.cache/user_artifacts/trainers_main/server/vendor/megatron-bridge/3rdparty/Megatron-LM/megatron
SHADOW=/root/megatron_shadow
rm -rf "$SHADOW"
mkdir -p "$SHADOW"
cp -r "$SRC" "$SHADOW/megatron"

# Fingerprint helper
cat > "$SHADOW/megatron/core/bt_fprint.py" <<'EOF'
"""Env-gated per-call tensor fingerprints (BT_FPRINT_DIR). Diagnostic only."""
import hashlib
import json
import os

_DIR = os.environ.get("BT_FPRINT_DIR", "")
_SEQ = {"n": 0}


def fprint(site, **tensors):
    if not _DIR:
        return
    rank = os.environ.get("RANK", "0")
    rec = {"seq": _SEQ["n"], "site": site}
    _SEQ["n"] += 1
    for name, t in tensors.items():
        if t is None:
            rec[name] = None
            continue
        try:
            data = t.detach().contiguous().cpu().numpy().tobytes()
            rec[name] = hashlib.sha256(data).hexdigest()[:16]
        except Exception as e:  # never break the forward
            rec[name] = f"ERR:{type(e).__name__}"
    with open(os.path.join(_DIR, f"rank{rank}.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
EOF

python3 - <<'EOF'
import re

base = "/root/megatron_shadow/megatron/core"

# 1) dsa_cudnn_kernels: fingerprint indexer top-k and sparse-attn fwd outputs.
p = f"{base}/transformer/experimental_attention_variant/dsa_cudnn_kernels.py"
s = open(p).read()
assert "def _indexer_topk_bshd(" in s and "def _run_sparse_attention_forward(" in s

s = s.replace(
    "def _indexer_topk_bshd(",
    "def _indexer_topk_bshd_inner(",
    1,
)
s += '''

def _indexer_topk_bshd(*args, **kwargs):
    out = _indexer_topk_bshd_inner(*args, **kwargs)
    from megatron.core.bt_fprint import fprint
    fprint("dsa_topk", indices=out[0], length=out[1])
    return out
'''

s = s.replace(
    "def _run_sparse_attention_forward(",
    "def _run_sparse_attention_forward_inner(",
    1,
)
s += '''

def _run_sparse_attention_forward(*args, **kwargs):
    out = _run_sparse_attention_forward_inner(*args, **kwargs)
    from megatron.core.bt_fprint import fprint
    fprint("dsa_sparse_fwd", out_flat=out[0], lse=out[1])
    return out
'''
open(p, "w").write(s)
print("patched dsa_cudnn_kernels")

# 2) MoE router: fingerprint routing decisions (wrap post-creation; never
# rename the method — TopKRouter subclasses an ABC and renaming re-abstracts it).
p = f"{base}/transformer/moe/router.py"
s = open(p).read()
assert "class TopKRouter" in s
s += '''

_bt_orig_topk_router_forward = TopKRouter.forward


def _bt_topk_router_forward(self, *args, **kwargs):
    out = _bt_orig_topk_router_forward(self, *args, **kwargs)
    from megatron.core.bt_fprint import fprint
    try:
        a, b = out
        fprint("moe_router", probs=a, routing_map=b)
    except Exception:
        pass
    return out


TopKRouter.forward = _bt_topk_router_forward
'''
open(p, "w").write(s)
print("patched moe router")
EOF

echo "shadow ready at $SHADOW (prepend PYTHONPATH=$SHADOW)"
