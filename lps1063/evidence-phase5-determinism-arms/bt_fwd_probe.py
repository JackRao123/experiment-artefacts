"""LPS-1063 per-layer forward parity probe.

Registered from megatron_controller right after model build (patched copy
only). Active only when BT_FWD_PROBE_DIR is set. For every top-level forward
call of each model chunk, records — in execution order — a fingerprint of:

  - every decoder layer output   (name matches ...layers.<N>)
  - every MoE router output      (module class name contains 'Router')

Fingerprint per tensor: float64 sum/abssum/min/max + sha256 of the raw
bytes (bitwise identity check). Tensors with numel > SIZE_SKIP (warmup-scale)
are skipped entirely so boot/warmup speed is unaffected.

Output: $BT_FWD_PROBE_DIR/fwdprobe.rank<r>.json, rewritten after every
root forward call. Structure:
  {"rank": r, "calls": [[{seq, module, cls, tensors: [{shape, dtype, sha,
    sum, abssum, min, max}]}, ...], ...]}
"""
from __future__ import annotations

import hashlib
import json
import os
import re

import torch

_DIR = os.environ.get("BT_FWD_PROBE_DIR")
_LAYER_RE = re.compile(r"\.layers\.\d+$")
# Optional: regex of module names to hook exhaustively (all submodules) with
# per-token row norms recorded — for drilling into a specific layer.
_SUB_RE = (
    re.compile(os.environ["BT_FWD_PROBE_SUBMODULE_RE"])
    if os.environ.get("BT_FWD_PROBE_SUBMODULE_RE")
    else None
)
SIZE_SKIP = 8_000_000  # elements; probe forwards are ~175-698 tokens/rank

_state = {
    "calls": [],       # list per root call: list of entries
    "current": None,   # entries for the in-flight root call
    "seq": 0,
}


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _tensors_of(out):
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        return [t for t in out if isinstance(t, torch.Tensor)]
    return []


def _fingerprint(t: torch.Tensor, rows: bool = False) -> dict | None:
    if t.numel() == 0 or t.numel() > SIZE_SKIP:
        return None
    with torch.no_grad():
        d = {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
        }
        tc = t.detach().contiguous()
        if tc.is_floating_point():
            f = tc.double()
            d["sum"] = f.sum().item()
            d["abssum"] = f.abs().sum().item()
            d["min"] = f.min().item()
            d["max"] = f.max().item()
        else:
            d["sum"] = int(tc.long().sum().item())
        raw = tc.cpu().numpy().tobytes() if tc.dtype != torch.bfloat16 else (
            tc.view(torch.uint8).cpu().numpy().tobytes()
        )
        d["sha"] = hashlib.sha256(raw).hexdigest()[:16]
        if rows and tc.dim() >= 2 and tc.is_floating_point():
            d["row_norms"] = [
                round(v, 10)
                for v in tc.flatten(1).double().norm(dim=1).cpu().tolist()
            ]
    return d


def _post_hook(name: str, cls: str, rows: bool = False):
    def hook(module, args, output):
        cur = _state["current"]
        if cur is None:
            return
        fps = [
            fp for fp in (_fingerprint(t, rows=rows) for t in _tensors_of(output)) if fp
        ]
        if not fps:
            return
        _state["seq"] += 1
        cur.append({"seq": _state["seq"], "module": name, "cls": cls, "tensors": fps})
    return hook


def _root_pre_hook(module, args):
    _state["current"] = []
    _state["seq"] = 0


def _root_post_hook(module, args, output):
    cur = _state["current"]
    _state["current"] = None
    if not cur:
        return
    _state["calls"].append(cur)
    path = os.path.join(_DIR, f"fwdprobe.rank{_rank()}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"rank": _rank(), "calls": _state["calls"]}, f)
    os.replace(tmp, path)


def register_hooks(model_list) -> None:
    if not _DIR:
        return
    os.makedirs(_DIR, exist_ok=True)
    n_layer = n_router = 0
    for chunk in model_list:
        chunk.register_forward_pre_hook(_root_pre_hook)
        chunk.register_forward_hook(_root_post_hook)
    n_sub = 0
    for chunk in model_list:
        for name, mod in chunk.named_modules():
            cls = type(mod).__name__
            if _SUB_RE is not None and _SUB_RE.search(name):
                mod.register_forward_hook(_post_hook(name, cls, rows=True))
                n_sub += 1
            elif _LAYER_RE.search(name):
                mod.register_forward_hook(_post_hook(name, cls))
                n_layer += 1
            elif "Router" in cls:
                mod.register_forward_hook(_post_hook(name, cls))
                n_router += 1
    print(
        f"[bt_fwd_probe] rank {_rank()}: hooks on {n_layer} layers + "
        f"{n_router} routers + {n_sub} submodules -> {_DIR}",
        flush=True,
    )
