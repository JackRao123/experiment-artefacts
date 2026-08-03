"""LPS-1003 instrument v6: double-execution nondeterminism detector.

BT_DSA_DOUBLE selects comma-separated kernels to run TWICE per call with a
bitwise output comparison (consecutive-run disagreement during an event pins
the internally nondeterministic kernel, mechanism-agnostic):

  indexer  - cudnn DSA indexer_forward_wrapper (compares scores)
  flashmla - dsa_cudnn_kernels._dsa_fwd_flash_mla (compares out and lse)

The FIRST result is returned/used (faithful to the unpatched run); the second
execution is discarded after comparison. Disagreements print one line and log
to $BT_DSA_DOUBLE_DIR/rank<R>.jsonl with per-region stats (128-col tile mask
for scores; row mask for flashmla out).
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import sys
import time

_SEL = {s.strip() for s in os.environ.get("BT_DSA_DOUBLE", "").split(",") if s.strip()}
_DIR = os.environ.get("BT_DSA_DOUBLE_DIR", "/tmp/dsa_double")
_IDX_TARGET = "cudnn.deepseek_sparse_attention.indexer_forward.api"
_KRN_TARGET = "megatron.core.transformer.experimental_attention_variant.dsa_cudnn_kernels"

if _SEL:

    def _say(msg):
        print(f"[dsa_double r{os.environ.get('RANK', '?')}] {msg}", flush=True)

    os.makedirs(_DIR, exist_ok=True)
    _jf = open(
        os.path.join(_DIR, f"rank{os.environ.get('RANK', 'x')}_pid{os.getpid()}.jsonl"),
        "a",
        buffering=1,
    )

    def _emit(rec):
        rec["ts"] = time.time()
        _jf.write(json.dumps(rec) + "\n")

    def _install_indexer(mod):
        import torch

        orig = mod.indexer_forward_wrapper
        state = {"n": 0}

        def wrapped(*a, **k):
            out1 = orig(*a, **k)
            state["n"] += 1
            try:
                # Wrapper allocates its output internally each call, so a
                # plain second call gets a fresh buffer (run1 stays intact).
                out2 = orig(*a, **k)
                s1 = out1["scores"] if isinstance(out1, dict) else out1
                s2 = out2["scores"] if isinstance(out2, dict) else out2
                with torch.no_grad():
                    neq = s1 != s2
                    both_nan = torch.isnan(s1) & torch.isnan(s2)
                    diff = (neq & ~both_nan).sum().item()
                if diff:
                    rec = {"kernel": "indexer", "call": state["n"],
                           "n_diff_elems": int(diff), "shape": list(s1.shape)}
                    _emit(rec)
                    _say(f"INDEXER SELF-DISAGREEMENT call={state['n']} "
                         f"elems={diff} shape={tuple(s1.shape)}")
                else:
                    _emit({"kernel": "indexer", "call": state["n"], "n_diff_elems": 0})
            except Exception as exc:
                _say(f"indexer double failed: {exc}")
            return out1

        mod.indexer_forward_wrapper = wrapped
        try:
            import cudnn

            if hasattr(cudnn, "DSA"):
                cudnn.DSA.indexer_forward_wrapper = wrapped
        except Exception:
            pass
        _say("indexer double-exec armed")

    def _install_flashmla(mod):
        import torch

        orig = mod._dsa_fwd_flash_mla
        state = {"n": 0}

        def wrapped(*a, **k):
            out1, lse1 = orig(*a, **k)
            state["n"] += 1
            try:
                out2, lse2 = orig(*a, **k)
                with torch.no_grad():
                    d_out = (out1 != out2).any(-1).any(-1).sum().item()
                    d_lse = (lse1 != lse2).sum().item()
                if d_out or d_lse:
                    _emit({"kernel": "flashmla", "call": state["n"],
                           "rows_diff": int(d_out), "lse_diff": int(d_lse),
                           "shape": list(out1.shape)})
                    _say(f"FLASHMLA SELF-DISAGREEMENT call={state['n']} "
                         f"rows={d_out} lse={d_lse}")
                else:
                    _emit({"kernel": "flashmla", "call": state["n"], "rows_diff": 0})
            except Exception as exc:
                _say(f"flashmla double failed: {exc}")
            return out1, lse1

        mod._dsa_fwd_flash_mla = wrapped
        _say("flashmla double-exec armed")

    class _Finder(importlib.abc.MetaPathFinder):
        def __init__(self):
            self._busy = set()

        def find_spec(self, fullname, path=None, target=None):
            want = (fullname == _IDX_TARGET and "indexer" in _SEL) or (
                fullname == _KRN_TARGET and "flashmla" in _SEL
            )
            if not want or fullname in self._busy:
                return None
            self._busy.add(fullname)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                self._busy.discard(fullname)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader

            class _Loader(importlib.abc.Loader):
                def create_module(self, s):
                    return loader.create_module(s) if hasattr(loader, "create_module") else None

                def exec_module(self, module):
                    loader.exec_module(module)
                    try:
                        if module.__name__ == _IDX_TARGET:
                            _install_indexer(module)
                        else:
                            _install_flashmla(module)
                    except Exception as exc:
                        _say(f"install FAILED for {module.__name__}: {exc}")

            spec.loader = _Loader()
            return spec

    sys.meta_path.insert(0, _Finder())
    _say(f"armed: {sorted(_SEL)}")
