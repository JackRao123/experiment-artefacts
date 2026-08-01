"""LPS-1003 instrument v2: DSA top-k SELECTION digest.

Wraps megatron...dsa_cudnn_kernels._indexer_top_k_one_chunk (same seam probe2
verified on this stack) and logs, per call, a content digest of the SELECTED
indices: per-row int64 sums -> sha1. If, during an event, digests match the
healed reps call-for-call while core_attention output diverges, the top-k
selection is exonerated and the fault is in the attention gather/math.
If digests differ on the event's chunk-2 calls, the selection itself is wrong.

Inert unless BT_DSA_ROWSUM=1. Output: $BT_DSA_ROWSUM_DIR/rank<R>_pid.jsonl
(plain text, one line per call: ts, call#, rows, sk, k, total, sha1[:16]).
Healed-vs-healed digest stability is the built-in control (ties would break
it; measure before trusting).
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import json
import os
import sys
import time

_TARGET = "megatron.core.transformer.experimental_attention_variant.dsa_cudnn_kernels"
_ON = os.environ.get("BT_DSA_ROWSUM") == "1"
_DIR = os.environ.get("BT_DSA_ROWSUM_DIR")

if _ON and _DIR:

    def _install(mod) -> None:
        import torch

        rank = os.environ.get("RANK", "?")
        os.makedirs(_DIR, exist_ok=True)
        jf = open(os.path.join(_DIR, f"rank{rank}_pid{os.getpid()}.jsonl"), "a", buffering=1)
        st = {"n": 0, "errors": 0}

        def say(msg):
            print(f"[dsa_rowsum r{rank}] {msg}", file=sys.stderr, flush=True)

        orig = mod._indexer_top_k_one_chunk

        def wrapped(scores_flat, seq_lens, topk_k, return_topk_scores):
            out = orig(scores_flat, seq_lens, topk_k, return_topk_scores)
            st["n"] += 1
            if st["errors"] > 20:
                return out
            try:
                with torch.no_grad():
                    idx = out.get("indices") if isinstance(out, dict) else None
                    if idx is not None:
                        rows, k = idx.shape
                        rs = idx.clamp(min=0).sum(dim=1, dtype=torch.int64).cpu().numpy()
                        h = hashlib.sha1(rs.tobytes()).hexdigest()[:16]
                        jf.write(json.dumps({
                            "ts": round(time.time(), 3), "call": st["n"],
                            "rows": int(rows), "sk": int(scores_flat.shape[1]),
                            "k": int(k), "total": int(rs.sum()), "sha1": h,
                        }) + "\n")
            except Exception as e:  # noqa: BLE001 — must never break training
                st["errors"] += 1
                if st["errors"] <= 5:
                    say(f"digest error call {st['n']}: {type(e).__name__}: {e}")
            return out

        mod._indexer_top_k_one_chunk = wrapped
        say(f"installed (dir={_DIR})")

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != _TARGET:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                try:
                    _install(module)
                except Exception as e:  # noqa: BLE001
                    print(f"[dsa_rowsum] install failed: {e}", file=sys.stderr, flush=True)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
