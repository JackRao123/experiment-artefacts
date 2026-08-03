"""LPS-1003 instrument v7: pre-forward lever hook (+ chains harness6).

Goal: re-create individual "cold start" ingredients on a WARM trainer, one
lever at a time, without restarting — so the rep0 phenomenon can be re-armed
and studied over and over.

Env:
  BT_LEVER_FILE  shared JSON control file (CPFS) read by every rank
  BT_LEVER_DIR   per-rank jsonl logs
  BT_H6_SITE     path to harness6/sitecustomize.py to chain (double-exec),
                 active only if BT_DSA_DOUBLE is also set

Control file format:
  {"seq": 3,                  # bump to trigger a one-shot application
   "mode": "oneshot",         # or "every": apply before EVERY forward
   "actions": ["empty_cache", "clear_cute_caches", "clear_cublas_ws"],
   "pycode": "..."}           # optional arbitrary python, runs on every rank

Semantics: each rank checks the file at the START of every
execute_forward_backward (once per /forward op per rank). oneshot: applied
iff seq > last-applied seq, BEFORE the forward runs — i.e. the next /forward
behaves like "the first forward after <levers>". Levers:

  empty_cache        gc.collect() + torch.cuda.empty_cache()  (allocator must
                     re-map physical memory during the next forward = cold
                     allocator)
  clear_cute_caches  clear every module-level `_compile_cache` dict under
                     cudnn.deepseek_sparse_attention* (next forward re-runs
                     cute.compile + cuModuleLoad = cold kernels)
  clear_cublas_ws    torch._C._cuda_clearCublasWorkspaces()
  pycode             anything else, without a reboot
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import sys
import time

# ── chain harness6 (double-exec detector) ──────────────────────────────────
_H6 = os.environ.get("BT_H6_SITE", "")
if _H6 and os.path.exists(_H6):
    try:
        exec(compile(open(_H6).read(), _H6, "exec"), {"__name__": "sitecustomize_h6"})
    except Exception as _exc:  # noqa: BLE001
        print(f"[lever] harness6 chain FAILED: {_exc}", flush=True)

# ── lever hook ──────────────────────────────────────────────────────────────
_LFILE = os.environ.get("BT_LEVER_FILE", "")
_LDIR = os.environ.get("BT_LEVER_DIR", "/tmp/levers")
_CTL_TARGET = "trainers_server.dp_worker.api.megatron_controller"

if _LFILE:

    def _say(msg):
        print(f"[lever r{os.environ.get('RANK', '?')}] {msg}", flush=True)

    os.makedirs(_LDIR, exist_ok=True)
    _jf = open(
        os.path.join(_LDIR, f"rank{os.environ.get('RANK', 'x')}_pid{os.getpid()}.jsonl"),
        "a",
        buffering=1,
    )

    def _emit(rec):
        # Never let logging failures (e.g. CPFS quota) break the forward op.
        try:
            rec["ts"] = time.time()
            _jf.write(json.dumps(rec) + "\n")
        except Exception as exc:  # noqa: BLE001
            print(f"[lever r{os.environ.get('RANK', '?')}] emit failed: {exc}",
                  flush=True)

    _state = {"seq": 0, "n": 0}

    def _apply(cfg):
        import gc

        import torch

        applied = []
        for act in cfg.get("actions", []):
            try:
                if act == "empty_cache":
                    gc.collect()
                    torch.cuda.empty_cache()
                    applied.append("empty_cache")
                elif act == "clear_cute_caches":
                    cleared = 0
                    for name, mod in list(sys.modules.items()):
                        if not name.startswith("cudnn.deepseek_sparse_attention"):
                            continue
                        d = getattr(mod, "_compile_cache", None)
                        if isinstance(d, dict) and d:
                            cleared += len(d)
                            d.clear()
                    applied.append(f"clear_cute_caches:{cleared}")
                elif act == "clear_cublas_ws":
                    torch._C._cuda_clearCublasWorkspaces()
                    applied.append("clear_cublas_ws")
                else:
                    applied.append(f"UNKNOWN:{act}")
            except Exception as exc:  # noqa: BLE001
                applied.append(f"FAILED:{act}:{exc}")
        code = cfg.get("pycode")
        if code:
            try:
                g = {"torch": __import__("torch"), "sys": sys, "os": os,
                     "gc": __import__("gc"), "json": json}
                exec(code, g)
                applied.append("pycode:ok")
            except Exception as exc:  # noqa: BLE001
                applied.append(f"pycode:FAILED:{exc}")
        return applied

    def _pre_forward():
        _state["n"] += 1
        try:
            with open(_LFILE) as fh:
                cfg = json.load(fh)
        except FileNotFoundError:
            return
        except Exception as exc:  # noqa: BLE001
            _emit({"call": _state["n"], "read_error": str(exc)})
            return
        seq = int(cfg.get("seq", 0))
        every = cfg.get("mode") == "every"
        if every or seq > _state["seq"]:
            applied = _apply(cfg)
            _state["seq"] = seq
            _emit({"call": _state["n"], "seq": seq, "mode": cfg.get("mode", "oneshot"),
                   "applied": applied})
            _say(f"call={_state['n']} seq={seq} applied={applied}")

    def _install(mod):
        targets = [
            obj for name, obj in vars(mod).items()
            if isinstance(obj, type)
            and "execute_forward_backward" in vars(obj)  # defined here, not inherited
        ]
        if not targets:
            raise RuntimeError("no class defining execute_forward_backward found")
        for cls in targets:
            orig = cls.execute_forward_backward

            def wrapped(self, *a, _orig=orig, **k):
                _pre_forward()
                return _orig(self, *a, **k)

            cls.execute_forward_backward = wrapped
            _say(f"lever hook armed on {cls.__name__}.execute_forward_backward")

    class _Finder(importlib.abc.MetaPathFinder):
        def __init__(self):
            self._busy = set()

        def find_spec(self, fullname, path=None, target=None):
            if fullname != _CTL_TARGET or fullname in self._busy:
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
                        _install(module)
                    except Exception as exc:  # noqa: BLE001
                        _say(f"install FAILED: {exc}")

            spec.loader = _Loader()
            return spec

    sys.meta_path.insert(0, _Finder())
    _say(f"armed: file={_LFILE}")
