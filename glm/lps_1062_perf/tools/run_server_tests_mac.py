# MAC pytest harness for server unit tests that import megatron (volta, LPS-1062).
"""Run server unit tests that import megatron on a Mac (no GPU, no worker venv).

The trainers server test files import megatron (via the backends package init),
which is unavailable on macOS. This harness puts the vendored sources on
sys.path and auto-stubs the CUDA-only import trees (triton, modelopt,
transformer_engine) via a meta-path finder — enough for import-time
decorators/attribute chains; the stubbed functionality is never called by
these CPU tests.

Prereqs (one-time):
  1. Worktree with submodules: git submodule update --init --recursive \\
       loops server/vendor/megatron-bridge
  2. Pure-python deps Jack's base server venv lacks (installed to a scratch
     dir, NOT the venv):
       uv pip install --python ~/Documents/trainers/server/.venv/bin/python \\
         --target /tmp/lps1062_pylibs omegaconf einops tensorboard
  3. Edit WT and PYLIBS below if your paths differ.

Usage:
  ~/Documents/trainers/server/.venv/bin/python run_server_tests_mac.py \\
      [pytest args...] <test file(s)>

Known limitation: tests using torch.multiprocessing.spawn fail because child
processes don't inherit the import stubs (5 such in
test_router_replay_dp_consensus.py — they fail identically on unmodified
code; ignore them or run on a Linux box).
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

WT = Path.home() / "Documents/wt-pp2-packing/server"
PYLIBS = "/tmp/lps1062_pylibs"

for p in (
    str(WT / "src"),
    # loops_models lives at <worktree>/models/src, NOT under server/ — without
    # this it silently resolves to the main checkout's editable install and
    # worktree edits to loops_models go untested (pc_5e53d57839fe).
    str(WT.parent / "models" / "src"),
    str(WT / "vendor/megatron-bridge/src"),
    str(WT / "vendor/megatron-bridge/3rdparty/Megatron-LM"),
    PYLIBS,
):
    sys.path.insert(0, p)

STUB_PREFIXES = ("triton", "modelopt", "transformer_engine", "transformer_engine_torch")


def _make_stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__path__ = []  # type: ignore[attr-defined]
    m.__version__ = "0.0.0"  # packaging.Version parses this
    m.__getattr__ = lambda k: MagicMock()  # type: ignore[attr-defined]
    return m


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return _make_stub(spec.name)

    def exec_module(self, module):
        return None


class _StubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in STUB_PREFIXES):
            return importlib.util.spec_from_loader(fullname, _StubLoader())
        return None


sys.meta_path.append(_StubFinder())

import pytest  # noqa: E402

args = sys.argv[1:] or ["tests/unit/dp_worker/api/test_cp_thd_dispatch.py"]
raise SystemExit(
    pytest.main(["-q", "-m", "not gpu", "--no-header", *[str(a) for a in args]])
)
