"""Smoke test for the LPS-1062 activation-offload changes to
megatron/core/pipeline_parallel/fine_grained_activation_offload.py (CPU-only).

Exercises the patched module directly:
- pool: pad-to-bucket makes routing-jittered shapes share pool keys;
  offload/reload round-trips data correctly through a padded pooled buffer;
  pool free() identity check accepts the padded buffer.
- NUMA/verify helpers: cpulist parsing, env flags (NUMA resolution itself is
  CUDA-dependent and no-ops cleanly without a GPU).

Run with a python that has torch importable, e.g.:
    trainers/server-interface/.venv/bin/python bt_offload_pool_numa_valve_smoke.py

Module under test is located automatically relative to this file's repo
checkout, or override with:
    FGAO_PATH=/path/to/fine_grained_activation_offload.py
"""
import os, sys, importlib.util, pathlib

import torch  # provided by whichever venv runs this

_REL = "server-megatron-bridge/vendor/megatron-bridge/3rdparty/Megatron-LM/megatron/core/pipeline_parallel/fine_grained_activation_offload.py"


def _find_module_under_test():
    override = os.environ.get("FGAO_PATH")
    if override:
        return override
    # search plausible checkouts: worktrees next to ~/Documents/trainers
    roots = [pathlib.Path.home() / "Documents" / d for d in
             ("wt-actplace-hooks", "wt-actplace", "trainers")]
    for r in roots:
        cand = r / _REL
        if cand.is_file():
            return str(cand)
    raise SystemExit(
        "cannot locate fine_grained_activation_offload.py; set FGAO_PATH"
    )


_MOD = _find_module_under_test()
print(f"module under test: {_MOD}")

# Load the module standalone (bypasses megatron package imports).
spec = importlib.util.spec_from_file_location("fgao", _MOD)
fgao = importlib.util.module_from_spec(spec)

# Stub the megatron imports the module does at top level.
class _FakeStream:
    def __init__(self, *a, **k): pass
class _FakeEvent:
    def __init__(self, *a, **k): pass
import types
fake_cuda_graphs = types.ModuleType("megatron.core.transformer.cuda_graphs")
fake_cuda_graphs.is_graph_capturing = lambda: False
fake_utils = types.ModuleType("megatron.core.utils")
fake_utils.nvtx_range_push = lambda *a: None
fake_utils.nvtx_range_pop = lambda *a: None
sys.modules["megatron"] = types.ModuleType("megatron")
sys.modules["megatron.core"] = types.ModuleType("megatron.core")
sys.modules["megatron.core.transformer"] = types.ModuleType("megatron.core.transformer")
sys.modules["megatron.core.transformer.cuda_graphs"] = fake_cuda_graphs
sys.modules["megatron.core.utils"] = fake_utils
# torch.cuda.Stream/Event exist on CPU torch but raise on use; patch the two
# stream attrs the manager builds (not exercised in this test).
torch.cuda.Stream = _FakeStream
torch.cuda.Event = _FakeEvent
spec.loader.exec_module(fgao)

failures = []

# --- helpers -------------------------------------------------------------
assert fgao._parse_cpulist("0-3,8,10-11") == {0, 1, 2, 3, 8, 10, 11}, "cpulist parse"
assert fgao._env_flag("BT_NOPE", "1") is True
os.environ["BT_X"] = "off"
assert fgao._env_flag("BT_X", "1") is False
print("helpers OK")

# --- Hunk 1: pool + pad-to-bucket ----------------------------------------
pool = fgao.OffloadTensorPool(device="cpu", pin_memory=False)
assert pool._numa_cpus is None  # no CUDA on Mac -> clean no-op

handler = object.__new__(fgao.ChunkOffloadHandler)  # bypass __init__ (needs no CUDA here)
handler.cpu_tensor_pool = pool
handler._pool_row_bucket = 8192

# Two routing-jittered shapes that must share one pool key.
x1 = torch.randn(130000, 64, dtype=torch.bfloat16)  # -> padded 131072
x2 = torch.randn(131000, 64, dtype=torch.bfloat16)  # -> padded 131072

state1 = handler.offload(x1)
assert len(state1) == 4, "state carries real shape"
dev, cpu_backup1, use_pool, real_shape1 = state1
assert tuple(cpu_backup1.shape) == (131072, 64), f"padded shape {cpu_backup1.shape}"
assert real_shape1 == (130000, 64), "real shape preserved"

# reload from the padded buffer: correct shape AND correct data
back1 = handler.reload(state1)
assert tuple(back1.shape) == (130000, 64), "reload restores real shape"
assert torch.equal(back1, x1), "reload data matches byte-exactly"
# pool now holds one free padded buffer
assert pool._stats["total_allocated"] == 1

# second tensor with different real shape -> SAME pool key -> reuse (hit)
state2 = handler.offload(x2)
back2 = handler.reload(state2)
assert torch.equal(back2, x2), "second round-trip data matches"
assert pool._stats["pool_hits"] == 1, f"expected reuse, stats={pool._stats}"
assert pool._stats["total_allocated"] == 1, "no fresh allocation for jittered shape"
print("pool + pad-to-bucket OK")

# bucket=0 disables padding
handler._pool_row_bucket = 0
pool2 = fgao.OffloadTensorPool(device="cpu", pin_memory=False)
handler.cpu_tensor_pool = pool2
sA = handler.offload(torch.randn(130000, 8))
handler.reload(sA)
sB = handler.offload(torch.randn(131000, 8))
handler.reload(sB)
assert pool2._stats["total_allocated"] == 2, "bucket=0 keeps shapes distinct"
print("bucket=0 escape OK")

# --- Hunk 2: NUMA resolve no-op without CUDA ------------------------------
cpus, node = fgao._resolve_local_numa_cpus.__wrapped__() if hasattr(fgao._resolve_local_numa_cpus, "__wrapped__") else (None, None)
# direct call would warn/fail on missing torch.cuda.current_device; ensure fail-loud path returns (None, None)
os.environ["BT_OFFLOAD_NUMA_BIND"] = "1"
try:
    cpus, node = fgao._resolve_local_numa_cpus()
except Exception as e:
    failures.append(f"resolve raised instead of fail-loud: {e}")
assert (cpus, node) == (None, None), "no-GPU resolve must fail open to (None, None)"
os.environ["BT_OFFLOAD_NUMA_BIND"] = "off"
cpus, node = fgao._resolve_local_numa_cpus()
assert (cpus, node) == (None, None), "off arm"
print("NUMA resolve fail-loud + off arm OK")

# --- Hunk 4: env gate default ---------------------------------------------
import importlib
assert fgao._env_flag("BT_OFFLOAD_VALVE_TELEMETRY", "0") is False, "telemetry default off"
print("valve telemetry gate default off OK")

if failures:
    print("FAILURES:", failures)
    sys.exit(1)
print("ALL SMOKE TESTS PASSED")
