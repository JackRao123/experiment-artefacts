#!/usr/bin/env python3
"""offload_preflight.py — exercise every boot-time offload code path WITHOUT
loading a model (conway, 2026-08-20).

Why this exists: the first two 131k offload boots died on bugs that needed no
model, no distributed group, and no 131k context to reproduce.

  - `PipelineOffloadManager.__init__` called `reset()` before declaring the
    counters `reset()` reads -> AttributeError. Reproducible by constructing
    ONE object.
  - `_current_gpu_pci_bus_id` returned torch's integer bus number where a BDF
    string was required, so the NUMA lookup ENOENT'd and every rank silently
    fell back to below-requirement page placement. Reproducible by ONE call.

Each cost a ~20-minute weight load to discover. This script runs in seconds.

Run it on the box, in the worker venv, on an IDLE GPU, BEFORE every offload
boot:

    $VENV/bin/python offload_preflight.py

Exit code 0 = every check passed. Non-zero = do not spend a boot.
"""

from __future__ import annotations

import os
import sys
import traceback

FAILURES: list[str] = []
NOTES: list[str] = []


def check(name: str):
    def deco(fn):
        print(f"\n--- {name} ---")
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001 — a harness reports, never raises
            FAILURES.append(f"{name}: {type(e).__name__}: {e}")
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
        return fn

    return deco


def main() -> int:
    import torch

    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
          f"devices {torch.cuda.device_count()}")
    print(f"NVTE_CPU_OFFLOAD_V1={os.environ.get('NVTE_CPU_OFFLOAD_V1', '<unset>')} "
          f"BT_OFFLOAD_NUMA_BIND={os.environ.get('BT_OFFLOAD_NUMA_BIND', '<unset, defaults on>')} "
          f"BT_OFFLOAD_VALVE_TELEMETRY={os.environ.get('BT_OFFLOAD_VALVE_TELEMETRY', '<unset, defaults off>')}")

    from megatron.core.pipeline_parallel import fine_grained_activation_offload as fo

    @check("1. TE import-time latch agrees with the environment")
    def _te_latch():
        from transformer_engine.pytorch import cpu_offload as te_cpu

        env = os.environ.get("NVTE_CPU_OFFLOAD_V1", "0")
        latch = getattr(te_cpu, "NVTE_CPU_OFFLOAD_V1", None)
        print(f"env={env} latch={latch}")
        if env == "1" and latch is not True:
            raise AssertionError(
                "env=1 but TE latched False — TE was imported before the export. "
                "Fix the LAUNCHER environment; this would run the pre-V1 path silently."
            )
        if env != "1":
            NOTES.append("NVTE_CPU_OFFLOAD_V1 is not 1 here; an offload boot REQUIRES it.")

    @check("2. GPU PCI BDF resolves to a real sysfs path")
    def _bdf():
        bdf = fo._current_gpu_pci_bus_id(torch.cuda.current_device())
        print(f"bdf={bdf}")
        if ":" not in str(bdf):
            raise AssertionError(f"not a BDF: {bdf!r} (torch's integer bus number?)")
        path = f"/sys/bus/pci/devices/{bdf}/numa_node"
        if not os.path.exists(path):
            raise AssertionError(f"{path} does not exist")
        print(f"{path} -> {open(path).read().strip()}")

    @check("3. NUMA-local CPU set resolves (the gate, not a nicety)")
    def _numa():
        cpus, node = fo._resolve_local_numa_cpus()
        print(f"node={node} local_cpus={len(cpus) if cpus else 0}")
        if node is None or not cpus:
            raise AssertionError(
                "unresolved -> pinned buffers fall back to process-default "
                "(interleaved) placement, measured ~30% BELOW the offload's "
                "bandwidth requirement while still running and looking healthy"
            )

    @check("4. PipelineOffloadManager constructs (the AttributeError class)")
    def _mgr():
        fo.PipelineOffloadManager.reset_instance()
        m = fo.PipelineOffloadManager()
        for attr in ("_valve_telemetry_enabled", "_valve_telemetry_every",
                     "_valve_telemetry_iter"):
            if not hasattr(m, attr):
                raise AssertionError(f"missing {attr} after __init__")
        m.reset()          # the call that failed on the first boot
        m.reset()          # and again, the steady-state path
        print(f"telemetry enabled={m._valve_telemetry_enabled} "
              f"every={m._valve_telemetry_every}")
        agg = m.get_valve_telemetry()
        print(f"valve telemetry aggregate (empty is correct here): {agg}")

    @check("5. pinned pool allocates, and pages land on the local node")
    def _pool():
        cpus, node = fo._resolve_local_numa_cpus()
        pool = fo.OffloadTensorPool(pin_memory=True)
        shape, dtype = (256, 1024, 1024), torch.bfloat16   # 512 MiB
        buf = pool.get_tensor(shape, dtype) if hasattr(pool, "get_tensor") else None
        if buf is None:
            NOTES.append("OffloadTensorPool has no get_tensor(); skipped the pool probe")
            return
        print(f"pinned buffer {tuple(buf.shape)} {buf.dtype} "
              f"{buf.numel() * buf.element_size() / 2**20:.0f} MiB pinned={buf.is_pinned()}")
        if not buf.is_pinned():
            raise AssertionError("buffer is not pinned — D2H will use the pageable path (5x slower)")
        if node is not None and hasattr(fo, "_verify_page_placement"):
            fo._verify_page_placement(buf, node)

    @check("6. a real device-to-host round trip through the pinned buffer")
    def _roundtrip():
        pool = fo.OffloadTensorPool(pin_memory=True)
        shape, dtype = (64, 1024, 1024), torch.bfloat16
        src = torch.randn(*shape, dtype=torch.float32, device="cuda").to(dtype)
        host = pool.get_tensor(shape, dtype) if hasattr(pool, "get_tensor") else \
            torch.empty(shape, dtype=dtype, pin_memory=True)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            host.copy_(src, non_blocking=True)
        stream.synchronize()
        back = host.to("cuda", non_blocking=True)
        torch.cuda.synchronize()
        if not torch.equal(back, src):
            raise AssertionError("round trip changed the data")
        print(f"round trip bit-exact over {src.numel() * src.element_size() / 2**20:.0f} MiB")

    print("\n" + "=" * 64)
    for n in NOTES:
        print(f"NOTE  {n}")
    if FAILURES:
        print(f"\n{len(FAILURES)} CHECK(S) FAILED — do not spend a boot:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\nAll offload preflight checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
