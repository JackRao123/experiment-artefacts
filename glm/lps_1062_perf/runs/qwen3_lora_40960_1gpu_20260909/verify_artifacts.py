"""Validate this run's own captures and write a local SHA-256 manifest."""

import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

from mfu import INVENTORIES, estimate

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    manifest = {}
    for short, model, logged_parameters in (
        ("06b", "0.6b", 613482496),
        ("30b-a3b", "30b-a3b", 30567327744),
    ):
        run_path = ROOT / f"runs/qwen3-{short}-40960-1gpu-c10.json"
        run = json.loads(run_path.read_text())
        assert Counter(w["phase"] for w in run["windows"]) == {
            "warmup": 1, "control": 10, "memory_profile": 1, "runtime_profile": 1
        }
        assert run["initial_status"]["step"] == 0
        assert run["final_status"]["step"] == 13
        assert run["seq_len"] == 40960 and run["num_gpus"] == 1
        assert run["final_status"]["world_size"] == 1
        assert run["final_status"]["device"] == "cuda:0"
        assert all(w["num_tokens"] == 40960 for w in run["windows"])
        flops = estimate(json.loads(INVENTORIES[model].read_text()), 40960, 32)
        assert (flops["unique_base_parameters"] +
                flops["adapter_parameters_ep1_shared_experts"]) == logged_parameters

        trace_path = ROOT / f"qwen3-{short}.pt.trace.json"
        assert trace_path.stat().st_size == run["runtime_profile_stop"]["size_bytes"]
        trace = json.loads(trace_path.read_text())
        events = trace["traceEvents"]
        kernels = [e for e in events if e.get("cat") == "kernel"]
        assert kernels, "No CUDA kernels in Kineto capture"
        trace_info = {
            "path": trace_path.name, "bytes": trace_path.stat().st_size,
            "sha256": digest(trace_path), "trace_events": len(events),
            "cuda_kernel_events": len(kernels),
            "summed_kernel_seconds": sum(e.get("dur", 0) for e in kernels) / 1e6,
        }
        del trace, events, kernels

        memory_path = ROOT / f"qwen3-{short}.memory.rank0.pickle"
        assert memory_path.stat().st_size == run["memory_profile_stop"]["size_bytes"]
        # These are captures produced by this session, not third-party pickle files.
        with memory_path.open("rb") as stream:
            snapshot = pickle.load(stream)
        assert snapshot["segments"] and any(snapshot["device_traces"])
        memory_info = {
            "path": memory_path.name, "bytes": memory_path.stat().st_size,
            "sha256": digest(memory_path), "segments": len(snapshot["segments"]),
            "allocator_history_events": sum(len(t) for t in snapshot["device_traces"]),
        }
        manifest[model] = {"runtime": trace_info, "memory": memory_info,
                           "protocol_verified": True, "tensor_census_verified": True}
    rendered = json.dumps(manifest, indent=2) + "\n"
    (ROOT / "artifact_manifest.json").write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
