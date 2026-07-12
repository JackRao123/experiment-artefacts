"""Drive one real LoRA export through the dp_worker server under probes (H1/H2).

Boots a single-node torchrun cluster via the repo's mb_cluster test helper,
runs one fb + optim step, then /save_weights_for_sampler. Probe/forensics
behavior is controlled by env vars set by the caller:

    BT_PROBE_BCAST_LOG=<dir>         per-rank JSONL probes (see _bt_probe)
    BT_PROBE_SKIP_SET_DEVICE=1       reintroduce the pre-fix thread-device bug
    BT_PROBE_FORCE_STOCK_EXPORT=1    force upstream stock materialization at PP>1

On an export timeout (the expected wedge for SKIP_SET_DEVICE), sends SIGUSR1
to every dp_worker rank so the registered faulthandler dumps all-thread stacks
into the worker log, then re-raises. Worker log is copied to --out-dir.

Usage (from repo root on the devbox):
    server/.venv/bin/python experiment_artefacts/pp_patch_forensics/run_server_export_probe.py \
        --model Qwen/Qwen3-0.6B --nproc 2 --pp 2 --out-dir /root/forensics/qwen_pp2_fixed \
        [--tp N] [--ep N] [--lora-rank R] [--seq-len S] [--export-timeout 60]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "server" / "tests"))

import httpx  # noqa: E402
from helpers import (  # noqa: E402
    make_text_datum,
    mb_cluster,
    submit_and_wait,
)


def _sigusr1_all_ranks() -> None:
    """Dump all-thread stacks on every dp_worker rank via faulthandler."""
    subprocess.run(["pkill", "-USR1", "-f", "dp_worker.main"], check=False)
    time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--nproc", type=int, default=2)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--ep", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--export-timeout", type=float, default=60.0)
    parser.add_argument("--boot-timeout", type=float, default=600.0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = Path(tempfile.mkdtemp(prefix="export_probe_"))
    weight_sync_dir = tmp_path / "lora_sync"
    status = "unknown"
    export_seconds = None
    try:
        with mb_cluster(
            tmp_path,
            base_model=args.model,
            nproc=args.nproc,
            max_seq_len=args.seq_len,
            lora_rank=args.lora_rank,
            pipeline_parallel_size=args.pp,
            tensor_parallel_size=args.tp,
            expert_parallel_size=args.ep,
            weight_sync={"type": "local", "path": str(weight_sync_dir)},
            boot_timeout=args.boot_timeout,
        ) as base_url:
            tokens = list(range(100, 116))
            with httpx.Client(base_url=base_url, timeout=600.0) as c:
                r = submit_and_wait(
                    c, "/forward_backward", json={"data": [make_text_datum(tokens)]}
                )
                assert r.status_code == 200, f"fb: {r.status_code} {r.text}"
                r = submit_and_wait(
                    c, "/optim_step", json={"adam_params": {"learning_rate": 1e-4}}
                )
                assert r.status_code == 200, f"opt: {r.status_code} {r.text}"

                t0 = time.perf_counter()
                try:
                    r = submit_and_wait(
                        c,
                        "/save_weights_for_sampler",
                        json={"name": "probe", "trainer_server_id": "forensics"},
                        timeout=args.export_timeout,
                    )
                    export_seconds = time.perf_counter() - t0
                    status = f"http_{r.status_code}"
                    assert r.status_code == 200, r.text
                    status = "completed"
                except TimeoutError:
                    export_seconds = time.perf_counter() - t0
                    status = "wedged_timeout"
                    print(
                        f"EXPORT WEDGED after {export_seconds:.1f}s; "
                        "dumping stacks via SIGUSR1",
                        flush=True,
                    )
                    _sigusr1_all_ranks()
    finally:
        workers_log = tmp_path / "workers.log"
        if workers_log.exists():
            shutil.copy(workers_log, out_dir / "workers.log")
        adapter_dir = weight_sync_dir / "sampler_weights" / "probe"
        if (adapter_dir / "adapter_model.safetensors").exists():
            shutil.copy(
                adapter_dir / "adapter_model.safetensors",
                out_dir / "adapter_model.safetensors",
            )
            for extra in adapter_dir.glob("*.json"):
                shutil.copy(extra, out_dir / extra.name)
        summary = {
            "status": status,
            "export_seconds": export_seconds,
            "model": args.model,
            "nproc": args.nproc,
            "pp": args.pp,
            "tp": args.tp,
            "ep": args.ep,
            "lora_rank": args.lora_rank,
            "probe_env": {
                k: v for k, v in os.environ.items() if k.startswith("BT_PROBE")
            },
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"SUMMARY {json.dumps(summary)}", flush=True)
        shutil.rmtree(tmp_path, ignore_errors=True)

    if status not in ("completed", "wedged_timeout"):
        sys.exit(1)


if __name__ == "__main__":
    main()
