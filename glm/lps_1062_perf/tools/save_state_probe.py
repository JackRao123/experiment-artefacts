#!/usr/bin/env python3
"""Bounded /save_state probe (LPS-1062, dedekind's CP>1 async-hang follow-up).

POSTs /save_state and polls with a hard deadline. On timeout, captures py-spy
stacks from ALL trainer ranks on both nodes (dedekind's CP2 evidence pattern)
into $OUT_DIR, then exits 2. On success prints the result and exits 0.

Run on the leader: python3 save_state_probe.py --name pp2cp8ep8-save-probe
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8001"
OUT_DIR = Path("/root/.cache/user_artifacts/lps1062_pp2/export_test")
VENV_BIN = "/root/.cache/user_artifacts/trainers_main/server/.venv/bin"


def capture_stacks(label: str) -> None:
    """py-spy dump every dp_worker.main process on both nodes via srun."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = (
        f'for pid in $(pgrep -f "[d]p_worker.main"); do '
        f"{VENV_BIN}/py-spy dump --pid $pid > {OUT_DIR}/pyspy_{label}_$(hostname)_$pid.txt 2>&1; "
        f"done; echo captured on $(hostname)"
    )
    subprocess.run(
        ["srun", "--overlap", "-N2", "-n2", "bash", "-c", cmd],
        timeout=300,
        check=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pp2cp8ep8-save-probe")
    ap.add_argument("--timeout-s", type=float, default=600.0)
    args = ap.parse_args()

    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        r = client.post(
            "/save_state",
            json={"name": args.name},
            headers={"Idempotency-Key": uuid.uuid4().hex},
        )
        if r.status_code != 202:
            print(f"submit failed: {r.status_code} {r.text[:1000]}")
            return 1
        op_id = r.json()["operation_id"]
        print(f"save_state op {op_id} submitted; deadline {args.timeout_s}s")
        deadline = time.monotonic() + args.timeout_s
        while time.monotonic() < deadline:
            rr = client.get(f"/operations/{op_id}", timeout=60.0)
            if rr.status_code == 408:
                continue
            rr.raise_for_status()
            body = rr.json()
            status = body.get("status")
            if status == "done":
                print(f"SAVE_STATE DONE: {str(body.get('result'))[:500]}")
                return 0
            if status == "error":
                print(f"SAVE_STATE ERROR: {str(body.get('error'))[:2000]}")
                return 1
            time.sleep(10)
    print(f"TIMEOUT after {args.timeout_s}s — capturing py-spy stacks on all ranks")
    capture_stacks(args.name)
    print(f"stacks under {OUT_DIR}/pyspy_{args.name}_*")
    return 2


if __name__ == "__main__":
    sys.exit(main())
