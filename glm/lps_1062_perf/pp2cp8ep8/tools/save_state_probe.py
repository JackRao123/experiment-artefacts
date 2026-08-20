#!/usr/bin/env python3
"""save_state_probe.py — bounded /save_state probe for a RUNNING trainer (LPS-1062).

Purpose: determine whether DCP checkpoint save hangs under CP>1 on the BIG
trainer (GLM-5.2 PP2/CP8/EP8 @131k), where a wedge costs a boot — so this is
designed to run AFTER parity/perf legs, as the last act before a planned
restart.

Background (dedekind's small-model matrix, Qwen3-0.6B, w56lorq):
  PP2/CP1 PASS | PP1/CP2 HANG | PP2/CP2 HANG  → CP-triggered, PP-independent.
  All ranks park in mcore dist_checkpointing maybe_finalize_async_calls →
  is_current_async_call_done; forked async-writer children spin forever.
  Sync path (BT_SAVE_STATE_SYNC=1) is GREEN under CP2 (confirmed 2026-08-12).

What it does:
  1. POST /save_state {name, run_id} to the trainer API (default
     127.0.0.1:8001, the devbox trainer port), poll /operations/{id} with a
     hard bound (default 600s; healthy big-model adapter+optim-state save is
     minutes at most — the 0.6B probe saved in seconds).
  2. On timeout: py-spy dump EVERY trainer worker process on ALL nodes of the
     trainer's Slurm allocation, via `srun --jobid=<id> --overlap` (the same
     mechanism wait_trainer_health.sh uses — direct ssh leader→worker does NOT
     work on these devboxes: tj-<job>-<rank> doesn't resolve and key auth
     fails). Dumps land in --out (shared CPFS mount) AND are echoed to the
     captured srun stdout (out/srun_dump_stdout.txt) as a NUL-read-proof copy
     (CPFS close-to-open quirk can read sibling-written files back as NULs).
  3. Exit 0 on save success, 2 on hang (stacks captured), 1 on other error.

Usage (on the leader node, trainer RUNNING and idle):
  python3 save_state_probe.py --job <jobid> --run-id <trainer_id> \
      [--port 8001] [--timeout 600] [--out /root/.cache/user_artifacts/lps1062_pp2/save_probe]

  # sync-path verify (only if the async probe hung): relaunch the trainer
  # with BT_SAVE_STATE_SYNC=1 in the env (toggle exists on branch
  # jackrao/lps-1062-pp2-export @ db5d1826 — cherry-pick the
  # megatron_config.py hunk onto the bring-up branch), re-run this script.

Afterwards: stop_trainer.sh (a hung save leaves NCCL/looper state suspect —
do NOT continue profiling on the same server), report to the orchestrator.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import httpx


def submit_and_poll(port: int, name: str, run_id: str, timeout: float) -> str:
    """Mirror of the test-harness long-poll: 202 + operation_id, then
    /operations/{id} held 30s per poll; 408 = still running."""
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60.0) as c:
        r = c.post("/save_state", json={"name": name, "run_id": run_id})
        r.raise_for_status()
        op_id = r.json()["operation_id"]
        print(f"[probe] save_state submitted: op={op_id}", flush=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            r = c.get(f"/operations/{op_id}")
            if r.status_code == 200:
                return "done"
            if r.status_code == 500:
                print(f"[probe] op errored: {r.text}", flush=True)
                return "error"
            # 408 / still-running: re-poll immediately
        return "timeout"


def find_trainer_slurm_jobid() -> str | None:
    """The trainer runs as `srun --job-name=devbox_trainer` (start_trainer.sh).
    Auto-detect its job id; None if not found."""
    r = subprocess.run(
        ["squeue", "-u", "root", "-h", "-o", "%A %j"],
        capture_output=True, text=True, check=False,
    )
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == "devbox_trainer":
            return parts[0]
    return None


def dump_stacks(job: str, out: Path, nodes: list[str], slurm_jobid: str | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if not slurm_jobid:
        slurm_jobid = find_trainer_slurm_jobid()
    # Find trainer worker PIDs (the -u -m trainers_server.dp_worker.main
    # processes), dump each. py-spy resolution order: the shared-CPFS copy
    # (dedekind installed only on the leader; /root/.local is node-local, so
    # the worker has no py-spy — hausdorff staged a shared copy at
    # lps1062_pp2/bin/py-spy, verified runnable on both nodes of w56lorq),
    # then the node-local install, then PATH.
    inner = (
        # /tmp is node-local: each node must create the out dir itself.
        f"mkdir -p {out}; "
        "for pid in $(pgrep -f 'trainers_server.dp_worker.main' | head -20); do "
        "(/root/.cache/user_artifacts/lps1062_pp2/bin/py-spy dump --pid $pid || "
        "/root/.local/bin/py-spy dump --pid $pid || py-spy dump --pid $pid) "
        f"> {out}/pyspy_$(hostname)_$pid.txt 2>&1; done; "
        "echo dumped on $(hostname); "
        # Echo every dump to stdout as well: srun forwards it leader-side into
        # srun_dump_stdout.txt — immune to the CPFS sibling-node NUL-read quirk.
        f'for f in {out}/pyspy_$(hostname)_*.txt; do echo "===== $f"; cat "$f"; done'
    )
    if not slurm_jobid:
        print("[probe] WARN: no devbox_trainer slurm job found; dumping leader only",
              flush=True)
        with open(out / "srun_dump_stdout.txt", "w") as fh:
            subprocess.run(["bash", "-c", inner], stdout=fh, check=False)
        return
    n = len(nodes)
    print(f"[probe] py-spy dump via srun --jobid={slurm_jobid} across {n} node(s)",
          flush=True)
    cmd = [
        "srun", f"--jobid={slurm_jobid}", "--overlap",
        f"--nodes={n}", f"--ntasks={n}", "--ntasks-per-node=1",
        "--cpus-per-task=1", "--label", "bash", "-c", inner,
    ]
    with open(out / "srun_dump_stdout.txt", "w") as fh:
        subprocess.run(cmd, stdout=fh, check=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="baseten job id (record-keeping only now)")
    ap.add_argument("--slurm-jobid", default=None,
                    help="slurm job id of the running devbox_trainer allocation "
                         "(default: auto-detect via squeue)")
    ap.add_argument("--run-id", required=True, help="trainer_id from the server config (run fence)")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--name", default="pp2cp8-save-probe")
    ap.add_argument("--out", default="/root/.cache/user_artifacts/lps1062_pp2/save_probe")
    ap.add_argument("--nodes", default="localhost", help="comma list: localhost for leader + worker rank indexes, e.g. 'localhost,1'")
    args = ap.parse_args()

    out = Path(args.out)
    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]

    t0 = time.monotonic()
    result = submit_and_poll(args.port, args.name, args.run_id, args.timeout)
    dt = time.monotonic() - t0

    if result == "done":
        print(f"[probe] PASS — save_state completed in {dt:.1f}s", flush=True)
        return 0
    if result == "timeout":
        print(f"[probe] HANG — no completion in {args.timeout:.0f}s; dumping stacks", flush=True)
        dump_stacks(args.job, out, nodes, args.slurm_jobid)
        print(f"[probe] stacks under {out}/pyspy_*.txt — report to orchestrator; "
              "restart the trainer before further work", flush=True)
        return 2
    print(f"[probe] ERROR after {dt:.1f}s (see above)", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
