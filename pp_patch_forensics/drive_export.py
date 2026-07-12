"""Drive fb + optim + /save_weights_for_sampler against an already-running
dp_worker server (used for the cross-node legs where mb_cluster can't launch).

Exits 0 with status=completed, or status=wedged_timeout after SIGUSR1-dumping
all local dp_worker stacks. The caller owns launch/teardown of the cluster.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "server" / "tests"))

import httpx  # noqa: E402
from helpers import make_text_datum, submit_and_wait  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--export-timeout", type=float, default=120.0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    status = "unknown"
    export_seconds = None
    tokens = list(range(100, 116))
    with httpx.Client(base_url=args.url, timeout=600.0) as c:
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
            assert r.status_code == 200, r.text
            status = "completed"
        except TimeoutError:
            export_seconds = time.perf_counter() - t0
            status = "wedged_timeout"
            print("EXPORT WEDGED; dumping local stacks via SIGUSR1", flush=True)
            subprocess.run(["pkill", "-USR1", "-f", "dp_worker.main"], check=False)
            time.sleep(5)

    summary = {"status": status, "export_seconds": export_seconds}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"SUMMARY {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
