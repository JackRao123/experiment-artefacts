#!/usr/bin/env python3
"""Resume devbox-up for job w7982d3 after step-8 clone died on a transient
curl error. Drives the idempotent package steps directly (per devbox-up-cli
memory): never re-calls provision.create (no double-queue)."""
import sys

sys.path.insert(0, "/Users/jackrao/.local/bin/devbox_up")

from devbox_up import config, preflight, provision  # noqa: E402


def main() -> None:
    ctx = config.parse_args(["16", "b300", "ali"])
    preflight.run(ctx)
    ctx.job = "w7982d3"
    # Steps 3-5 already passed; re-run 4-5 cheaply for ctx topology, then 6-13.
    provision.wait_running(ctx)      # returns immediately (RUNNING)
    provision.ssh_topology(ctx)
    provision.cluster_sanity(ctx)
    provision.apt_deps(ctx)
    provision.stage_uv(ctx)
    provision.clone_trainers(ctx)    # skips: both clones exist by now
    provision.build_venvs(ctx)       # skips: venv probes pass
    provision.write_env_and_helpers(ctx)
    provision.verification_gate(ctx)
    provision.ssh_aliases(ctx)
    provision.print_done(ctx)


if __name__ == "__main__":
    main()
