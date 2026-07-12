# pp_patch_forensics

PP>1 adapter-export wedge forensics. Reproduces and diagnoses the
`/save_weights_for_sampler` deadlock that occurs when exporting LoRA adapters
under pipeline parallelism > 1.

## Scripts

- `make_tiny_nemotronh.py` — generate a random-weight tiny NemotronH hybrid
  checkpoint (8-layer mamba/attention/moe) for PP-export forensics. The
  `--size big` variant scales hidden to 8192 so LoRA adapter tensors cross
  typical NCCL eager/inline-buffer thresholds, testing whether the stock
  export path's PP broadcast is size-dependent (H2 size escalation).

- `run_server_export_probe.py` — boots a single-node torchrun cluster via the
  repo's `mb_cluster` test helper, runs one fb + optim step, then
  `/save_weights_for_sampler`. Probe/forensics behavior is controlled by env
  vars:

  ```
  BT_PROBE_BCAST_LOG=<dir>       per-rank JSONL probes
  BT_PROBE_SKIP_SET_DEVICE=1    reintroduce the pre-fix thread-device bug
  BT_PROBE_FORCE_STOCK_EXPORT=1 force upstream stock materialization at PP>1
  ```

  On an export timeout (the expected wedge for `SKIP_SET_DEVICE`), sends
  SIGUSR1 to every dp_worker rank so the registered faulthandler dumps
  all-thread stacks into the worker log, then re-raises. Worker log is copied
  to `--out-dir`.

- `drive_export.py` — drives fb + optim + `/save_weights_for_sampler` against
  an already-running dp_worker server (used for the cross-node legs where
  `mb_cluster` can't launch). Exits 0 with `status=completed`, or
  `status=wedged_timeout` after SIGUSR1-dumping all local dp_worker stacks.
  The caller owns launch/teardown of the cluster.

## Usage

From the repo root on the devbox, with the server venv:

```bash
server/.venv/bin/python experiment_artefacts/pp_patch_forensics/run_server_export_probe.py \
    --model Qwen/Qwen3-0.6B --nproc 2 --pp 2 --out-dir /root/forensics/qwen_pp2_fixed \
    [--tp N] [--ep N] [--lora-rank R] [--seq-len S] [--export-timeout 60]

server/.venv/bin/python experiment_artefacts/pp_patch_forensics/make_tiny_nemotronh.py \
    --template /root/.cache/user_artifacts/tiny-nemotronh-hybrid \
    --out /root/.cache/user_artifacts/tiny-nemotronh-hybrid-big --size big
```

See `NUMBER_ONE_PP_SAFE_COLLECTIVE.md` at the parent dir for the findings.
