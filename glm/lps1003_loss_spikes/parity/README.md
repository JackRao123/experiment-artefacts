# parity/ — prod-vs-devbox parity + layer bisection (2026-07-31 overnight)

Goal (Jack's directive): stop assuming the top-k kernel; get a **deterministic
repro on a devbox** or **prove** why prod fires destruction windows 4/4 while
the devbox is 0/9. Method: identical `/forward` payloads on both sides with
FULL per-token logprob dumps, environment/binary fingerprints, and a
layer-by-layer activation tracer to localize where corruption enters.

## Tonight's live resources

| thing | id |
|---|---|
| loops session | `8w6k4y3` (launched 2026-07-31 06:37Z, jack.rao) |
| loops run | `4q9zxjw` |
| trainer deployment | `5wolkzw` — image `trainer-cuda13-sm103-0e0b65a` (verified) |
| trainer pods | `baseten-trainer-5wolkzw-multinode-0` (leader, rank0, port 8000) on `e02-sg-e1n4vn65z0g`; `-0-1` on `e02-sg-e1n4vn65z0t` |
| devbox | `tj-3y0gjkq`, Slurm job on `b300-1-4wprtzyj-0003` + `b300-1-h673xc6t-0010` (NOTE: different node pool than prod!) |
| launcher keepalive | glm_r1 venv, log: scratchpad/launch_parity.log |

## Files

| file | role |
|---|---|
| `probe_lp.py` | stdlib probe: POST payload, poll, dump FULL per-token logprobs gz per rep |
| `rebuild_windows.py` | stdlib: fb → optim → init_trainer_server REBUILD → probes (≤2 cycles/process; 3rd deadlocks) |
| `fingerprint.py` / `fingerprint_diff.py` | loaded-.so sha256 + env + driver capture, and the diff table |
| `compare_lp.py` | wobble/cross parity stats + event anatomy (deciles, onset, worst tokens) |
| `harness/sitecustomize.py` | layer-bisection activation tracer (BT_LTRACE=1); hooks embedding/layers/attn/indexer/mlp/final/logits, per-token-bin rms/absmax/mean/nonfinite |
| `ltrace_analyze.py` | event-vs-healed z-score per (hook,bin) → first divergent layer |
| `devbox_d0.sh` / `boot_ltrace.sh` | devbox drivers (plain / traced boots) |
| `prod_window.sh` | in-pod driver: window probes at READY → steady → fingerprint |
| scratchpad `lws_ltrace_patch.json` | LWS JSON patch injecting the tracer into prod pods (initContainer + emptyDir + PYTHONPATH=/probe:/app/src) |

## Probe tags

- `d0_window` / `p0_window`: reps fired the moment /health answers (fresh boot window)
- `d1_steady` / `p1_steady`: reps after warm-through, weights at base (LoRA B=0)
- `dreb<k>` / `preb<k>`: reps right after an in-process REBUILD (window)
- traced variants live under ltrace run dirs

Destroyed datum = NLL > 2.0 (prod events historically 5-11 nats).
