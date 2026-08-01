# probe2 — DSA top-k output-indices under-write harness (LPS-1003 Issue 2)

Built 2026-07-30 night to run the rewritten `experiment_handoff.md` program.
**Outcome: the under-write did not reproduce.** Numbers and interpretation live
in `../INVESTIGATION.md` ("2026-07-30 night") and `../VERDICT.md`; this file is
how to re-run the tooling.

## What's here

| file | role |
|---|---|
| `sitecustomize.py` | the harness. Env-gated, wraps `_indexer_top_k_one_chunk` via a meta-path hook. Deploy as `<dir>/sitecustomize.py` on the trainer ranks' PYTHONPATH; never modify vendored `3rdparty/`. |
| `boot_probe2.sh` | boot wrapper: exports harness env, then dispatches the normal `.devbox_up/start_trainer.sh`. Modes `audit` / `stage` / `stagefix`. |
| `audit_summary.py` | aggregates the per-rank JSONLs into a verdict summary. |
| `test_staging.py` | standalone: proves the allocator staging actually reaches the wheel. Run this before spending a boot. |
| `test_flood.py` | drives the kernel across the smem-capacity boundary (candidate-flood hypothesis). |
| `test_prodgeom.py` | drives the kernel at geometries measured from a real run (odd sk, varying windows). |
| `runs/{audit,stage}/` | results: `SUMMARY.txt`, probe jsonl/log. |
| `results_flood_sweep.txt`, `results_prodgeom.txt` | local sweep transcripts. |

## The detector, and why it has a control

The wheel allocates `output_indices_torch = torch.empty(num_rows, top_k,
int32)` (`indexer_top_k_decode_varlen.py:684`) and never pre-fills it. It is the
first real allocation after the compile-cache lookup, so the harness fills a
block of exactly that shape with a sentinel and frees it immediately before the
call: PyTorch's caching allocator hands that block straight to the wheel. Any
slot still holding the sentinel afterwards is a slot the kernel never wrote.

Sentinel 200003 is chosen to be larger than every observed `sk` (so its presence
is unambiguous) and below 262144 (so it also mimics the in-range garbage the
whole-GPU poison in the spec would have produced).

**The control is not optional.** Per shape the harness does stage → free →
`torch.empty` of the wheel's exact shape → measure sentinel fraction, and logs
`staging control OK` or `STAGING CONTROL FAILED`. Without it, zero unwritten
slots is ambiguous between "kernel is correct" and "our block was never handed
over". Measured 1.0000 on all 91 in-situ shapes and all 48 local cases,
including under `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Re-running

Local kernel drivers (seconds, one GPU, no trainer — do this first):

```bash
source /root/.cache/user_artifacts/env.sh
cd /root/.cache/user_artifacts/lps1003/probe2
V=/root/.cache/user_artifacts/trainers_main/server/.venv/bin/python
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  NVTE_FRAMEWORK=pytorch $V test_staging.py     # control + one shape
... $V test_flood.py --sweep                    # capacity boundary
... $V test_prodgeom.py                         # measured prod geometries
```

In-situ boot (~20 min to READY), from the devbox **leader**:

```bash
bash /root/.cache/user_artifacts/lps1003/probe2/boot_probe2.sh stage
bash /root/.cache/user_artifacts/.devbox_up/wait_trainer_health.sh   # background, rerun per cycle
```

Then the batch-0 probe — note the FastAPI server runs on **global rank 0**,
which is the Slurm NODEID-0 node, not necessarily the leader pod:

```bash
cd /root/.cache/user_artifacts/lps1003
$V probe_nll.py --mode batches --bundle train_bundle_0_31.jsonl.gz \
    --batches 0 --repeats 4 --out probe2/runs/stage/probe_stage.jsonl --tag stage
python3 probe2/audit_summary.py "$(cat probe2/runs/stage/latest_audit_dir)"
bash /root/.cache/user_artifacts/.devbox_up/stop_trainer.sh && squeue   # always verify
```

Baseline gate: batch-0 `mean_nll` must land in 0.760–0.771 with the harness on,
or the audit is perturbing and nothing gathered is trustworthy.

## Prod use

`audit` mode is the prod-safe one (no extra allocation; adds GPU→CPU syncs
only). It gives the spec's E1 dump metrics: out-of-window indices, duplicates
within a row, negative-sentinel counts, per-call shapes and windows. `stage`
mode additionally turns an under-write into a counter, at the cost of one
transient (rows × k) int32 allocation per call.

Env: `BT_DSA_AUDIT=1` (required), `BT_DSA_STAGE=1`, `BT_DSA_FIX=1`,
`BT_DSA_STAGE_VAL`, `BT_DSA_DUP_EVERY`, `BT_DSA_SCORES_EVERY`,
`BT_DSA_AUDIT_DIR`, `BT_DSA_HB`, `BT_DSA_ECHO_CALLS`. Anomalies always print
regardless of sampling.
