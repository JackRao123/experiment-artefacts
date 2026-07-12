# sampler_4vs8 — TRN-1488 harness

Measures the NVFP4 Ultra sampler golden recipe at TP=8 (8×B200) vs TP=4
(4×B200) on one node. Results land in `../profiling_4vs8_gpus.md`.

## Layout

- `bench.py` — one (TP, ctx) cell: boot with the production engine recipe,
  introspect KV capacity, run a saturating synthetic generate workload
  (auto-sized to ~1.05× the KV-pool-implied concurrency, capped at 1000).
- `run_cell.sh` — box-side wrapper: CUDA_VISIBLE_DEVICES, patch-gate env
  (`_BASETEN_SERVED_MODEL` + `MODEL_PATH`), 1 Hz nvidia-smi poller over all
  8 GPUs, tee'd log, then `parse_cell.py`.
- `parse_cell.py` — pulls the authoritative vLLM boot-log numbers (KV pool,
  max concurrency, model-loading time/size), peak Running/KV-usage%,
  preemptions, patch/CUTLASS proof lines, per-GPU peak memory from the poller.
- `run_sweep.sh` — all 8 cells (TP 8→4 × ctx 8k/32k/131k/256k).

## Run

```bash
# copy this dir to the devbox, then:
ssh tj-<job>
cd /root/.cache/user_artifacts/sampler_4vs8
./run_sweep.sh                 # full sweep
./run_cell.sh 4 262144         # single cell (the E4 fit question)
./run_cell.sh 8 8192 0 --boot-only   # capacity probe only
```

Verify per cell in `out/<tag>.summary.json`:
`log.nemotron_patches_applied == true` and `log.cutlass_moe_selected == true`,
else the run did not exercise the production path and its numbers are void.

Adapter: uses the GSM8K-RL Ultra adapter
(`user_artifacts/rl/gsm8k-ultra-weights/sampler_weights/step-199`) when the
cluster has it; otherwise runs without a LoRA request (LoRA buffers still
allocated via `enable_lora`, matching golden-config memory). The summary
records which.
