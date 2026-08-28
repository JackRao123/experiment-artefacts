# GLM-5.2 debug-model all-rank runtime profiles

## Scope

- Devbox: `tj-w5y89m3`, two nodes x eight B300 GPUs (`NVIDIA L20D`).
- PR 1157 tip: `8209793c89ff6be2288b25668041bf27598bb0ce`.
- Current main / PR base at run time:
  `2e829d2bc6f162820188b7762c293053cdbc8cad`.
- The isolated devbox worktree was checked out detached at the PR tip. The
  shared golden checkout was not edited.
- Launch, readiness, and shutdown used only the generated `.devbox_up`
  lifecycle scripts.

## Cases

| case | model | topology | seq len | datums | GPUs | runtime ranks |
|---|---|---|---:|---:|---:|---|
| 1 | local GLM-5.2 0D1M debug proxy | TP1/PP1/CP16/EP16/ETP1/DP1 | 131072 | 1 | 16 | 0-15 |
| 2 | local GLM-5.2 0D1M debug proxy | TP1/PP1/CP8/EP8/ETP1/DP1 | 131072 | 1 | 8 | 0-7 |

Both cases preserve the earlier benchmark protocol: one untraced warmup,
three untraced controls, then one runtime-profiled forward/backward and
optimizer step. No memory profiling was enabled.

## Procedure

1. Confirmed both nodes were reachable, no Slurm jobs were active, all GPUs
   had zero utilization, the shared debug model existed, and local/remote disk
   space was sufficient.
2. Fetched PR 1157 and checked the devbox worktree out at exact head
   `8209793c`. Verified both node-local trainer venvs import the server code
   directly from that worktree.
3. Modified the copied driver only at runtime-profile start: it sends every
   rank in `range(num_gpus)`.
4. Launched Case 1 with `start_trainer.sh --num-nodes 2`, waited with
   `wait_trainer_health.sh`, and ran the driver with `--control-repeats 3
   --runtime-profile`.
5. Copied all 16 Case 1 traces before any later profile could prune them, then
   stopped the trainer with `stop_trainer.sh`.
6. Launched Case 2 with `start_trainer.sh --num-nodes 1`, repeated the same
   driver protocol for ranks 0-7, copied all eight traces, and stopped with the
   generated script.

## Results

| case | control tok/s/GPU | mean control FB | runtime-profile FB | trace count | trace bytes |
|---|---:|---:|---:|---:|---:|
| 1 CP16/EP16 | 7,621.1 | 1.0749 s | 1.7986 s | 16 | 128,111,715 |
| 2 CP8/EP8 | 11,022.7 | 1.4864 s | 2.0029 s | 8 | 49,342,799 |

Losses and gradient norms remained finite. The raw MFU/HFU fields are not
valid for this one-layer debug proxy because `mfu.py` models the full 78-layer
GLM-5.2 architecture.

## Verification

- Case 1 API returned 16 unique rank-qualified filenames covering ranks 0-15.
- Case 2 API returned eight unique rank-qualified filenames covering ranks 0-7.
- For both cases, the files copied to the laptop exactly match the API filename
  sets and combined `size_bytes` values.
- Both trainer allocations were stopped. Final Slurm and GPU compute-process
  queries were empty.
