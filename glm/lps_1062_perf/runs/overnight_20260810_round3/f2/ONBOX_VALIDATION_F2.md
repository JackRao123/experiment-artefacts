# F2 on-box validation recipe — phantom partition fix (DP>1 THD-CP deadlock)

Box: any 2×8 B300 devbox (validated target: 318g61w-class, ali). Mesh under
test: **EP16/CP8/DP2** (expB config — the F2-relevant topology; EP16 spans
both DP replicas). Patch under test: `f2.patch` (server/src only) applied to
the shared clone on top of the B/F/A(+C) vendored tree — server/src and
vendored mcore are disjoint, so it stacks cleanly.

Apply:
```bash
cd /root/.cache/user_artifacts/trainers_main
git apply --check /root/.cache/user_artifacts/lps1062/f2.patch && git apply /root/.cache/user_artifacts/lps1062/f2.patch
git diff --stat   # expect: packing.py +megatron_controller.py +test file
```

All benches via `runs/overnight_20260809_mfu_sweep/run_bench2.sh`/`runs/overnight_20260809_mfu_sweep/bench_driver2.py` from the node where
`:8001/health` answers. Ship env on every launch (NCCL×3 + BT_TF32_LM_HEAD=1;
B/F gates per the round's matrix). **Verify env from /proc/<pid>/environ.**

---

## (a) Boot-deadlock repro → fix (the deterministic gate)

1. Config expB (EP16/CP8/DP2, max_seq_len=131072). `BT_SKIP_WARMUP` **unset**
   (default warmup sends exactly 1 datum → dp_rank1 packs 0 real partitions).
2. Boot with the patch, default env ⇒ server must reach READY / health OK.
   Pre-fix this hangs >35 min in warmup pass-1 (py-spy: node0 in
   token_dispatcher all_gather, node1 in finalize_model_grads).
3. Kill-switch control: reboot with `BT_F2_PHANTOM_PARTITIONS=0` ⇒ the hang
   MUST reproduce (proves the flag and the causal mechanism, not a fluke
   boot). Then reboot with it on for the remaining steps.

## (b) B-custmix heterogeneous shape (the empirical repro)

20 heterogeneous datums (10,240–63,488 tok, customer histogram, 530,432
tok/step; driver recipe in runs/overnight_20260809_mfu_sweep/results/B_custmix_f2/context_node0.txt).
Pre-fix: deadlocked mid-warmup (3 vs 2 partitions). With the patch: runs to
completion, per-window losses sane, no NCCL-spin (GPU util returns to idle
between windows; `nvidia-smi` not pegged at 100% with 0 progress).

## (c) Loss canary vs DP1 golden

Same total tokens, same seed-fixed synthetic data, same `--warmup-datums`:
- DP1 golden (expA131, EP16/CP16) reference window losses (CPFS-persisted
  A131-131k-d4*.json) vs DP2-with-phantoms run of the same shape.
- Bar: per-window |dloss| ≤ ~5e-3 (cross-config reduction-order tolerance),
  grad-norm series in the same band. Phantom partitions contribute exact
  [0,0], so any larger drift is a bug, not noise.

## (d) Perf non-regression

Equal-length DP2 131k×d4 bench (`--datums 4 --repeats 3 --warmup-datums 2`):
expect steady ≈ 745 tok/s/GPU (the pre-fix equal-length number, which only
ran because equal lengths happened to pack equal counts). Phantom overhead at
equal counts = one 4-byte all_reduce per step — unmeasurable.

---

## fibonacci review additions (blocking)

### 1. NaN×0 grad-corruption canary

Zero loss seeds give exact-zero grads ONLY if no phantom intermediate goes
inf/NaN (NaN×0 = NaN would silently corrupt accumulated real grads).

- **(1a) Phantom-only microbatch backward (GPU test):** on a booted trainer,
  submit a forward_backward whose per-replica datum counts differ by ≥1
  partition (e.g. DP2, 3 datums of 131k → rank0: 2 partitions, rank1: 1 +
  1 phantom). After the op, read back grad buffers on BOTH nodes and assert:
  all-finite (`torch.isfinite().all()` per buffer) — and for a phantom-ONLY
  op (kill real data: submit 1 datum at DP2 so rank1 is all-phantom), rank1's
  grads are exactly zero. Implementation: trainer debug endpoint
  (`/debug/...` grad dump if available) or a sidecar py-spy/torch script in
  the trainer venv attaching to the worker; simplest robust form: run one
  `forward_backward` + `optim_step`, then checkpoint and inspect the
  optimizer/grad state offline.
- **(1b) Phantoms-on vs phantoms-off checksum soak:** DP2, UNEVEN datum
  counts across replicas (phantoms fire on the short replica). Run the same
  real data twice: once with the patch default (phantoms on), once as DP1
  with the identical per-replica data (no phantoms possible). Compare grad
  buffers (or post-optim weights) per replica: must match to reduction-order
  tolerance; the no-phantom DP1 run is the ground truth that the phantom
  replica's real-data grads are unaffected.

### 2. DDP bucket / grad-sync ordering

`finalize_model_grads` must fire exactly once per op (post-loop, via the
controller's `_suppress_schedule_finalize`); the phantom suffix must not
re-enable any per-partition sync path.

- Instrument: wrap `finalize_model_grads` (or the saved finalize callable)
  with a counter/logging shim in the trainer venv's site-packages copy for
  one debug boot, run a mixed real+phantom step (uneven DP2 data), assert:
  finalize called exactly once per forward_backward op, and zero finalize
  calls from inside the per-partition loop. Also confirm
  `grad_sync`/`bucket` counters (if exposed via /debug or logs) are unchanged
  vs a no-phantom equal-length step.

---

## Sign-off matrix

| gate | pass condition |
|---|---|
| (a) boot | READY with default warmup, DP2; hang reproduces under kill-switch |
| (b) custmix | completes; sane losses; no NCCL-spin |
| (c) canary | |dloss| ≤ 5e-3 per window vs DP1 golden |
| (d) perf | steady ≈745 tok/s/GPU @131k-d4 DP2 |
| 1a/1b | grads finite; phantom-only grads exactly 0; on/off checksums match |
| 2 | finalize exactly once per op; no per-partition sync |
