# Activation placement — current state

Single entry point for this workstream. Updated 2026-08-20 (banach).

## Where the code is

**PR: basetenlabs/trainers#1074** (draft) — `jackrao/lps-1062-actplace`, based on
#1070's branch. Retarget to `main` when #1070 merges.

The vendored megatron-core changes are **real commits**, not patch files:
- `basetenlabs/Megatron-LM` @ `jackrao/lps-1062-activation-offload` — pinned-buffer
  pooling, in-process NUMA-local binding, boot page-placement verification,
  valve telemetry, the conditional `attn_proj` guard relaxation, the AbsorbedMLA
  `attn_proj` hook.
- `basetenlabs/Megatron-Bridge` @ `jackrao/lps-1062-activation-offload` — pointer bump.

Workflow: commit on the laptop, push, `git pull` on the box. Do not scp code.

## Status

Nothing has run on GPU hardware. Green so far, all CPU-only: `models` 165 passed;
`server-megatron-bridge` off-Linux subset 163 passed (3 pre-existing collection
errors from an unguarded `megatron` import); repo lint/format/typecheck; the pool
smoke test (`../tools/offload_pool_smoke_test.py`).

**Blocked on hardware.** The region is out of B300 capacity —
`FailedScheduling: Insufficient nvidia.com/gpu`, autoscaler cannot add nodes.
Three provisioning attempts produced no usable box. Queued jobs will start when
capacity frees. There is no fallback accelerator: B200's 180 GB is under our
projected ~200-227 GiB peak even after offload.

## Ladder

Rung 1 (131k census, gate #1) is the blocking measurement — it replaces the one
number in the design that was inferred by subtraction. Rung 2a (32k selective,
no offload) and 2c (32k bring-up) can run on a single node and are staged.
Rungs 3a/3b and the rung-5 matched d16 pair need two nodes.

Recipes and pre-registered bars: `RUNG1_CENSUS_RUNBOOK.md`.
Config inventory and the ladder map: `configs/README.md`.

## Findings that changed the design

- **Selective recompute and the offload are one package.** Selective alone
  projects to ~292 GiB on the binding stage against a ~248 GiB ceiling. There is
  no "recompute win without offload" fallback; the alternative is the block+K
  dial, a different mechanism.
- **NUMA-local pinning is a gate, not a tuning knob.** All-8 bidirectional,
  sustained, per GPU: local 27.5/28.8 GB/s (meets the ~22/24 requirement),
  interleaved 15.6/16.8 (**below** it — and interleaved is the unbound default),
  remote 7.4/7.8. Cross-socket penalty is 73% at 8 GPUs vs <1% at one, so the
  old single-GPU figure does not generalize. Full data:
  `results/ALL8_BIDI_OFFLOAD_BW_BENCH.md`.
- **`moe_combine` captures no bytes** on this path, so there is no combine arm.
  Its bytes move into glue, tightening the worst case from ~213 to ~227 GiB.
  Reasoning in `configs/README.md`.
- **`qkv_linear` is deferred.** Its tensor is shared with the core-attention
  checkpoint, so offloading it either double-stores or puts a host-to-device
  transfer inside the critical path of the one thing we chose to recompute.
- **Shared-indexer index lifetime is safe.** A sharing layer builds no indexer
  module at all, so it cannot recompute a wrong selection; the leader's indices
  live in a per-microbatch Python dict outside autograd, so recompute granularity
  cannot affect them. A missing entry raises loudly. Mechanism:
  `analysis/OFFLOAD_HOOK_CODE_MAP.md` §g.
- **`NVTE_CPU_OFFLOAD_V1=1` must be in the launcher env.** TE latches it at
  import while the validator reads it lazily, so setting it later passes
  validation while TE silently keeps the old path. Absent entirely = loud failure.

## Deliberately out of scope

Core-attention offload (later A/B; roughly doubles PCIe demand and does not fit
the phase seams). The TF32 LM-head change (Jack: not wanted; the rung-5
deliverable is the **ratio** from a matched pair on one tree, so TF32 cancels).

## Removed in the 2026-08-20 cleanup

The `bt_offload_*` patch files and their hunk-structured SPECs — superseded by
real commits on the forks. The combine ladder arm and its config. The
`moe_combine` vocabulary entry (added then reverted; history kept because the
revert records the finding). `MULTINODE_BOX_FAILURES_RETRACTED_DIAGNOSIS.md` is
kept but retracted — its causal claim was wrong; the real cause was GPU capacity.
