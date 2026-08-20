# BT_DSA_CACHE_CLEAR_MB — one-shot DSA compile-cache clear probe (gauss spec, 2026-08-13)

## Purpose

Empirically exonerate (or convict) the cuDNN DSA compile caches from the
box side, closing the cache class from both directions (the source hunt
refuted the coarse-key mechanism at design level; this probe does it
empirically). Rides the gate-split boot.

## Mechanism

Env-gated hook in the mcore schedule's `forward_step`, fired once per
microbatch before the forward: when the current microbatch index matches the
env list, clear every cuDNN DSA compile cache, forcing a fresh recompile for
that microbatch. If the slot's corruption pattern is UNCHANGED after a fresh
compile, the caches are exonerated empirically. If it vanishes/changes, the
cached plan was the defect site.

## Arm

```
BT_DSA_CACHE_CLEAR_MB=<comma-separated slot indices>   # e.g. "1" for the middle partition
```

Default unset = inert. One-time WARNING line per cleared slot lands in
trainer_srun.log.

## Cost

One cuDNN DSL recompile per cleared slot (minutes). Clear ONLY the target
slot (the larger-than-predecessor one; poincare's tables name it). Do not
clear slot 0 (its compile is the cache's origin — clearing it tests nothing).

## Read-out

Compare the corrupted slot's per-decile table vs the same slot's uncleared
reference run (poincare's framework). Unchanged ⇒ caches exonerated
empirically. Changed ⇒ the fresh compile differs from the cached one ⇒
conviction + the fix is per-shape revalidation in the wrapper.

## Patch

`patches/bt_dsa_cache_clear_mb_probe.patch` (below, also on disk). Applies to
the vendored mcore at 57efae08b (box tree — apply with the other box
patches). Hook point: `schedules.py:forward_step`, immediately after
`set_current_microbatch` (the only per-microbatch hook point with the slot
index in scope; covers warmup + steady + cooldown uniformly).
