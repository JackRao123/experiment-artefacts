# Rung 3 pre-registration — the 131k offload arms

Written BEFORE the arms boot (conway, 2026-08-20). Box qed7z1w, pin
`7eec3054`. Jack has deferred the 32k rungs (2a parity, 2c bring-up), so the
ladder here is: rung 1 census → 3a → 3b. Bars are fixed in this file first;
numbers land in the report.

## What each arm changes (one variable per arm)

| arm | config | delta vs the arm before it |
|---|---|---|
| rung 1 | `trainer_pp2cp8ep8_131k.json` | baseline: full recompute, no offload |
| 3a | `..._selective_offload_moe_act.json` | selective recompute **and** `moe_act` offload, together |
| 3b | `..._selective_offload_moe_act_attn_proj.json` | adds `attn_proj` to the offload module list |

3a moves two things at once and that is deliberate, not sloppiness: selective
recompute alone does not fit at 131k (~292 GiB projected against a ~248 GiB
effective ceiling), so there is no selective-only configuration to measure at
mission length. Selective recompute plus offload is ONE package. Nothing in
any report may be phrased as "the recompute change gets X and offload adds Y".

**3a is also a bring-up milestone, not a clean A/B point.** It is the first
configuration in this package that has ever run at 131k, and it first-boots
eight never-booted pieces at once (valve, pinned-pool fix, in-allocator NUMA
binding, page-placement verification, valve telemetry, the new module
vocabulary and validator, the unpermute hook, the projection hook). Treat a
first clean step as the result; budget for first-boot failures.

## Boot-time checks — all three are pass/fail, read before any timing

1. `activation-offload NVTE latch: env=1 latch=True -> OK`. **`latch=False` is
   a FAILED boot, not a slow one** — TE latched the variable at import before
   the environment set it, so TE is silently on its pre-V1 path while every
   other surface reads as engaged. Fix the launcher environment; do not debug
   TE. (This line exists only as of `7eec3054`; it was missing before.)
2. `activation-offload engagement:` — `fine_grained_activation_offloading=True`,
   `offload_modules` matching the arm, and the first-MoE-layer flags
   (`experts.offload_moe_act`, and for 3b `self_attention.offload_attn_proj`)
   reading True rather than None.
3. `BT_OFFLOAD_NUMA_BIND: GPU ... -> NUMA node N` and
   `BT_OFFLOAD_NUMA_VERIFY: pinned buffer ... page nodes ...` reporting
   **NUMA-local**. Unbound placement is the OS default and measured 15.6/16.8
   GB/s per GPU against a ~22/24 requirement — it runs, ~30% under the
   bandwidth the design needs, while looking healthy. Not NUMA-local = stop.

Arm boot environment (the first two default to silent — without them the arms
produce no valve data at all, and the "valve counters report sane values"
criterion cannot be evaluated):

```
NVTE_CPU_OFFLOAD_V1=1 BT_OFFLOAD_VALVE_TELEMETRY=1 BT_OFFLOAD_VALVE_TELEMETRY_EVERY=1 \
  bash stage_and_boot.sh <arm config>
```

`NVTE_CPU_OFFLOAD_V1` must be in the LAUNCHER environment — `stage_and_boot.sh`
exports it before `srun --export=ALL`, so it reaches the worker before TE
imports. `BT_OFFLOAD_NUMA_BIND` is left UNSET (defaults on); `=off` is the
opt-out A/B switch and is not used in the real arms.

## The valve stays at its default for both arms

Neither arm config sets `max_inflight_offloads`, so the backpressure valve is
uncapped. That is a deliberate pre-registered choice: tuning the valve in the
same run that introduces the arm would break one-variable-at-a-time. The valve
blocks the compute stream rather than skipping, so an undersized valve costs
throughput and never memory — the safe direction. Read the telemetry
(commits, drain firings, events drained, max pending per group) and only tune
if the numbers or the memory read demand it, as a separate run.

## Arm ORDER is decided by the census, not by the numbering

Counter-intuitively **3a is the memory-tightest arm, not 3b**. In 3a
`attn_proj` is neither recomputed (selective recompute keeps only core
attention) nor offloaded (the module list is `moe_act` alone), so it stays
resident: ~0.20 GiB per set, about **14 GiB on rank 0** at 70 sets and
**8 GiB on rank 8** at 40. 3b offloads it and gives that back. So the
projected deltas over the measured base are:

- 3a: `sets_r x (glue + attn_proj)`  ← the peak of the ladder
- 3b: `sets_r x glue`

Pre-registered rule, decided from the rung-1 census before either arm boots:

- Both projections clear the FAIL line (< ~227.7 GiB on both ranks) → run
  3a then 3b as numbered, one variable at a time.
- 3a projects at or over the line but 3b clears it → **run 3b first**, and
  treat 3a as a deliberately-skipped or after-the-fact point. Burning the
  first mission-length boot on the configuration most likely to OOM would
  waste the expensive slot and confound bring-up failures with memory
  failures. Say plainly in the report that the arms ran out of order and why.
- Both project over the line → neither arm boots as designed; the plan's
  contingency activates (largest glue tensors move into the offload bucket)
  and that is a design change to raise before spending box time.

## Memory bar

[BAND: TO BE WRITTEN FROM THE RUNG-1 CENSUS BEFORE 3a BOOTS.] Standing rule
from the campaign: the projected band goes in this file BEFORE the boot, never
after — that discipline caught a guaranteed-OOM boot earlier in this
workstream. Effective ceiling ≈ 247.7 GiB (267.7 cap, verified by CUDA on this
box, less the ~20 GiB cold-allocator burst). Read memory only at the plateau:
torch-reserved creeps ~+21 GiB over the first several steps and declares by
about step 9, so the read is poller-max over the last two control windows,
which must agree within ~2 GiB.

## Throughput bar

The deliverable is a **ratio** from a matched pair on one tree, not an
absolute. Report tokens/sec/GPU for baseline and arm from the untraced control
windows only (kineto costs ~5-8%), with mean and spread, and the ratio between
them. Absolute figures are context only and explicitly non-comparable; **no
comparison to the historical 984 or 1089-1103 bands**, which came from a
different tree and environment. The whole ladder runs one no-TF32 tree — that
is the environment, not a caveat.

Describe the win as ONE number. About 10 s of the ~39 s of recompute the
change is meant to recover is all-to-all traffic the recompute pass was
re-shipping; that is a component of the number, not an extra win on top.

## Parity bar — noise-relative only

The base path is intrinsically nondeterministic. Never import an absolute
1e-6/1e-3 bar, and do not reuse the campaign's 3.7-5.5 per-token floor: it
came from a different box and wheel. **The floor is the repeat pair measured
on THIS box** (rung 1 runs A and B, plus the two within-boot parity legs).

Each arm boot takes one forward-only parity leg on fresh weights, before
anything steps the optimizer. Statistics recorded: per-token max-abs
difference, share of tokens above 1e-3, and loss relative spread.

**Cross-boot floor — decision rule, pre-registered.** Arm parity legs are on a
different boot from the baseline's, so the honest comparison needs a
cross-boot noise cell, which costs one extra baseline boot (~15 min weight
load plus ~5 min). Rather than spend it unconditionally:
- Arm parity within the WITHIN-boot floor → pass, no extra boot. (Cross-boot
  noise is ≥ within-boot noise, so anything inside the tighter floor is inside
  the looser one too.)
- Arm parity outside the within-boot floor → the result is ambiguous, not a
  failure: run the §6b baseline boot to measure the cross-boot cell and
  adjudicate against it. Report the ambiguity either way.
- Arm parity far outside any plausible floor → stop and escalate. LPS-1063 was
  exactly this shape: silent mis-attention with no crash and no memory anomaly.

## What is out of scope

`offload_core_attention` stays OFF (phase 2, and conditionally alive at best —
it fits the sustained unidirectional bandwidth regime but not the phase
seams). No `qkv_linear` offload: its tensor is shared with the core-attention
checkpoint, so offloading it either double-stores or puts a host-to-device
transfer inside the critical path of the one thing we chose to recompute.
Never call `/save_state` (the async save path hangs under CP>1).
