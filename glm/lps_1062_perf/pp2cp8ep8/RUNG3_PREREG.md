# Rung 3 pre-registration — the 131k offload arms

Written BEFORE the arms boot (conway, 2026-08-20). Box qed7z1w, pin
`7eec3054`. Jack has deferred the 32k rungs (2a parity, 2c bring-up), so the
ladder here is: rung 1 census → 3b, at d2 only. 3a is cancelled (see below)
and no d16 pair is scheduled — Jack will decide that separately. Bars are
fixed in this file first; numbers land in the report.

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

## SCOPE CHANGE (Jack, 2026-08-20): 3a is CANCELLED as a planned rung

**Only 3b runs.** Jack's call, and it is the right one:

- **3a cannot ship.** The phase-1 configuration of record is 3b — selective
  recompute plus BOTH offloads. 3a is a waypoint that isolates the `moe_act`
  offload's contribution on its own; that attribution changes no decision we
  are going to take.
- **3a costs a full boot** (~20 min weight load plus the run) on a scarce
  box, and it is the configuration most likely to fail: it holds `attn_proj`
  resident, which is ~14 GiB more on rank 0 than 3b, putting it at or past
  the gate-#1 fail line and above the ceiling at the top of the glue band.
- Spending the expensive slot on a configuration that cannot ship and may OOM
  is a bad trade against spending it on the rung-5 matched d16 pair, which
  IS the headline deliverable.

**3a survives only as a CONDITIONAL DIAGNOSTIC.** Run it if and only if 3b's
throughput disappoints AND the copy-exposure trace implicates `attn_proj`
traffic — i.e. the hypothesis "offloading attn_proj costs more PCIe than it
saves" is live and worth one boot to test. Otherwise it never runs.

Everything below that reasons about arm order is superseded by this, and is
kept because it is what established that 3a is the memory-tightest arm.

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

## Instruments validated before the arms (conway, 2026-08-20)

Both analysis instruments were exercised on real data before any arm boots, so
rung 3 never has to debug its own tools mid-analysis.

- **`memory_census.py`** runs and parses real snapshots. Its smoke run on an
  older 131k snapshot immediately earned its keep: the replay peak read
  23.1 GiB against 123.8 GiB live at dump time, because the allocator event
  ring (server default 100,000 events) covered only 10.2 s — under half a d2
  step — so the per-bucket composition was window-local rather than a step's
  peak instant. The tool detects and warns about this. Fixed at the source:
  the driver now takes `--max-entries`, default 1,000,000 (~4 steps of
  headroom, ~126 MB per rank at ~0.13 KB/event).
  It also confirmed the size-alias hazard is real — on that snapshot the
  exact-size DSA topk match caught 2 tensors whose alloc sites were
  `gemm.py`/`graph.py`/`functional.py`, nothing to do with the indexer. **The
  printed alloc sites must be eyeballed on our data** (expect dsa.py/indexer
  sites; expected counts 22 on rank 0, 10 on rank 8).
- **`copy_exposure.py`** reproduces the reference trace exactly:
  **a2a 28.55 s, p2p 8.55 s** over a 132.5 s window, matching the plan's
  constants of record, with 66,126 background pinned device-to-host copies
  fully hidden (0.23 s resident, no copy >=100 us). On this no-offload trace
  exposed copy is 0.00 s and the tool correctly declares its host-sync
  cross-check vacuous. That 66k-copy background is the baseline the offload
  arms read their copy numbers as a DELTA against.

## Memory bar

**WRITTEN 2026-08-20 22:35Z, BEFORE ANY ARM BOOTS.**

Measured base (nvidia-smi reserved, plateau declared — last two control
windows identical on both nodes): **rank 0 = 152.5 GiB, rank 8 = 162.6 GiB.**

The glue row is NOT measured — see the rung-1 report: a full-recompute boot
cannot measure it, because under full recompute the eager per-layer sets are
not resident at all. So the band below uses the plan's own glue range
(0.64-1.14 GiB per MoE set, per AMENDMENT 1, which already absorbs the void
combine bytes) against the measured base:

| glue | 3a rank 0 | 3a rank 8 | 3b rank 0 | 3b rank 8 |
|---|---|---|---|---|
| 0.64 (low) | 216.3 clears | 201.2 clears | 202.3 clears | 193.2 clears |
| 0.89 (mid) | **233.8 FAIL** | 211.2 clears | 219.8 clears | 203.2 clears |
| 1.14 (high) | **251.3 — above the 247.7 ceiling, predicted OOM** | 221.2 clears | **237.3 FAIL** | 213.2 clears |

Rank 0 is the binding rank at every point in the band; rank 8 clears
throughout despite its higher base, because it holds 40 sets to rank 0's 70.

**Consequence — the arm-order rule above fires: 3b boots FIRST.** 3a is at or
past the fail line across most of the band and is predicted to OOM at the top
of it, while 3b clears at low and mid glue. Burning the expensive
first-mission-length slot on the arm most likely to OOM would confound
bring-up failures with memory failures.

**And 3b's measured peak MEASURES the glue row**, which is what gate #1
wanted and could not get: glue = (peak_3b - base - prefetch) / sets. That
number then decides whether 3a is worth a boot at all, from data rather than
from a 78%-wide inherited band. Standing rule
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
