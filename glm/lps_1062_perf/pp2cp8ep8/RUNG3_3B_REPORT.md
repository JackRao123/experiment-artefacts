# Rung 3, arm 3b — the 131k offload arm: results

turing, 2026-08-20 (night). Box **qed7z1w** (2 nodes x 8 B300, `ali`).
Config `trainer_pp2cp8ep8_131k_selective_offload_moe_act_attn_proj.json`
(selective recompute + offload of `moe_act` and `attn_proj`), d2, 131k.

Framing rules carried from rung 1: one tree, no TF32 anywhere — that is the
environment, not a caveat. Absolute tokens/sec figures are context only and
are **not** comparable to any historical band. Selective recompute and the
offload are ONE package; nothing here may be read as "recompute gets X and
offload adds Y".

Two attempts are reported. Attempt 1 (valve uncapped, the pre-registered
default) **ran out of memory** and produced no throughput number. Attempt 2
(valve capped) is the fix and is reported below it.

---

## 0. The headline: the arm does not fit at 131k/d2 with the valve uncapped

Attempt 1 booted cleanly, passed all three boot checks, engaged the offload
for real — and then ran out of GPU memory on the first full-size step, before
any timed window. It never reached a memory plateau, so **the pre-registered
glue row could not be computed from it** (see §4 for what can and cannot be
salvaged).

The cause is a specific, fixable interaction, not a wrong memory estimate:

> With the backpressure valve uncapped, the main compute stream never waits on
> a device-to-host copy event. Every tensor that has been handed to the
> offload keeps its **device** block un-reusable until that copy actually
> finishes. Nothing bounds how far behind the copy stream may fall, so the
> count of "already offloaded, still copying, still occupying GPU memory"
> blocks grows without limit inside a step.

Mechanically: `bulk_offload_group` calls
`tensor_on_device.record_stream(self.d2h_stream)` after queueing the copy.
`record_stream` tells the caching allocator it may not hand that block to
anyone else until the copy stream has passed that point. The only thing that
makes the compute stream wait for the copy stream is the valve
(`_drain_offload_pending`), and the valve is inert when
`max_inflight_offloads is None`. So uncapped, the offload frees host-side
pressure while **retaining** device-side pressure.

## 1. The boot checks all passed — the offload really was engaged

This matters because the worst failure this ladder can produce is the baseline
wearing the arm's label. It did not happen here.

| check | result |
|---|---|
| 1. NVTE import-time latch | `env=1 latch=True -> OK` on all 16 ranks |
| 2. hook engagement | `fine_grained_activation_offloading=True`, `offload_modules=['moe_act','attn_proj']`, `self_attention.offload_attn_proj=True`, first-MoE-layer `experts.offload_moe_act=True` (layer 4 on the first stage, layer 39 on the last) |
| 3. NUMA-local pinned buffers | all 16 GPUs resolved (`cuda:0-3` -> node 0, `cuda:4-7` -> node 1, 128 local CPUs each); every `NUMA_VERIFY` line reported page nodes matching the expected node |

And a fourth, stronger piece of evidence that bytes actually moved, which the
three checks do not cover: **482 distinct pinned host destination pools were
allocated, totalling 279.3 GiB across the 16 ranks — about 17.5 GiB per
rank**, in sizes from 32 MiB to 1.86 GiB.

That number is load-bearing because of where the allocation happens. A pinned
pool is only created inside `OffloadTensorPool.allocate()`, which is only
reached from `offload()`, which allocates the host destination and then
immediately performs `copy_dst.copy_(src_tensor, non_blocking=True)` — the
real device-to-host copy. One `NUMA_VERIFY` line is emitted once per new
(shape, dtype) pool. So **every one of those 482 lines sits downstream of a
real copy.** The offload was not idle.

## 2. The instrument that should have caught this was blind by construction

The valve telemetry printed, on every rank:

```
BT_OFFLOAD_VALVE_TELEMETRY iter=2 max_inflight_offloads=None | (no offload commits recorded)
```

Read literally that says no tensor was ever offloaded, which — per §1 — is
false. The counters were dead:

```python
if self._max_inflight_offloads is not None:      # <- the whole block, including
    gname = group_to_offload._name               #    the counters, was nested here
    self._offload_pending_by_name[gname].append(group_to_offload._offload_event)
    self._drain_offload_pending(gname)
    if self._valve_telemetry_enabled:
        _st = self._valve_stats[gname]
        _st[0] += 1                              # commits
        _st[3] = max(_st[3], len(...))           # max_pending
```

`commits` and `max_pending` only incremented when a cap was set. The
pre-registration deliberately ran **both arms with the valve uncapped**
("tuning the valve in the same run that introduces the arm would break
one-variable-at-a-time") and then asked us to "read the telemetry (commits,
drain firings, events drained, max pending per group) and only tune if the
numbers demand it". Those two instructions are mutually unsatisfiable: in the
uncapped configuration the telemetry cannot produce a number, so the
pre-registered criterion "valve counters report sane values" was unevaluable
in the only configuration the arms were specified to use.

Worse, `max_pending` is precisely the quantity that measures the failure in
§0, and uncapped is precisely the configuration where it matters — with a cap
set, the depth is bounded by the cap by definition.

**Fixed** in `basetenlabs/Megatron-LM` @ `2c3d720cf`: enqueue and count on
every commit, capped or not. `_drain_offload_pending` already returns
immediately when uncapped, so capped behaviour is byte-for-byte unchanged, and
the pending deque is cleared per iteration in `reset()`. Vendored up through
`basetenlabs/Megatron-Bridge` @ `f95828b6` to trainers PR #1074 @ `1764a816`.

## 3. What actually failed, in order

Boot dispatched 22:55:51Z, healthy at 23:15Z (~19 min weight load). Then
`rung3_arm.sh 3b`: three boot checks passed, and the run died in the parity
leg — step 4 of 11 — so **no timed window, no plateau read, no valve
counters** were produced.

The failing operation was a `ForwardBackwardOp` over the fixed parity datum
set: 9 datums, **262,032 real tokens**, which pack into 3 partitions of 131k.
Rank 0's node lost five of its eight GPUs to out-of-memory within the same
step; rank 0 exited first (exit code 1) and the remaining 15 ranks were
SIGTERMed as a consequence. Representative error, GPU 0:

```
Tried to allocate 854.00 MiB. GPU 0 has a total capacity of 267.69 GiB
of which 461.19 MiB is free. ... this process has 266.22 GiB memory in use.
Of the allocated memory 258.40 GiB is allocated by PyTorch, and 978.28 MiB
is reserved by PyTorch but unallocated.
```

978 MiB reserved-but-unallocated against a 854 MiB request rules out
fragmentation as the story: this was a genuine 258 GiB of live allocation.

**The timed run would have failed the same way.** It is easy to assume the
parity leg is an unusually heavy shape and the d2 timed run would have been
lighter. It is not: the timed run is `--datums 2 --seq-len 131072` =
**262,144 tokens per step**, against the parity leg's 262,032 real tokens.
Same token volume to within 0.04%. Nothing was going to fit.

## 4. What this does and does not tell us about the glue row

The pre-registered plan was `glue = (peak_3b - base - prefetch) / sets`, with
`base` = 152.5 GiB (rank 0, nvidia-smi reserved at plateau), `prefetch` = 5
GiB, `sets` = 70 MoE layers on rank 0. **That cannot be evaluated here**,
for two independent reasons, and it is worth being precise about both rather
than reporting a number.

**First, there is no plateau.** The run OOMed while still growing, so 266.22
GiB is a *lower bound on demand*, not a peak. Substituting it gives a lower
bound on the per-set cost, and the answer depends entirely on an assumption
the pre-registered formula never stated:

| activation sets assumed live at once | implied lower bound, GiB per set |
|---|---|
| 70 (one microbatch's worth — what the formula's `sets = 70` implies) | **>= 1.55** |
| 140 (two microbatches' worth) | **>= 0.78** |

Under PP2 with a 1F1B pipeline schedule, the first stage holds up to two
microbatches in flight, not one — so 140 is the physically right denominator
at d2, and the projection's `sets = 70` silently costed a single microbatch.
On that reading the plan's inherited glue band (0.64-1.14 GiB/set, mid 0.89)
was *approximately right per set* and the projection was wrong by a factor of
about two in the number of sets. That single correction moves rank 0's
predicted requirement from `152.5 + 70x0.78 + 5 = 212 GiB` ("clears") to
`152.5 + 140x0.78 + 5 = 267 GiB` (over the 247.7 GiB effective ceiling), which
is what we observed.

**Second, and more importantly, the two terms are confounded in this run.**
Part of that 113.7 GiB above base is genuine resident glue, and part is the
unbounded pile of offloaded-but-still-copying blocks from §0. Attempt 1 cannot
separate them, so quoting any single glue figure from it would be
overclaiming. Capping the valve is what un-confounds them: with a cap of `N`
the copy-in-flight term is bounded by roughly `N x (bytes per offload group) x
(number of group names)` instead of being unbounded, and the residue is glue.

So the glue row stays open, and attempt 2 is the run that can close it.

## 5. Attempt 2 — the same arm with the valve capped

One variable changed against attempt 1:
`activation_offload.max_inflight_offloads: 4`, in
`trainer_pp2cp8ep8_131k_selective_offload_moe_act_attn_proj_valve4.json`
(diff against the attempt-1 config is exactly that one key). Everything else —
tree, boot environment, module list, recompute granularity, d2 shape — is
identical.

Why 4, stated before the run: the cap bounds, per offload group name, how many
device-to-host copies may be in flight before the main stream waits on the
oldest. The group names here are the module kinds, `moe_act` and `attn_proj`,
so uncapped the depth per name reaches roughly 140 at d2 (70 MoE layers x 2
in-flight microbatches). A cap of 4 bounds it to 4 per name, i.e. 8 groups of
copy-in-flight device memory instead of ~280. The valve blocks the compute
stream rather than skipping work, so an undersized cap costs throughput and
never memory — the safe direction — and the now-working telemetry
(`drain_firings`, `events_drained`) is what will say whether 4 is leaving
throughput on the table.

Pre-boot gates run, all green: `offload_preflight.py` on an idle GPU in the
worker venv against the patched tree (all six checks, including a bit-exact
128 MiB device-to-host round trip); config validated through the real
`TrainerControllerConfig` schema; all 16 GPUs confirmed at 0 MiB before
dispatch.

That last one was not a formality. After attempt 1 died, three orphaned
`multiprocessing.spawn` children (parent PID 1) survived on GPUs holding 273,
273 and 121 GiB, with `nvidia-smi` attributing only 618 MiB each to a live
process. Booting on top of that would have produced an out-of-memory failure
with a completely misleading cause.

### Results

*(pending — attempt 2 in flight)*

---

## 6. Corrections to the surrounding documents

1. **"Forward-only parity leg" is a misnomer** and it should stop propagating.
   `rung3_arm.sh` and `rung1_parity_and_smoke.sh` both invoke
   `parity_driver.py` **without** `--forward-only`, so both legs go through
   `/forward_backward` and run a real backward pass; grads accumulate and are
   never applied. What the legs actually guarantee is that **the weights never
   move** (no `optim_step`), which is the property the comparison relies on.
   The two legs are therefore comparable to each other — this is not a
   harness mismatch — but the parity leg is a full-size training step for
   memory purposes, which is why it, and not some later timed window, is where
   attempt 1 died.
2. **The memory projection needs an in-flight-microbatch factor.** As written,
   `base + sets x glue + prefetch` with `sets` = the stage's layer count costs
   one microbatch. At PP2/d2 the first stage holds two. Every row of the
   attempt-1 projection in `RUNG3_PREREG.md` and `RUNG1_REPORT.md` §5 is
   therefore optimistic by close to a factor of two on the activation term.
3. **`rung3_arm.sh` boot check 3 prints a false failure.** It runs
   `grep ... | head -4`, so `grep` dies on SIGPIPE with a non-zero status and
   the `|| echo "(no NUMA verify lines yet)"` fallback fires *unconditionally*
   — directly after printing four genuine `NUMA_VERIFY` lines. Since check 3
   is pass/fail, that line can make a passing boot read as a failed one. Same
   bug on the `NUMA_BIND` line above it. Fix: `sed -n 1,4p` instead of
   `head -4`, or capture to a variable first.
4. **Archived trainer logs are named for the *next* dispatch, not their own.**
   `trainer_srun.20260820T225551Z.log` holds the run *before* the 22:55:51Z
   dispatch, complete with that run's SIGTERM and its NUMA-resolution failure.
   Reading it while the 22:55:51Z boot was healthy makes a healthy boot look
   dead and un-NUMA-bound. The live log is always the unstamped
   `trainer_srun.log`.

## 7. Evidence

On the box under `/root/.cache/user_artifacts/lps1062_pp2/`:
`logs/trainer_srun.20260820T*.log` (attempt 1's trainer log, archived at the
attempt-2 dispatch stamp), `logs/rung3_3b.log` (the arm script's own record,
including the three boot checks and the parity-leg traceback),
`logs/boot_env.log`.

Code: trainers PR #1074 branch `jackrao/lps-1062-actplace` @ `1764a816`;
`basetenlabs/Megatron-Bridge` @ `f95828b6`; `basetenlabs/Megatron-LM` @
`2c3d720cf`. Attempt 1 booted `bb027cc6` / `e0e7fe77` / `720d18ea9`.
