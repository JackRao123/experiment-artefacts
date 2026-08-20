# R2_SEAM_PROBE — py-spy probe protocol for the step-seam untraced python

**Author:** jacobi (ex-kepler) · **Date:** 2026-08-14 · **For:** bayes / lovelace (box schedule)
**Status:** spec; runs as a micro-window (~10 min) on lovelace's schedule AFTER
W2's canary+parity legs. Diagnostic only — **probed steps are excluded from
perf reads** (py-spy overhead, ~1–3% at the spec'd rate).

## 1. What we're probing (the evidence to date)

The step seam (last backward kernel → next-step first forward) is the last
big idle class standing after W1c:

| trace | seam idle | the holes (untraced python, GPU-empty) |
|---|---:|---|
| fe127 (fixed wheel, pre-B/F) | 1.93 s ≥1 ms / 2.04 s wall | 1294 + 414 + 116 ms |
| w1c off-arm (tonight) | 1.80 s | 1224 + 326 + 110 ms |
| w1c on-arm (B/F armed) | **2.66 s** | **1247 + 828 + 362 + 122 ms** |

Host activity inside the holes: only `ProfilerStep#0` + sub-ms `aten::to` — no
traced torch work, no dataloader-worker aten ops on any thread. The trace path
is slimmed (no python stacks), so the content is invisible to the kineto
artifact. Seam bookends (fe127, on-arm same shape): grad RS → grad-norm
broadcasts → 18 ms metrics DtoH → holes → HtoD scalars → FusedAdam.step
(28.8 ms host / 1.0 ms GPU — tiny) → bf16 copy-back storm → AllReduce_u64
(15 ms) → final broadcasts. **The optimizer is not the hole.**

The on-arm residual idle is seam-dominated: **2.66 s of 4.07 s (65%) = 2.2% of
the new 119.85 s step.** R2 is the top residual idle lever.

## 2. The question B/F's seam growth poses (pre-registered hypotheses)

Seam idle grew 1.80 → 2.66 s (+0.87 s, absolute — not a shorter-step
artifact). Two live hypotheses:

- **H1 — exposure shift (my read, favored):** the seam python is ~constant
  wall work (~2.6–2.7 s); pre-B/F the launch-bound host was *behind*, so
  ~0.9 s of the python ran while the GPU was still draining the backward
  backlog and didn't show as idle. Post-B/F the host is caught up; the same
  python runs against an empty queue and shows in full.
- **H2 — new work:** B/F's caches (layout carrier, RoPE FIFO-64) or
  cache-adjacent bookkeeping added real seam work.

**Discriminator (pre-registered):** the probe reads the on-arm seam's
function-level composition. If the samples show only the old suspects
(optimizer glue / logging / next-batch / NCCL host wait) with **no
cache-maintenance functions above noise (~2%)**, H2 is refuted and H1 stands
by elimination (the caches are dict ops, µs-scale — if they're visible at all
they're small). If cache functions appear materially, H2 lives and B/F needs a
seam-maintenance look. (An off-arm py-spy leg would be the direct control;
not required for the elimination read, and not worth a boot by itself.)

## 3. Protocol

### 3.0 Install (once, isolated — never the trainer venv)

`uv tool install py-spy` (or `pip install py-spy` into a scratch venv) on both
nodes. Verify: `py-spy --version`. ptrace: the trainer pods have allowed
py-spy-class attach before (the campaign's profiling runs attach routinely);
if attach fails twice → STOP condition S1.

### 3.1 Targets

- **rank 0 (node 0) and rank 8 (node 1)** — the trace ranks, one per PP stage.
  Each rank is its own python process; identify by the rank env on the
  trainer command line (`pgrep -af 'dp_worker\|trainer'`, pick the process
  whose env/cmdline carries RANK=0 on node 0 / RANK=8 on node 1; the devbox
  `*trainer*` scripts know the process names — driver confirms at attach).
- `--threads` ON: the seam's python is expected main-thread, but the autograd
  thread and dataloader workers must be visible for the elimination read.

### 3.2 Window and alignment (event-driven, wait-rule compliant)

- **Record 3 consecutive d16 steps** at 250 Hz:
  `py-spy record -p <pid> -r 250 -d 420 -f speedscope -o /tmp/r2_seam_rank{N}.speedscope --threads`
  (420 s ≈ 3.5 steps at ~120 s/step — covers 3 full seams even with step
  jitter). One attach per rank, started any time; no synchronization needed.
- **Alignment:** the trainer's per-step log lines (trainer_srun.log,
  timestamped) mark step boundaries; the seam windows for analysis are
  [step-N end line … step-N+1 first compute]. Cross-check: the seam is the
  per-step GPU-util dip (nvidia-smi dmon -s u, 1 s cadence, on for the
  window). Both alignment sources are wall-clock; py-spy samples are
  wall-clock. No log-tailing by a session — the driver's existing watcher
  pattern (nohup, completion-line watch, message on fire) is the model.
- No trainer-state blocking waits anywhere in this protocol; the record runs
  detached and its output file is the artifact.

### 3.3 Expected artifacts (pull Mac-side immediately after)

1. `r2_seam_rank0.speedscope` / `r2_seam_rank8.speedscope` (→
   `pp2cp8ep8/traces/` or the run folder; small, MBs).
2. The window's trainer_srun.log slice (step-boundary alignment).
3. Analysis output (mine): seam-aligned stack histogram per rank — samples
   inside seam windows grouped by root→leaf function, % of seam samples per
   function family: (i) optimizer glue, (ii) logging/telemetry (wandb/TB/
   print/metrics formatting), (iii) next-batch/dataloader, (iv) NCCL/comm
   host wait, (v) cache/maintenance, (vi) other.
4. One-paragraph verdict in the NOTEBOOK + this doc's results section.

### 3.4 Pre-registered reads and decision rules

- **R2-Q1 (composition):** % of seam samples per family, per rank. The
  dominant family names the fix:
  - logging/telemetry dominant → async/deferred logging (format + emit off
    the critical path); est. win up to ~1.5–2 s/step.
  - next-batch/dataloader dominant → prefetch overlap (start the fetch during
    backward); est. win ~1–1.3 s (the ~1.25 s hole class).
  - optimizer glue dominant → slim the glue (precompute clip coeff, fuse the
    copy storm); est. win ~0.5–1 s.
  - NCCL host wait dominant → not python at all — route to the contract-shim
    lane (lever 1) and re-scope R2.
- **R2-Q2 (H1 vs H2):** cache/maintenance family <2% of seam samples → H2
  refuted, H1 (exposure shift) stands → the seam prize is *real new work to
  overlap/remove*, not B/F bookkeeping. ≥2% → look directly at the cache
  functions named.
- **EV bound (honesty):** total seam prize ≤ 2.66 s/step (2.2%); expect
  1–2 s recoverable. This is a small, clean lever — size the follow-up work
  accordingly (a day of glue/logging surgery, not a campaign).

### 3.5 STOP conditions

- **S1:** py-spy can't attach after 2 attempts (ptrace perms, pid churn) →
  stop, report the blocker, do NOT improvise a different blocking profiler;
  the fallback is the env-gated `with_stack=True` one-step trace (heavier
  boot; bayes's call whether to spend it).
- **S2:** the probed boot shows any canary/health regression during the window
  (py-spy is observation-only, so none is expected; if the trainer errors
  anyway, the window is cursed — stop, keep the partial captures).
- **S3:** capture cap — 420 s per rank, one retry max if a window missed the
  seam (step-boundary misalignment). Two misses → stop and analyze what
  exists; the holes are ~2 s wide at 250 Hz, a miss means the alignment
  failed, not that the seam vanished.
- **S4:** any pressure to extend the probe into a perf-measurement role →
  refuse; probed steps are contaminated and excluded from perf reads.

## 4. Scheduling + dependencies

- **Slot (bayes, 2026-08-14, revised):** rides the **P4 SOAK boots** — the
  mission + B/F ship stack, which is exactly the config whose residual seam
  matters for the report (on-arm = the prize). (Earlier slot "post-W2
  canary/parity" is void: W2 is blocked upstream, route b.) Same spec, same
  ~10-min micro-window (install 2, attach 1, record 7, pull 1); lovelace
  informed. No dedicated boot. No config changes. If a gates-off soak boot
  is what runs, the composition read (R2-Q1) still stands and H1/H2 gets its
  direct control — either arm answers the questions; on-arm preferred.
- Dependencies: py-spy installability box-side (S1), the driver knowing the
  rank→pid mapping (§3.1), Mac-side pull path (standing).
- Trace servers :9002/:9004/:9005/:9006–9009 stay up Mac-side for the report
  (bayes's instruction).

## 5. Results (2026-08-14 ~21:0x UTC — probe ran in the S1 soak, job wprm693)

Capture: py-spy 0.4.2, 420 s @ 250 Hz `--threads`, rank0 (pid 99478, node 0) +
rank8 (pid 79110, node 1), ~3.5 d16 steps mid-soak (soak healthy, 1121–1128
t/s/GPU). Artifacts: `pp2cp8ep8/r2_seam_probe/r2_seam_rank{0,8}.speedscope`
(sha256 in `r2_probe.sha256`). Seam anchors: the per-step optimizer-step
frames (`step (optimizer.py:1739)` et al.), periodic at the step boundary;
narrative below is the seam window around a mid-window anchor (both ranks
showed the same structure on every anchor).

### R2-Q1 — the seam's host content, named

**rank0 (stage 0), loop thread = `asyncio_0`:**
| content | call site | seam time |
|---|---|---:|
| **full-device `torch.cuda.synchronize`** | `pipeline_parallel/p2p_communication.py:263` (`_communicate_shapes`) | ~4.1 s block |
| MoE dispatcher DtoH sync | `token_dispatcher.py:1611` (`_maybe_dtoh_and_synchronize`) | ~1.5 s |
| THD next-microbatch packing | `thd_cp.py:390` (`_row` ← `pack_thd_cp_microbatch` ← `packer.py:144`) | ~0.45 s |
| object broadcast (small here) | `distributed_c10d.py` ← `dp_worker/worker.py` | ~0.2 s |

**rank8 (stage 1), loop thread = `MainThread` (`_peer_loop`, worker.py:311):**
| content | call site | seam time |
|---|---|---:|
| **`broadcast_object_list` (python-object broadcast, per step)** | `distributed_c10d.py:3839` ← `c10d_logger` wrapper ← `_peer_loop` (worker.py:311) | **~7.1 s block** |
| THD next-microbatch packing | same `pack_thd_cp_microbatch` path | ~0.65 s |
| shape sync | `_communicate_shapes` | ~0.3 s |

Mapping to the kineto idle holes (honest): the GPU-idle portion = the
pure-python segments (THD packing + broadcast pickling/rendezvous + optimizer
glue) — invisible to kineto because pure python emits no aten/cuda slices
(only the sub-ms `aten::to` seen in the holes). The `torch.cuda.synchronize`
block and the broadcast's rendezvous-wait portion are host-blocked-on-drain /
host-blocked-on-peers (GPU not necessarily idle) — they are seam-*adjacent*
costs, not all pure idle. The attackable pure-idle slack ≈ the kineto-measured
2.66 s; the named python content accounts for it.

### R2-Q2 — H1 vs H2 (the pre-registered discriminator)

**Cache/maintenance functions: 0% of seam samples on both ranks** (not present
above noise). **H2 (B/F added cache work) REFUTED; H1 (exposure shift)
CONFIRMED** — the seam content is the same old suspects (device sync,
per-step object broadcast, THD packing, dispatcher sync); B/F added nothing.
Per the ratified framing: **B/F removed the launch backlog that hid this
python; it did not grow the seam.**

### Decision-rule outcome (per §3.4)

Dominant families: **logging/coordination broadcast (rank8, ~7.1 s)** and
**pipeline-schedule full-device sync (rank0, ~4.1 s)**, then dispatcher DtoH
sync (1.5 s) and THD data packing (~0.5–0.65 s). Fixes by the pre-registered
rules:
1. **The per-step `broadcast_object_list` (worker.py:311) → async/deferred or
   shrunk.** Caution: its 7.1 s is likely rendezvous-dominated (all 16 ranks
   must arrive; the slowest — stage 1, late from the drain skew — sets it),
   so part of it *is* the boundary skew wearing a broadcast. Async emission
   (overlap with the next step's start) is the right-shaped fix; shrinking
   the object helps only the pickle share. Est. ~1–1.5 s of the pure-idle
   class.
2. **`_communicate_shapes`' full-device sync → narrow the scope** (a
   stream-scoped or event-scoped wait), or skip shape exchange entirely for
   static-shape runs (the shape is constant at fixed 131k every step — the
   per-step full-device sync is paying a dynamic-shape cost in a static-shape
   regime). Also seen in-step (per-microbatch; ~20% of the loop thread's
   active time window-wide — worth its own look beyond the seam).
3. **THD packing (~0.5–0.65 s, pure python) → prefetch/overlap** with the
   backward tail.
4. Optimizer glue: NOT the hole (confirmed again — the optimizer-step frames
   are ~0.2 s class).

**EV bound (restated honestly):** total seam prize ≤ 2.66 s/step (2.2% of the
post-W1c step); expect ~1–2 s recoverable from items 1+3. Item 2 is bigger
but is in-step territory, not the seam.

### STOP-condition accounting

S1 (attach failure) — not triggered (attached first try, both nodes).
S2 (regression during window) — none observed (soak held 1121–1128 t/s/GPU).
S3 (capture cap) — respected (420 s, one window, no retry needed).
S4 (perf-role creep) — probed steps excluded from perf reads; soak's
steady-state read undisturbed (lovelace's numbers stand).
