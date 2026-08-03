# LPS-1003 Issue 2 — HANDOFF (rewritten 2026-08-02 PM; supersedes all prior versions)

> **RESOLVED 2026-08-02 (late PM session) — see `rearm/ARMING_MECHANISM.md`.**
> The rep0/rep1 question is answered, and the answer dissolves the framing
> below: there is NO driver anomaly and NO arming state. The unfixed
> wheel's `torch_stream_context` helper calls
> `torch.cuda.ExternalStream(int(handle))`; for handle 0 (legacy default
> stream) torch's constructor treats `stream_ptr=0` as ABSENT and returns
> a stream from torch's internal 32-stream pool — a different real stream
> every call (measured, cycles at 32). The -inf prefill (F1+F2) and the
> `out` allocation therefore run on a fresh private pool stream each call
> while the tvm-ffi launch (S) goes to the legacy stream: F2 and S are
> never ordered, on ANY call. Corruption manifests iff S starts before F2
> finishes — i.e., iff the legacy stream is shallow at submission. Cold
> windows (boot/empty_cache/rebuild) fire because their synchronous
> alloc/module-load storms collapse the host-ahead pipeline; warm calls
> heal because S sits behind milliseconds of backlog. Candidate A was a
> host-stall correlate, not a mechanism; B protects (first-call host
> latency), never arms; C/D/E moot. Standalone single-GPU lab
> (`rearm/arming/`): unordered-stream execution visible in CUPTI traces
> both ways (armed: F1→S→F2 mid-S; disarmed-by-busy-queue: F1→F2→S),
> allocator toggles all fire, busy-queue toggle protects, conn1 and fixed
> wheel 0/N. **NVIDIA driver report withdrawn** — replace with a
> cudnn-frontend wheel-glue report (60+ affected `_torch_stream_context`
> sites) and optionally a PyTorch footgun report for
> `ExternalStream(0)`. The fix (PR #875) is unaffected and remains
> validated.

## Mission

**Answer exactly why the corruption fires on rep0 (the first execution in a
cold window) and never on rep1 (the immediate warm re-execution of the same
call).** The race's mechanism, geometry, and fix are fully solved and
validated (see "Settled facts"). The ONE open question is the **arming
condition**: what cold-start state makes the driver mis-order two
already-submitted kernel launches — and hold the second one until the other
completes — when the very same submission pattern is handled correctly on
every warm call? Deliverable = the smallest identified driver-visible
trigger, a minimal repro, and trace evidence, packaged as NVIDIA-report
inputs.

This is a why-question, not a fix hunt. The fix (PR #875 dedicated launch
stream; +dsatopk4, revised to +dsatopk5 in the 08-02 review with a minimal
single-GPU repro + fails-without/passes-with test A/B —
PR875_REVIEW_0803.md) is correct and validated on the real recipe
(0 spikes / 100 steps vs 26/147 baseline, Fisher p=6.5e-7 —
`FIX_VALIDATION_100STEP.md`); it does not depend on this answer. Do not
touch the fix. Jack's standing rule applies to the diagnosis: **bisect,
don't guess** — every claim must be a measured toggle with predictions
written down before the run.

## Settled facts (do not re-derive; pointers)

Read first: `rearm/NIGHT_0801_FINDINGS.md` (root cause + correction banner)
and `rearm/DETERMINISM_MECHANISM_0801.md` (determinism mechanism, verified
08-01 PM; experiment scripts in `rearm/exp/`).

1. **Defect**: in `indexer_fwd` (vendored cudnn-frontend wheel,
   `cudnn/deepseek_sparse_attention/indexer_forward/_interface.py`), the
   `-inf` prefill `out.fill_(-inf)` and the tvm-ffi-dispatched CuTe kernel
   launch (S) lose mutual ordering when the caller is on the legacy
   default stream and `CUDA_DEVICE_MAX_CONNECTIONS` is unset; the prefill
   lands after S's stores and erases them.
2. **Determinism mechanism**: the prefill is TWO launches — the scores
   buffer ([15914, 60192] fp32 = 3.83 GB on cp_rank 0) exceeds
   TensorIterator's INT32_MAX-**byte** limit, so torch splits fill_ at
   exactly numel/2 → F1 = rows [0,7957) then F2 = rows [7957,15914)
   (torch 2.11.0 source + kineto + device-spy verified). Destroyer = F2
   executing after S. 0/2059 corpus mask events start below row 7957;
   cp_rank15 (5% under the byte limit → single launch) = 147/147 clean
   while rank14 (0.36% over) fired. Corrupt masks match the command
   sequence **F1 → S → F2 with full stream-serialization semantics**;
   the 8-row spared holes = S's TMA store flush tail.
3. **Exonerated** (dedicated A/Bs, 07-31/08-01): kernel compute, CLC
   scheduler, metadata staleness, TLB, pointer marshaling, cache hierarchy
   (volatile loads), CU_STREAM_LEGACY handle, top-k kernel (torch.topk swap
   still fired), NCCL delivery, and **SM-resource starvation** (measured
   100% co-residency of a fill-style kernel with a persistent all-SM
   kernel even at 200KB smem/CTA — `rearm/exp/exp3_starvation.py`).
4. **Stream-handle facts**: in-situ streamlog: torch_stream == 0 ==
   tvm-ffi env stream on every call; the inversion happens BELOW the
   stream-handle level.

## The open question, decomposed

Constraint that kills the lazy answer (established 08-01 PM): **F2 is
still pending at S-submission on EVERY call, warm or cold** — the host
reaches the S launch ~100µs after enqueueing a fill whose execution takes
~1.2ms. So "F2 pending when S submitted" is the normal, usually-harmless
case: the driver ordinarily orders F1→F2→S correctly across the two launch
APIs. Cold calls break that. The precise question:

> Which driver-visible state or operation, present only on the first
> execution of a window, causes the ordering of {torch-submitted F2,
> raw-cuLaunchKernel-submitted S} on handle 0 to invert — and hold F2's
> execution until S completes?

### Prior bisection results to build on (already measured — do not redo)

- **`torch.cuda.empty_cache()` on a warm trainer re-arms the next call
  14/14** (~80s/rep). Controls on the same substrate: kernel
  recompilation 0/N, cuBLAS-workspace clear 0/N, plain reps 0/20. The
  07-31 session's conclusion stands: for the empty_cache lever, rep0 and
  rep1 differ ONLY in whether the forward must acquire/map fresh physical
  memory mid-flight (warm rep reuses the same caching-allocator block).
- **BUT remap is not necessary**: armH8 (`expandable_segments:False`,
  everything else identical) still fired boot rep0. So boot-coldness has
  at least one additional trigger besides page mapping. (Both results in
  `rearm/NIGHT_0801_FINDINGS.md`.)
- In-process `init_trainer_server` rebuild re-arms 1/1 (likely reduces to
  the allocation trigger via fresh LoRA tensors — unproven).

### Candidate arming variables (separate them; likely non-exclusive)

A. fresh physical-page acquisition for the out buffer (cuMemMap under
   expandable_segments; first-touch cudaMalloc without it) — SUFFICIENT
   per empty_cache 14/14; not necessary per armH8;
B. lazy module/cubin load (`CUDA_MODULE_LOADING` defaults LAZY; a kernel's
   first-ever launch does module work — we measured it serializing a
   launch on the pod, exp3 warmup note). Note "kernel recompilation 0/N"
   tested REcompilation, not first-load — not the same toggle;
C. hardware-channel bring-up (first use of each channel with conn unset);
D. tvm-ffi executor first-call state (lazy init around cuLaunchKernel);
E. driver-version dependence (fires on B300 r5xx prod driver; B200
   historical spikes suggest not B300-specific).

## Plan (leverage order)

### Step 1 — Execution-order witness (make the inversion visible per call)

Build both; trust their agreement:
- **CUPTI latency timestamps** (`CUPTI_ACTIVITY_LATENCY_TIMESTAMPS`, via
  kineto or raw CUPTI): every kernel record gets queued/submitted/start/
  end. One corrupt-cold vs clean-warm trace pair answers: does F2 show
  normal queued but start-after-S-end? Which driver API calls (cuMemMap,
  cuModuleLoad*) sit in the hold gap? Overhead is acceptable — far heavier
  adjudicator wrappers never suppressed the bug.
- **Profiler-free device witness**: hot-patch `_interface.py` to replace
  `out.fill_(-inf)` with two explicit half fills (provably equivalent to
  what torch does anyway) and bracket F1/F2/S with 1-block stamp kernels
  writing `%globaltimer` to preallocated pinned logs. Corrupt call ⇒ watch
  F1→S→F2 directly, ns-resolution, negligible perturbation.

### Step 2 — Single-GPU minimal repro (kill the 25-min boot cost)

One process, one GPU, conn unset, `expandable_segments:True`: loop
{allocate [15914, 60192] fp32 → fill_(-inf) → raw launch of a dummy
row-stamping kernel on stream 0, via BOTH plain cuda-python AND the actual
tvm-ffi executor (the wheel's jit executor imports standalone; 08-01
parity harnesses ran it off-trainer) → scan rows [7957:] for -inf in the
valid region → empty_cache to re-arm}. Outcomes:
- fires → seconds-per-trial lab; proceed to Step 3; NVIDIA gets a trivial
  repro;
- fires only via the tvm-ffi path → arming involves tvm-ffi dispatch
  state (candidate D promoted);
- never fires standalone → bisect the trainer delta one ingredient at a
  time (NCCL channel usage, live stream count, MoE side streams, context
  flags).
Substrate: prod fired on B300 and historically B200 → 1×B200 pod is
legitimate. H100 pods land fastest; a fire on H100 counts, a null does NOT
(arch difference). Devbox q8x5ky3 (2×8 B300) is provisioned and idle if
B300 is required (trainer stopped; see FIX_VALIDATION_100STEP.md
end-state).

### Step 3 — Bisect the arming variable (witness attached to every run)

One toggle per run; detector = erased-rows scan + witness order; write
predictions BEFORE running:
| variable | isolation toggle |
|---|---|
| A page mapping | warm block vs empty_cache-then-alloc vs expandable_segments:False vs preallocated arena reuse |
| B module load | CUDA_MODULE_LOADING=EAGER; pre-launch every kernel once at init; then test rep0 |
| C channel bring-up | pre-touch: launch N dummies on N distinct streams at boot, sync, then rep0 |
| D tvm-ffi init | first tvm-ffi call = dummy kernel on scratch buffer, then the real rep0 |
| E driver | version matrix, only if the repro is standalone |
Example decision rules: C alone cleans boot-rep0 but empty_cache still
re-arms → two independent triggers; B(EAGER) alone cleans everything →
module loading was the story; A-arena cleans empty_cache-armed but not
boot → A and (B|C) both real.

### Step 4 — Package

Write `rearm/ARMING_MECHANISM.md` (matrix, one corrupt + one clean witness
trace, minimal repro); update this HANDOFF with a RESOLVED banner; stage
NVIDIA-report inputs (this + DETERMINISM_MECHANISM_0801.md). Report
send/no-send remains Jack's call.

### Floor / stop conditions

A perfect bisection still ends at "this driver-visible op flips the
ordering" — the semaphore/channel bookkeeping beneath is closed-source and
belongs to NVIDIA. Do not burn time past that line. If Steps 1–3 exceed
~2 focused days without a reproducible toggle, write up the proven
negative space and stop.

## Infrastructure & recipes

- **Pod recipe** (worked 08-01): baseten-pod skill; 1×H100 lands ~3 min on
  yta-aps1 (B200 on hyd may queue; 08-01 it never landed). Trainer-exact
  torch: `pip install torch==2.11.0 --index-url
  https://download.pytorch.org/whl/cu130` (pod driver 580.126.20 handles
  cu130) + ninja + numpy. load_inline gotcha: use `int64_t`, never
  `long long` (Linux ABI symbol miss).
- **Wheel source** (standalone tvm-ffi path + pristine _interface.py):
  overnight scratchpad mirror
  `/private/tmp/claude-501/-Users-jackrao-Documents-trainers/f49d39a1-a42b-4669-8099-40083fda1598/scratchpad/wheel/`
  (+dsatopk1 content). The FIXED wheel lives only in PR #875's branch
  (worktree `trainers-wt-dsa-launch-stream`) — arming experiments must use
  the UNFIXED default-stream arrangement.
- **Trainer repro** (only if Step 2 dead-ends): devbox q8x5ky3;
  prod-faithful boot via `parity/run_trainer_node_prodenv.sh` +
  `check_env_parity.sh`; conn must be UNSET (the stock devbox script
  exports conn=1 and masks everything — the single most expensive lesson
  of this investigation). Verify the venv wheel by **sha256 of
  _interface.py**, never dist-info version strings (dist-info lies).
- **Detectors**: call-level = adjudicator family d12 masks (overnight
  scratchpad + `/root/lps1003_local/` on devbox nodes); step-level =
  `analyze_replay_spikes.py` (calibrated).
- **Corpus** (2,059 cold-window mask events, for cross-checking any new
  theory): scratchpad `adj2_data/`; loaders `rearm/exp/corpus_split_check.py`,
  `analyze_masks.py`.

## Pitfalls (each cost hours once; do not repeat)

- pycode exec() hot-patch installs give wrappers NO closures →
  closure-walk recovery silently chains wrappers (once reached 18 kernel
  execs/call). Run `probe_chain.py` before trusting any wrapper-level
  instrument.
- harness6's MetaPathFinder re-wraps the api module on ANY
  importlib.reload — remove the finder before reloading.
- Instrumentation allocation churn itself arms the race — preallocate and
  reuse every witness buffer.
- A kernel's first-ever launch = lazy module load = fake serialization
  signal; warm up every instrument kernel before measuring (exp3 lesson).
- Never use NLL direction as the firing criterion — use the buffer mask.
- No in-place wheel swaps on a live multi-rank trainer (NCCL abort);
  wheel changes only via fresh boots.
- No ad-hoc sleeps on devbox — .devbox_up wait scripts, bounded waits.
  Debug runs go to jackrao-* W&B projects only, never customer projects.
- CPFS 1M-inode quota outage (08-01/02) can block devbox provisioning
  org-wide — `CPFS_QUOTA_ESCALATION.md`; hosts e02-sg-e1n4vn65z0k/r are
  dead (never returned from reboot), do not wait on them.

## Success criteria

1. A named arming trigger (or proven small set) with an ON/OFF toggle
   demonstrated ≥5/5 armed vs 0/5 disarmed on the same substrate,
   witness-confirmed (F1→S→F2 order visible on armed calls only).
2. Or, if no standalone repro: the trainer-ingredient bisection matrix
   completed to the same standard.
3. `rearm/ARMING_MECHANISM.md` written; this HANDOFF updated with a
   RESOLVED banner; NVIDIA-report inputs staged for Jack.
