# LPS-1003: why the "race" is deterministic — verified mechanism (2026-08-01 PM)

> **2026-08-02 CORRECTION (ARMING_MECHANISM.md):** all measurements here
> stand (fill split at numel/2, byte threshold, rank ladder, corpus
> geometry, exp1–exp5). Two interpretive clauses are superseded: (1) "the
> inversion happens below the submission API / only NVIDIA can inspect" —
> wrong; the fills were submitted to a DIFFERENT stream (torch pool stream
> via the ExternalStream(0) footgun), so no inversion ever occurred;
> (2) "F1→S→F2 with stream-serialization semantics imposed at the
> driver/channel layer" — the observed serialization is ordinary FIFO grid
> dispatch: F2 serializes behind F1 on its own pool stream, then its
> blocks queue behind S's oversubscribed grid. The NVIDIA-report framing
> in "Fix assessment" is withdrawn accordingly.

Session goal (Jack): the overnight root cause said "the -inf prefill lands
after the kernel's stores and erases the trailing rows; deterministic erase
boundary = fill-kernel block schedule." That last clause was inference, and
suspicious: a race should be noisy, and a boundary bit-stable at exactly
total/2 across 1500+ events looks structural, not stochastic. This session
verified the determinism question with sources and direct GPU observation.
Result: the boundary is structural — and the "fill-kernel block schedule" /
"wave" story is wrong. **The fix (PR #875) is unaffected and remains
validated; what changes is the explanation and the NVIDIA report framing.**

## The one-sentence answer

`out.fill_(-inf)` on the rank-0 scores buffer is not one kernel — PyTorch's
TensorIterator splits any elementwise op whose per-operand **byte** offsets
exceed INT32_MAX into **two back-to-back kernel launches, split at exactly
numel/2** (F1 = rows [0, R/2), then F2 = rows [R/2, R)); the destroyer is F2
— an entire *launch* — losing ordering against the DSA kernel, so the erase
boundary is a kernel-launch boundary, deterministic by construction, at
exactly row 7957 on every rank and any column width (N/2 = (R/2)·C).

## Source facts (trainer's exact torch, 2.11.0+cu130 per server/uv.lock@trainer-cuda13-sm103)

- `TensorIteratorBase::can_use_32bit_indexing()` — requires numel ≤ INT32_MAX
  AND per-operand `max_offset = 1 + Σ (shape−1)·stride_bytes` ≤ INT32_MAX.
  Strides are in **bytes**: a contiguous fp32 tensor over 2,147,483,647 bytes
  (536.9M elements) fails even though numel is far below int32 max.
- `gpu_kernel()` (Loops.cuh) then loops `iter.with_32bit_indexing()`;
  `split()` halves the (coalesced-1D) shape; `SplitUntil32Bit` stack order
  executes the **first half first**. One split suffices for ≤ 4 GiB.
- rank0 buffer [15914, 60192] fp32 = 3,831,581,952 B → split at element
  478,947,744 = row 7957.000 exactly. `out` is freshly-allocated and
  contiguous inside `indexer_fwd` (caller passes no out).

## GPU facts (1×H100 pod, torch 2.11.0+cu130 — pod job w64gyyq; scripts in exp/)

1. **exp1 (kineto):** fill_ launches exactly **2** `vectorized_elementwise_
   kernel<FillFunctor>` for rank0 [15914,60192] and rank14 [15914,33858]
   (0.36% over the limit), exactly **1** for rank15 [15914,31977] (5% under).
   Synthetic pair 16 bytes over / 16 bytes under INT32_MAX: 2 vs 1 kernels —
   the threshold is bytes, measured.
2. **exp2 (device spy, 20/20 trials):** polling elements N/2−1 and N/2
   (adjacent addresses — same 512-elem chunk, same warp if the fill were one
   kernel) during fill_: always state sequence (neither) → (**N/2−1 filled,
   N/2 unfilled, dwell 4.0–7.9 µs**) → (both). Reverse state 0/20. Same-warp
   stores cannot be split for microseconds — this directly observes the
   launch boundary at N/2 and the F1-first order.
3. **exp3 (co-residency):** a fill-style kernel launched on another stream
   runs 100% DURING a persistent all-SM 384-thread kernel, sharing all 132
   SMs, at 0/100/200 KB dynamic smem. **SM-resource starvation cannot delay
   F2** — the prod destroyer's "F2 executes after essentially all of S"
   ordering is imposed at the driver/channel level, not by occupancy.
   (Method note: kernel warmup matters — the first-ever launch of a kernel
   serializes on lazy module load and fakes a starvation signal.)
4. **exp5 (block scheduling):** for a 935,445-block uniform grid: dispatch
   order Spearman ρ = 1.000000 across 8 runs; start-time vs blockIdx Pearson
   r = 1.000000 (perfectly ascending, ~394 block-starts/µs). Hardware block
   dispatch for uniform grids is deterministic and low-to-high — the fill
   sweeps addresses in order, every time.
5. **exp4/4b (legal-stream race, real rank0 tile geometry):** producer
   (persistent work-queue kernel, per-tile spin ∝ real key width,
   progressive stores) vs real fill_ on unsynchronized streams, 96 trials:
   60 corrupt + 36 clean, **every corrupt mask lo ≥ 7957, zero below, zero
   full wipes; event-ordered control 10/10 clean.** Instrumented reruns show
   every per-segment bad/clean cell equals sign(store_time − fill_sweep
   arrival at that row): the mask is a deterministic function of the two
   kernels' store timelines.

## Corpus facts (2,059 adjudicator mask events, 15 firing ranks)

- **0/2059 events start below row 7957.** Destroyer class starts at exactly
  7957 (85–96% of events per rank); partial class lo ∈ [8449, 14189].
- Per-rank C ladder C(r) = 60192 − 1881·r (matches the glue's
  max_segment_k formula). **rank14 = 2,155,264,848 B (+0.36% over INT32_MAX)
  → split → 144 events. rank15 = 2,035,527,912 B (−5.2% under) → single fill
  kernel → 147/147 calls clean, zero events ever.** A natural experiment
  straddling the exact byte threshold; supersedes the "warm blocks/smaller
  fills win" guess for rank15 immunity.
- Destroyer "holes" (spared rows) are 8-row-aligned runs (TMA store-pipeline
  granule) at the start/end of seg13; partial-event bad sets = late-storing
  segments with mid-row first_bad — store-granularity race texture exactly
  where F2-vs-tail-stores interleaving puts it. The variance Jack asked
  about EXISTS — at store granularity, strictly inside the F2 region; the
  launch boundary is the deterministic part.

## Revised destroyer timeline (what the driver anomaly must do)

Program order on one nominal stream: F1, F2, S(tvm-ffi raw launch). All
masks are consistent with the channel executing **F1 → S → F2 with
stream-serialization semantics** (each waiting predecessor completion):
first half filled then overwritten normally; S completes; F2 erases the
whole second half; the 8-row spared runs = S's async TMA store flush tail
landing after its completion boundary while F2 sweeps. Partial events = the
rarer overlapped variants. Since co-residency is freely available (exp3),
only a launch-level ordering edge — not SM pressure — can hold F2 through
S's multi-ms runtime. Host-side submission interleaving is impossible (one
Python thread submits F1, F2, S sequentially), so the inversion happens
below the submission API: exactly the layer only NVIDIA can inspect.
conn=1 (single channel) removes it; explicit non-default stream (the fix)
removes it; single-launch fills (rank15) never expose a pending fill to
reorder past.

## Fix assessment — unchanged, plus one new lever

- **PR #875 (+dsatopk4) stands as-is**: prefill + launch on one dedicated
  stream, event-chained. All three kernels (F1, F2, S) share one concrete
  stream object across both launch APIs; validated 0 corrupt in ~24k calls,
  origin payload clean, 12 training steps clean.
- New defense-in-depth option (not required): keeping any pre-kernel
  prefill under 2^31 bytes per launch (chunked fills) empirically avoids
  this driver behavior (rank15: 0 events ever) — worth one line in the
  NVIDIA report and worth knowing for other >2 GiB prefill-then-raw-launch
  sites, but the stream fix is the correct general solution.
- NVIDIA report should include: minimal repro shape (>INT32_MAX-byte
  elementwise op split into two launches on the legacy stream, immediately
  followed by a raw cuLaunchKernel on handle 0, CUDA_DEVICE_MAX_CONNECTIONS
  unset, B300), the F1→S→F2 serialization signature, and the 8-row TMA
  flush-tail holes.

## Corrections to earlier docs

- NIGHT_0801_FINDINGS.md "deterministic erase boundary = fill-kernel block
  schedule" → the boundary is the TensorIterator 32-bit-indexing launch
  split at numel/2. "rank15's smaller fills win the race" → rank15's fill
  is a single launch (under the byte threshold); there is no F2 to lose.
- Any wave-boundary narrative for the destroyer cut is dead: the fill grid
  is ~1.87M blocks ≈ hundreds of waves; the bit-stable cut could never be a
  wave edge (and measured dispatch is a smooth 394-blocks/µs conveyor).

## Artifacts

- exp scripts + outputs: exp/ next to this file (exp1_fill_split.py,
  exp2_boundary_spy.py, exp3_starvation.py, exp4_synthetic_destroyer.py,
  exp4b instrumented variant, exp5_sched_determinism.py, corpus_split_check.py).
- Pod: jackrao-lps1003-microbench-h100 job w64gyyq (1×H100, torch
  2.11.0+cu130 — the trainer's exact pin), torn down after the session.
- Corpus source: adj2_data (overnight session scratchpad mirror).
