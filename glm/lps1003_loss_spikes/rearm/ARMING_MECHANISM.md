# LPS-1003 Issue 2 — the arming mechanism, SOLVED (2026-08-02)

**Question answered: why does the corruption fire on rep0 (cold window) and
never on rep1 (warm re-execution)?**

## One-paragraph answer

There is no driver anomaly and no arming state. The unfixed wheel's
`torch_stream_context` helper wraps the caller's stream handle as
`torch.cuda.ExternalStream(int(handle))`; for the legacy default stream the
handle is `0`, and **`torch.cuda.ExternalStream(0)` does not alias the
default stream — torch's C++ constructor treats `stream_ptr=0` as "not
provided" and hands back a stream from torch's internal 32-stream pool, a
different real stream on every call** (measured: 32 unique handles cycling
with period 32). So every `indexer_fwd` call runs its `-inf` prefill (F1+F2)
and the `out` allocation on a fresh private pool stream, while the tvm-ffi
raw launch of the CuTe kernel (S) goes to the legacy default stream. F2 and
S are **never ordered, on any call, warm or cold**. Whether the missing
ordering *manifests* as corruption is pure execution timing: corruption
requires S to begin executing before F2 finishes. Cold windows are exactly
the moments that make S start early; warm steady-state delays S by
milliseconds behind a deep default-stream queue, so the sub-millisecond
fills always win and the call looks healthy.

## Why cold fires and warm heals (the rep0/rep1 asymmetry)

- **Warm trainer call**: the host runs milliseconds ahead of the device;
  when `indexer_fwd` is reached at layer *k*, the legacy stream already
  holds a deep backlog of layer 0..k-1 kernels. S is enqueued *behind* that
  backlog and starts long after the fills (which run immediately on their
  idle private pool stream) have completed. Clean — deterministically, not
  luckily.
- **Cold window (boot / `empty_cache` / in-process rebuild)**: the forward
  is stalled throughout by synchronous driver work — `cuMemMap`/
  `cudaMalloc` for every reallocated buffer, lazy `cuModuleLoad` on first
  kernel launches — so the device drains and the host never gets ahead. At
  the `indexer_fwd` call the legacy stream is empty; S starts within launch
  latency (~60-150 µs), while F2 (which cannot start until F1's ~0.2-1.2 ms
  sweep completes, same private stream) is still pending or mid-flight. S's
  oversubscribed grid then occupies the SM work distributor; F2's blocks
  dispatch behind S's CTA queue and its -inf sweep lands mid/late-S, erasing
  every row S had already stored. That is the F1 → S → F2 signature.
- **Allocator correlation explained, causation refuted**: fresh-page
  acquisition correlated (empty_cache re-arm 14/14) because page-mapping
  storms are one of the two host-stall mechanisms that drain the pipeline —
  not because mapping itself disturbs launch ordering. In the standalone
  lab, `ec_prealloc` (empty_cache a no-op, zero allocator activity) still
  fires 8/8, and armH8 (expandable_segments:False) fired because boot is a
  drained-pipeline moment regardless of allocator backend.
- **rep0-exec1-of-process clean in the lab**: the process's first call pays
  ~868 µs of host-side latency between F2's enqueue and S's launch call
  (measured on the boot trace), so the fills finish before S is even
  submitted. Attribution nuance, honestly stated: `CUDA_MODULE_LOADING=
  EAGER` did NOT remove this protection (eager run: process-first exec
  still clean) and neither did a sub-threshold warm-up call with the same
  compiled artifact (wsmall run: first one-two big execs still clean) — so
  the first-call delay is not solely lazy cubin load; tvm-ffi/cute
  first-call setup and early-process host jitter dominate. Either way this
  is first-call *protection* via host delay, not an arming mechanism, and
  it is why the standalone lab inverts the trainer's rep0/rep1 pattern
  (in the lab, "cold" = idle device, and only the first call gets the
  host-delay shield).
- **rank15 immunity, restated**: its buffer is under 2^31 B → single fill
  launch, submitted ~1 ms before S and starting immediately on the idle
  pool stream; it always completes before S's first stores. The split
  fill's F2 is what inherits a start time *after* S has begun.

## Standalone lab (devbox q8x5ky3 node0, 1× sm103, single process)

Geometry = the PR #875 CI repro (out [8192, 73728] fp32 = 2.25 GiB → fill
splits at row 4096). Unfixed wheel via PYTHONPATH shadow (`_interface.py`
sha 5d0e429e…). conn UNSET, expandable_segments:True unless noted.
Detector: per-exec full scan vs exact causal reference; "fires" = fully
erased rows (observed stable core [6368, 8191] + ~1180-row racy fringe).
Every exec is preceded by a sync (idle device) — per the mechanism, that
means every exec should fire except first-ever-launch execs, regardless of
any allocator lever. Results (results/*.jsonl):

| run | lever | fired |
|---|---|---|
| ec | empty_cache each rep | 15/16 execs (only process-first exec clean) |
| warm | none | 15/16 execs (same) |
| ec_premap | empty_cache + premapped warm blocks | 15/16 |
| ec_prealloc | out= preallocated once; empty_cache no-op | 15/16 |
| ec_map_elsewhere | out preallocated; unrelated map/unmap churn | 14/16 |
| conn1 | CUDA_DEVICE_MAX_CONNECTIONS=1 | 0/8 execs — masked |
| fixed wheel (+dsatopk5, dedicated launch stream) | ec lever | 0/8 execs |
| eager (CUDA_MODULE_LOADING=EAGER) | ec lever | 15/16 (process-first exec STILL clean — first-call protection ≠ module load alone) |
| wsmall (sub-threshold warm-up call first) | ec lever | 14/16 (first two big execs still clean — same conclusion) |
| **busy** (~20 ms of matmuls pending on the DEFAULT stream, no sync, then call) | none | **exec1: 2/8 fired (13/16 clean across busy+busy_ec) — protected; exec2 (after the mid-rep sync drains the queue): 8/8 fired.** The same rep shows protected-then-armed with identical allocator/wheel state — queue depth is the whole gate |
| busy_ec | empty_cache + pending matmuls | exec1: 1/8 fired; exec2: 8/8 — allocator irrelevant even here |

## The witness (kineto traces, rearm/arming/traces/)

CUPTI kernel records, one exec each (stream = CUPTI execution stream):

- **corrupt warm exec** (`trace_w1_warm1.json.gz`): F1 on stream 25
  [0, 160 µs]; S on stream 7 (legacy) [158, 4896 µs]; **F2 on stream 25
  [3532, 3742 µs] — mid-S**. The fills are on a different stream than S,
  and the stream id increments every call (17, 25, 33, …, +8/exec = fresh
  pool stream per call).
- **clean boot exec** (`trace_w1_boot.json.gz`): same stream split, but the
  host's `cuLaunchKernelExC` for S sits at +868 µs (module load) — the
  fills [0, 323 µs] were long done. Order F1 → F2 → S by accident of host
  timing only.
- Profiling does not suppress: unprofiled control execs in the same session
  fired identically.
- **disarmed busy exec** (`trace_busy_exec1.json.gz`): 40 pending matmuls
  occupy stream 7 until 23,913 µs; F1 [12333] and F2 [13062] run during
  the junk on pool stream 25; **S starts at 23,915 µs — the instant the
  junk drains — long after both fills. Output fully clean.** The armed/
  disarmed trace pair is the ON/OFF witness: F1→S→F2(mid-S) when stream 7
  is idle at submission, F1→F2→S when it is deep.

Direct probe (`arm_lab`-independent, probe_stream.py): with the caller on
the default stream, `int(cuda.CUstream(torch.cuda.current_stream()
.cuda_stream)) == 0`, and `torch.cuda.ExternalStream(0).cuda_stream` is a
**nonzero, different handle on every construction** — 32 unique handles,
cycling, identical population to `torch.cuda.Stream()` pool handles. Inside
`with torch.cuda.stream(ExternalStream(0))`, kernels demonstrably execute
on the pool stream (CUPTI id ≠ legacy id 7).

## What this corrects in earlier docs

- NIGHT_0801_FINDINGS / DETERMINISM_MECHANISM: the launch-ordering
  inversion is NOT "imposed at the driver/channel layer" and there is no
  "hold" anomaly. F2 executes late because (a) it serializes behind F1 on
  its own (pool) stream and (b) its blocks then queue behind S's
  oversubscribed grid in the ordinary work distributor. The
  "F1→S→F2 with full serialization semantics" corpus signature is exactly
  what unordered streams + FIFO grid dispatch produce. Everything else in
  those docs (split at numel/2, byte threshold, rank ladder, corpus
  geometry, exp1-5 measurements) stands.
- The in-situ streamlog "torch_stream == 0 == tvm-ffi env stream" was
  measured OUTSIDE `_torch_stream_context`; the handle actually used for
  the fill launches was the pool stream's, one level deeper.
- **NVIDIA driver report: withdrawn.** The driver behaved correctly
  throughout. The defect is in the vendored wheel's stream glue
  (`cudnn/deepseek_sparse_attention/utils/runtime.py::torch_stream_context`
  and the same inline pattern in `sdpa/bwd/api.py`), which affects **60+
  call sites** across indexer fwd/bwd, score_recompute and sdpa bwd —
  every torch op the wheel runs "on the caller's stream" actually runs on
  a rotating pool stream whenever the caller is on the default stream.
  Two reports are warranted instead:
  1. NVIDIA cudnn-frontend/DSA wheel team: the `ExternalStream(int(h))`
     pattern is broken for h==0; other sites are latent hazards (currently
     mostly no-op'd by contiguous inputs / TMA-aligned shapes on our path).
  2. PyTorch upstream (optional): `torch.cuda.ExternalStream(0)` silently
     returning a pool stream is a footgun; it should raise.

## Why the fix works (unchanged)

PR #875 (+dsatopk5) routes prefill + launch through one dedicated
per-device `torch.cuda.Stream` with event chaining to the caller — all
three kernels share one concrete stream across both launch APIs, so
ordering is structural. Validated 0/100-step spikes on the real recipe.

## Status of the handoff's candidate list

- A (fresh page acquisition): refuted as mechanism (ec_prealloc fires with
  zero allocator activity); real as a *host-stall correlate* on the
  trainer.
- B (lazy module load): real, but inverted — it *protects* first-ever
  calls; never armed anything.
- C (channel bring-up): moot — no driver anomaly to arm.
- D (tvm-ffi first-call state): moot; tvm-ffi's env-stream launch to the
  legacy stream is fine — it's the *fill side* that wanders.
- E (driver version): moot.
