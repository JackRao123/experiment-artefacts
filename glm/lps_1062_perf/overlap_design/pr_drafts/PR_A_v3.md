# PRE-DRAFTED PR BODY — FIX A v3: first-pass-anchored DSA-bwd nonempty probe (mcore)

**Repo:** basetenlabs/Megatron-LM · **Branch:** to be cut at the A-v3 stack
commit (15d5679eb, "stack 8/8") — stacked on the W2 branch if W2 ships, else
re-based onto `jackrao/lps-1062-ship-w1` · **Base:** per stack state at open
**STATUS: OPENED AS DRAFT 2026-08-10 — basetenlabs/Megatron-LM#29** (the
word given by helmholtz: leg (b) mechanism+numerics PASS +4–5 % informational;
leg (a) composition PASS 6600/6600). Branch `jackrao/lps-1062-ship-av3` =
A-v3 cherry-picked onto ship-w1 (d58a3214a; dsa files only — W2/W3 excluded),
CPU suite ALL PASS re-run on the branch. The opened body carries the
three-regime robustness line, the settled D2H note, and the regime-finding
story; this file is kept as the pre-draft record.

---

## What

`BT_DSA_BWD_ASYNC_NONEMPTY_V3=1` (default OFF; distinct from the parked v2
gate `BT_DSA_BWD_ASYNC_NONEMPTY`): replace the per-layer-backward syncing
`torch.nonzero(topk_length > 0)` in DSA sparse-attention backward (312
drains/step) with an async 1-byte nonempty probe **anchored in the no-grad
first pass** (seconds before the backward, GPU queue shallow — the event is
long complete whenever the backward reads it, in every regime). Verify mode
`BT_DSA_BWD_ASYNC_NONEMPTY_V3_VERIFY=1` re-derives the flag on every replay
and asserts equality (loud stop; soaks only).

Files: `dsa_cudnn_kernels.py` (+267), `dsa.py` (+26), `dsa_kernels.py` (+11).

## Why — and what would make this merge-worthy

v3 is **a measurement of the post-C′ regime**, not a promised win. Under
C′-on the 312 DSA-bwd drains carry ~24 s/step of CPU wait that is 97.4 %
GPU-kernel-covered and wall-flat vs the C′-off control — mostly *absorbed*
wait (redistributed replay-throttle parking), not exposed stall. The
pre-registered expectation (boltzmann, ratified): the exposed fraction is
small and **a ~0 wall result is a valid publishable answer** (drains still
overlapped). The mechanism rows stay primary: `eventsync_a_*` ≈ 312 µs-scale
calls/step, ≤1.5 s total, ≤25 events >1 ms for a healthy A-active capture;
`aten::nonzero` slices >5 ms ≈ 0; `cudaStreamSynchronize` −312.

**Timed-arm verdict (leg (b), 2026-08-10, curie — record:
`~/perf_profiles/lps-1062/AV3_LEGB_VERDICT.md`):** **MECHANISM PASS ·
NUMERICS PASS · wall informational-positive (+4–5 % steady at 131k, honest
steady-vs-steady read).** Leg (a) — the composition soak (V3 invariant with
the replay cache engaged) — remains the last open gate before this PR opens.

**Known improvement (not a blocker; curie's leg-(b) finding):** V3-ON
introduces a recurring >1 ms D2H memcpy class — 33 calls/window, 0.71 s
total, max 132 ms tail — absent in the C′-era trace (6 calls, 0.062 s); the
boot delta is exactly V3. **Settled mechanism (curie's start-gap measurement
on the leg-(b) trace, host↔GPU correlation — final after two wrong
premises):** the GPU-side copies are µs-scale (p50 0.00 ms, max 0.02 ms —
execution is instantaneous); ~100 % of the host-side duration is WAIT
(host-minus-GPU p50 9.01 ms, max 132.3 ms); the compute stream is 100 % busy
during every call; all 33 copies are on stream 7 — the COMPUTE stream, and
32/33 sit inside `CheckpointFunctionBackward` (the checkpoint recompute +
backward). So the class IS the whole-backlog ordering defect: a
pageable/synchronizing D2H read issued on the compute stream blocks the host
until the stream drains to the copy point — the read inherits the recompute
backlog by direct stream order. (Settled after two wrong premises, both
recorded: my first read's locus — the kick's side-stream `wait_stream` —
was wrong, the copies are on the compute stream; curie's first refutation —
host-duration as GPU execution, a size/bandwidth story — was wrong. The
estate lesson cuts both ways: a defect-class pattern match is a hypothesis,
and so is a refutation built on an unchecked premise.) **Fix direction:**
keep the flag device-resident and read lazily, or a pinned-memory
truly-async copy with a deferred host read, or order the copy after only the
producer's event (input-dependency ordering, correctly located).
**Line-level locus (bounded negative result from the design-lane code read,
2026-08-10):** the op chains (20× `copy_←repeat`, 8× `clone←masked_fill←
MaskedFillBackward0`, 4× `_to_copy←to`, 1× `scatter`) match NO host-read
call in the V3 patch surface — the intentional D2H (the side-stream pinned
`.all()`-flag copy) is truly async and its source chain is `gt/all`, not
repeat; the read path is host-only (`event.synchronize()` + pinned
`.item()`); the verify path's device `.item()` is soak-gated and was off in
leg (b). So the copies are a V3-activated host read in the replay's
execution of EXISTING code (the checkpoint backward re-runs the indexer/
attention forward, which carries the repeat/masked_fill machinery), not a
line in the patch itself. The morning instrument is a capture with python
stacks (the slimmed trace bottoms out at CheckpointFunctionBackward); the
search is bounded to the replay-side indexer/attention forward path.
~1.8 % of window time + the 132 ms tail as measured; removal likely improves
the 131k wall beyond tonight's +4–5 %. (The memcpy_gt1ms checker row it
tripped is ruled intent-satisfied, era-exact; the BFC profile's coverage gap
is documented in curie's record.)

## Evidence / tests

- Design + hypothesis: [dispatcher_opt/fixa_v3/DESIGN_FIXA_v3.md](https://github.com/JackRao123/experiment-artefacts/blob/main/glm/lps_1062_perf/dispatcher_opt/fixa_v3/DESIGN_FIXA_v3.md)
  (incl. the v2 dependency chain: why the replay-anchored v2 probe measured
  ≈0 under the pre-C′ throttle)
- On-box validation recipe (verify soak → canary → steady capture):
  [dispatcher_opt/fixa_v3/ONBOX_VALIDATION.md](https://github.com/JackRao123/experiment-artefacts/blob/main/glm/lps_1062_perf/dispatcher_opt/fixa_v3/ONBOX_VALIDATION.md)
- Test: [test_fixa_v3_probe.py](https://github.com/JackRao123/experiment-artefacts/blob/main/glm/lps_1062_perf/dispatcher_opt/fixa_v3/test_fixa_v3_probe.py)
  (Mac-CPU, 33 assertions; v2 + FIX B suites re-run green)

## Numerics / memory

Bitwise-expected (the arange substitution is exact); canary drift >5e-3 =
STOP. Memory ~0 (one pinned 1-byte buffer + event per layer, carrier-scoped).

## Status

Prepared for review — **do not merge** (ship go/no-go is Jack's). Opens only
on an ARM-5 PASS; if the arm returns the tempered ~0-wall publishable
answer, the ship question goes to Jack explicitly (mechanism cleanliness vs
code carried).
