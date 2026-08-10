# PRE-DRAFTED PR BODY — FIX A v3: first-pass-anchored DSA-bwd nonempty probe (mcore)

**Repo:** basetenlabs/Megatron-LM · **Branch:** to be cut at the A-v3 stack
commit (15d5679eb, "stack 8/8") — stacked on the W2 branch if W2 ships, else
re-based onto `jackrao/lps-1062-ship-w1` · **Base:** per stack state at open
**OPEN CONDITION (helmholtz):** leg (b) PASSED (2026-08-10: mechanism +
numerics + wall +4–5 % informational-positive); **leg (a) — the composition
soak (V3 invariant with the replay cache engaged) — is the last open gate.**
Do NOT open before helmholtz confirms leg (a). (Body otherwise filled.)

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

**Known improvement (not a blocker; curie's leg-(b) finding, grothendieck's
read concurred):** V3-ON introduces a recurring >1 ms D2H memcpy class —
33 calls/window, 0.71 s total, max 132 ms tail — parented by `aten::copy_`
with grandparents `aten::repeat` ×20 / `clone` ×8 / `_to_copy` ×4 /
`scatter` ×1, ABSENT in the C′-era trace (6 calls, 0.062 s, zero
`aten::repeat`). Both boots B/F+C′+W1, so the delta is exactly V3.
**Mechanism read (CORRECTED after curie's sanity check — my first read
attributed it to the kick's `wait_stream(current)` inheriting the compute
backlog, the W3-v2 defect class; curie's rows refute that: the probe
eventSyncs are µs-scale (1.8 ms total across 312 probes, zero >1 ms — a
backlog-inheriting kick would show long host waits there), and kineto
gpu_memcpy duration is execution time, not queue wait — a 132 ms D2H is a
hundreds-of-MB-to-GB-class copy, not a delayed 1-byte flag).** The
evidence-supported mechanism is CONSTRUCTION-SIDE: a big repeat-constructed
tensor (the probe input's 131k-scale construction) moving to host. The fix
that the evidence supports: keep the probe input/result device-resident (or
shrink the repeat-class construction) — not (only) input-dependency
ordering. ~1.8 % of window time as measured + the 132 ms tail; removal
likely improves the 131k wall beyond tonight's +4–5 %. Filed as a follow-up
(the memcpy_gt1ms checker row it tripped is ruled intent-satisfied,
era-exact; the BFC profile's coverage gap vs the probe-input construction is
documented in curie's record). Residual measurement (morning-class, curie
offered): the 33 copies' start-gaps vs the compute backlog on the leg-(b)
trace — settles whether ordering plays any secondary role.

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
