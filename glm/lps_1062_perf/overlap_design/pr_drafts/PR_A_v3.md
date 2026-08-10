# PRE-DRAFTED PR BODY — FIX A v3: first-pass-anchored DSA-bwd nonempty probe (mcore)

**Repo:** basetenlabs/Megatron-LM · **Branch:** to be cut at the A-v3 stack
commit (15d5679eb, "stack 8/8") — stacked on the W2 branch if W2 ships, else
re-based onto `jackrao/lps-1062-ship-w1` · **Base:** per stack state at open
**OPEN CONDITION (helmholtz):** only after the A-v3 timed arm (ARM-5) verdict
lands PASS. Do NOT open before helmholtz confirms. (Skeleton — fill the
verdict numbers at open time.)

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

**Timed-arm verdict (ARM-5):** *fills at open time — mechanism rows + wall
vs the C′-on baseline, per the two-tier rule.*

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
