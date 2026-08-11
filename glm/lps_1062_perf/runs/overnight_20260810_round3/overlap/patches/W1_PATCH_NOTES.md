# W1 — BT_MOE_PROBS_A2A_COMM patch notes (helmholtz, 2026-08-09)

## What

`w1-probs-a2a.patch` — move the MoE dispatcher's probs all-to-all onto a
**second NCCL communicator** over the EP ranks, issued concurrently with the
tokens all-to-all (both issued before either wait), removing the latency-bound
probs call (~5.7 ms × 900 layer-passes ≈ 5.1 s/step at 131k×d4) from the EP
communicator's serial stream. Design: `../DESIGN_helmholtz.md` §4.

## Base (apply over)

- Repo: `server/vendor/megatron-bridge/3rdparty/Megatron-LM` @ `57efae08b`
  (= on-box pin) **plus the current dirty tree**: FIX A/B/F (dsa/rope) **plus
  hilbert's FIX C** (`BT_MOE_DISPATCH_REPLAY_CACHE`, frozen 2026-08-09).
- File states the patch was generated against:
  - `megatron/core/tensor_parallel/mappings.py` — clean @ `57efae08b`
    (hunk base: `_AllToAll` ends ~line 484; `all_to_all` wrapper ~line 558).
  - `megatron/core/tensor_parallel/__init__.py` — clean @ `57efae08b`.
  - `megatron/core/transformer/moe/token_dispatcher.py` — FIX-C working tree
    (blob `da42fc946`; hunk anchors: gate block after `_replay_pass_record`
    ~line 437; class attrs at `cuda_dtoh_stream = None` ~line 759; `__init__`
    after the dtoh-stream creation ~line 840; `token_dispatch` ~line 1067).
- Apply: `cd <mcore> && patch -p1 < w1-probs-a2a.patch` (or `git apply`).
  Verified: applies clean and reproduces the author's working tree byte-for-byte.

## Composition with FIX C (reviewed)

W1 consumes only `self.input_splits` / `self.output_splits` at
`token_dispatch` time. Those are valid host values in every FIX C mode
(normal: D2H'd; replay hit: restored from the cache with D2H/event-sync
skipped; verify: freshly D2H'd and compared). The issue/issue/wait-both
ordering adds no dependency on `d2h_event` — safe on the replay-cache skip
path. No shared state with the replay cache (W1 handles ride on the output
tensors, not the dispatcher instance).

## Gate + telemetry (all WARNING-level; reaches trainer_srun.log)

- `BT_MOE_PROBS_A2A_COMM=1` enables; **default OFF**; gate off ⇒ byte-identical
  upstream path.
- One-time gate-state line (ACTIVE or "present but DISABLED") at first use.
- One-time `armed — second communicator created over N EP ranks` at model
  build (first MoE layer's dispatcher init; `new_group` is a world collective,
  called at the same point on every rank). `armed=NO — EP size is 1` if EP=1.
- Per-window counters every 300 gated dispatches (~1 step), first 100 windows:
  `{token_issues, probs_issues, waits}` — a gate that can never fire is loud
  by absence of these lines.

## Numerics

Bitwise-safe: both collectives move bytes verbatim; the second communicator
carries the identical messages. Backward rides the second communicator
automatically (reverse A2A with swapped splits, issued+waited inline as
upstream). At `CUDA_DEVICE_MAX_CONNECTIONS=1` (devbox benches) the backward
timing is unchanged (the probs-reverse wait head-of-line-blocks the tokens
reverse issue); with CDMC unset (production pods) the backward also
de-serializes (~another 1.7 s/step).

## Tests

- **T1 (CPU, gloo, 2 procs)** — `../tests/test_w1_probs_a2a.py`: deferred
  issue+wait == reference `all_to_all_single` (unequal + equal splits); the
  issue/issue/fc1/wait/wait ordering; second-communicator value equivalence;
  work-handle lifecycle + world_size==1 bypass; backward parity with the
  reverse-A2A reference; gate/telemetry counters. **Status: all PASS on Mac
  CPU** (torch 2.11, gloo). Note: gloo shares one transport across groups and
  cannot run two communicators concurrently — the concurrent two-comm pattern
  is asserted mechanically on one group and covered for real by T3 on NCCL.
- **T3 (on-box bitwise canary)** — recipe below.

## T3 canary recipe (on-box)

Goal: prove step-by-step **bitwise** loss identity, gate off vs on, and
measure the step-time delta. (The design claims exactness — stronger than the
≤2e-3 loss-canary norm.)

1. Apply `w1-probs-a2a.patch` on the box checkout (over the same dirty base
   as above — the box tree already carries FIX A/B/F; apply FIX C first if
   hilbert's freeze has landed there, else apply W1 anyway: the W1 hunks do
   not overlap FIX A/B/F's files except `token_dispatcher.py`, where they
   need FIX C's version as base).
2. Canary run A (gate OFF): the standard 16k×d8 20-step bench
   (`runs/overnight_20260807_baseline_shipconfig/run_bench.sh w1-off`), `BT_MOE_PROBS_A2A_COMM` unset. Confirm the
   "present but DISABLED" line appears once.
3. Canary run B (gate ON): same but `export BT_MOE_PROBS_A2A_COMM=1` in the
   trainer env (run_trainer_node.sh export or bench-driver env pass-through).
   Confirm: one `ACTIVE` line, one `armed — second communicator created over
   16 EP ranks` line, and non-empty per-window counter lines.
4. Compare: per-step loss values must be **bitwise identical** across A/B
   (same data order, same seeds — the standard parity harness). Any mismatch
   ⇒ reject the patch.
5. Perf: step time delta at 16k×d8 and (if window allows) a short 131k×d4
   steady-state A/B. Expect ≈ −3.4 s/step on the devbox (CDMC=1: fwd+replay
   only) and ≈ −5.1 s/step with CDMC unset (prod parity; bwd also
   de-serializes). Also grep the trace/logs: probs A2A no longer on the EP
   comm stream (stream 83) between the two token A2As.
6. Memory: expect flat (one extra NCCL communicator, ~tens of MiB). Confirm
   via the standard per-GPU mem poller output.

## Pre-registered risk for T3 (fibonacci, 2026-08-09)

The probs Function's autograd sequence number moves under W1 (it is created
before the shared-expert fc1's ops, not after). Autograd accumulation order at
grad JUNCTIONS (the residual hidden-grad add: routed-path + shared-path; the
probs→router junction) follows execution order among ready nodes — so the
backward can change summation order at those junctions even though every
tensor's math is unchanged. **Diagnosis criterion: if T3 shows window-1 loss
EXACT but window-2+ ULP drift, this is the cause (scheduling, not math).**
Mitigation if seen: re-anchor the probs grad_fn seq-nr to sit exactly where
the old probs A2A sat (`set_tensor_grad_fn_sequence_sr`,
`shared_experts.py:585-593` — the same helper fc2 uses). Do NOT silently
accept tolerance-level drift — that call is Jack's.

## Final field disposition (2026-08-09)

W1 ships as v1. On-box: mechanism 100 % confirmed (probs off the EP stream
900/900; dispatch→combine gap 11.22 → 9.15 ms = −2.07 ms/pass; token A2A
flat; canaries clean); wall −0.46 s/step on this box vs the −3.4 s static
model. Post-mortem chain (memo §6.1): the residual is the backward-window
probs-reverse exposure — hilbert's cut showed the reverse's launch is ~24 ms
early on an idle stream but the kernel waits ~41 ms for the probs grad
(produced post-fc2-dgrad deep in the backward chain). The W1-v2 candidates
are both dead (seq-bump inert — the probs node already runs first by
construction, probe-verified; paired-node archived-unshipped — its
pre-registered ship-gate resolved negative: it would have chained the tokens
reverse to the late probs grad). The real lever (backward-chain reorder) is
worth only ~1–1.5 s/step and is subsumed by W3's lookahead — parked as
option 6 in ../W2V2_DECISION_stub.md. The residual (~0.5 s/step on this box)
is understood, bounded, and accepted.

## Known limits / follow-ups

- CUDA-graph capture of `token_dispatch` is not supported (same as upstream —
  the A2A is outside captured regions); no new guard added.
- W2 (chunked pipeline) will reuse `all_to_all_deferred`/`wait_deferred_a2a`
  and the same telemetry pattern; the probs path becomes a single comm-2 A2A +
  a 131 KB device gather per group (see DESIGN_helmholtz.md §4).
