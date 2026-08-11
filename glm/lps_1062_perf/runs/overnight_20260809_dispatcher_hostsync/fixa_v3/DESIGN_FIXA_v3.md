# DESIGN: FIX A v3 — first-pass-anchored DSA-bwd nonempty probe (hilbert, 2026-08-09)

**Gate:** `BT_DSA_BWD_ASYNC_NONEMPTY_V3=1` (default OFF; distinct from the
parked v2 gate `BT_DSA_BWD_ASYNC_NONEMPTY`) · **Verify:**
`BT_DSA_BWD_ASYNC_NONEMPTY_V3_VERIFY=1` (requires the gate; soaks only) ·
**Files:** `dsa_cudnn_kernels.py` (+267), `dsa.py` (+26), `dsa_kernels.py`
(+11) · **Patch:** `fixa_v3.patch` (against the B/F/A→fixc→w1→w2 stack;
applies cleanly, byte-identical to the reviewed tree) · **Test:**
`test_fixa_v3_probe.py` (Mac-CPU, 33 assertions, ALL PASS; v2 + FIX B suites
re-run green).

## 0. Hypothesis (what v3 is a measurement OF)

FIX A v2 (replay-anchored async nonempty probe) measured ≈0 on-box:
mechanically perfect (312 events, p50 5.5 µs) but the MoE dispatcher's replay
`d2h_event.synchronize()` (p50 77 ms) fired **upstream** of A's read point in
every layer-backward and throttled the autograd thread to GPU-lockstep — both
the probe event and the original 120 ms nonzero drains were cheap at that
point (ATTRIBUTION addendum §4, PATCH_NOTES dependency chain). FIX C removes
the replay syncs. In the post-C regime two things change: (a) the 312 DSA-bwd
drains may become **exposed** (the throttle is gone), and (b) a
replay-anchored probe (v2) would now wait on replay-fwd GPU progress — the
wait relocates back. v3 therefore anchors the probe in the **no-grad first
pass** (seconds before the backward, GPU queue shallow): the event is long
complete whenever the backward reads it, in every regime. **v3 is the test of
whether the drains are now exposed — expectation is measurement, not a
promised win.**

**Expectations temper (2026-08-09/10, boltzmann — from the C′ timed-arm
adjudication; written in per kepler's ratification):** under C′-on the 312
drains already carry ~24 s/step of CPU wait (was ~19 s pre-C′), but that wait
is **97.4% GPU-covered** and wall-flat vs the C′-off control — mostly
ABSORBED wait (the redistributed replay-throttle parking: eventsync+nonzero
wait conserved at 34.21s vs 34.01s), not exposed stall. Expectation for the
ARM-5 timed arm on this box: the exposed fraction is small, and a ~0 wall
result is a valid publishable answer (drains still overlapped). The mechanism
rows stay primary (`eventsync_a_*`: 312 calls, µs-scale, ≤1.5 s, ≤25 >1 ms
for a healthy A-active capture); wall remains informational under the
two-tier rule.

## 1. Mechanism

```
first pass (main thread, no-grad)                    replay (autograd thread, enable_grad)
-----------------------------------                  -------------------------------
dsa.py (plain function, is_grad_enabled=False):      dsa.py (is_grad_enabled=True):
  kick (topk_length > 0).all() as async 1-byte         pop carrier stash[layer_number]
  D2H on the side stream (v2 mechanics)              -> probe (or miss -> fallback)
  stash (event, pinned_buf, flag_gpu) on the           pass probe down:
  packed_seq_params carrier[layer_number]              _run_sparse_attention(...,
TOPK/attention continue as usual                         nonempty_probe_v3=probe)
... microbatch continues; the event completes          dsa_kernels: conditional kwarg pass
  within ms, seconds before the backward               FusedSparseAttentionFunc.apply(...)
                                                       ctx.dsa_nonempty_probe_v3 = probe
                                                     backward: _read_nonempty_probe_v3
                                                       event.synchronize()  # µs — long done
                                                       flag True  -> arange(N) substitution
                                                       (bitwise-identical to nonzero, v2)
                                                       flag False -> original syncing nonzero
```

- **Discrimination:** at the dsa.py level (a plain function),
  `torch.is_grad_enabled()` truthfully separates the passes (False in the
  checkpoint first pass, True in the replay). The v1 trap (grad-mode check
  inside `Function.forward`, always False) does not apply — the kick helper
  carries no grad-mode gate (source-guarded in the test).
- **Keying:** the per-microbatch `packed_seq_params` carrier, keyed by
  `layer_number` — the same lifetime pattern as the DSA top-k holder and
  BT_DSA_CP_LAYOUT_CACHE, and the same carrier FIX C uses (separate attr
  `_dsa_bwd_nonempty_probe_v3`; the two compose, per the DESIGN_FIXC
  composition note). Entries are **popped on fetch** — they die with their
  microbatch; no cross-step staleness by construction. v3 does ** not** depend
  on the FIX C gate or its frames (independent env gate, independent
  discrimination).
- **Plumbing:** the probe rides into `FusedSparseAttentionFunc.apply` as an
  opaque `_NonemptyProbeV3` wrapper object — autograd treats it as a
  non-tensor input (no pytree flattening of the tuple's tensors), costing
  exactly one extra `None` in the backward return arity (asserted in tests).
  `dsa_kernels.run_fused_absorbed_sparse_attention` passes the kwarg **only
  when present**, so backends predating it see a byte-identical call.
- **Eval passes** kick too (grad disabled, no checkpoint): harmless — the
  probes complete and are never read; the carrier dies with the eval
  microbatch. **Non-recompute training** finds empty stashes and falls back
  per layer-pass (loud via the miss counter — this gate targets
  full-recompute configs).

## 2. Safety argument (restated) + verification

The flag describes the **first pass's** `topk_length`; the backward consumes
the **replay's**. The claim: they always agree on emptiness.

**Restatement:** a row is empty iff it has zero valid top-k entries, which
happens iff the row is a THD **padding row** (zero causally-valid KV
candidates) — fixed by `cu_seqlens`/mask bounds. Those are integer metadata,
recomputed bitwise-identically in the replay (same packed layout inputs).
Score-level replay nondeterminism (ULP) **cannot** flip a row's emptiness:
emptiness is candidate-set membership (mask bounds), not score order. The
residual risk is a genuine kernel bug (e.g., the replay's mask path
diverging) — not a numerics-modeled class.

**Verification (two layers):**
1. `BT_DSA_BWD_ASYNC_NONEMPTY_V3_VERIFY=1` (soak-only): on every replay that
   consumes a stashed probe, the replay's forward recomputes the flag from
   its fresh `topk_length_flat` and asserts equality with the first-pass
   value — **raises** on mismatch (loud stop; the structural-emptiness
   assumption is then false on this configuration and the gate must not ship).
2. Loss canary: bitwise vs gates-off at identical `--warmup-datums` (the
   rng-stream rule); drift > 5e-3 = stop.

## 3. Telemetry (WARNING level; the v1 lesson)

- one-time gate-state line (`ACTIVE` / `present but DISABLED`); same for
  VERIFY (plus "ignored" if the main gate is off);
- one-time `armed — first first-pass probe kicked`;
- one-time `first replay hit — first-pass probe consumed`;
- immediate WARNING on every fetch miss and on the first `probe_dropped`
  (fetched but absent on ctx in backward — declined fused path or dropped
  plumbing);
- per-window counters every 312 probe reads (~1 step at 78 layers × 4 mb):
  `kicks / hits / misses / probe_dropped / fallbacks / verifies`.
  Healthy steady state at 4 mb: `kicks=312, hits=312, misses=0,
  probe_dropped=0, fallbacks=0` per step. A present-but-inert patch shows
  `kicks=0` or `misses=312`.

## 4. Risk register

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| Replay emptiness diverges from first pass | ~impossible by the structural argument (§2) | wrong arange substitution → wrong grads | VERIFY soak asserts on every replay; loss canary |
| Probe plumbing dropped (fused path declines after fetch) | low (shapes gate-independent) | silent fallback (status-quo cost) | `probe_dropped` counter + first-occurrence WARNING |
| Gate on without full recompute | config error | fallbacks every layer-pass (no corruption — status-quo path) | loud miss counters; documented scope |
| Eval-pass kicks | certain (by design) | 156 tiny async D2Hs per eval step, never read | negligible; documented |
| 156 in-flight probes/step | certain | a few KB pinned + events | fine (v2 proved the mechanics at the same rate) |
| Both v2 and v3 gates on | misconfig | v3 wins (documented precedence) | backward branches v3-first; note in the log lines |

## 5. On-box recipe

See `ONBOX_VALIDATION.md`.
