# MEMORY LEG DECISION — LPS-1062 overlap campaign (task 5 prep)

> **2026-08-13 late — CORRECTION (W1b fit data, bayes/grothendieck): the
> S_eager constant below was wrong.** K=21 block+K OOMed at 131k (266.3–266.7
> GiB vs 267.69 cap, stage-1 first, both warmup paths). Measured fit, exact:
> peak(K) = peak_full + 2·(40−K)·(e−c) with (e−c) = (266.5−162.2)/38 = 2.745
> → **S_eager ≈ 2.94 GiB/layer/mb, not 2.25**. Root of my error: 2.25 was the
> E1-*selective* constant (core_attn still checkpointed there); block+K eager
> layers are fully eager and retain the DSA attention internals (~+0.7
> GiB/layer/mb). Base (147) and I=2 validated by the exact fit. **Corrected
> K: ≥23.1 → K=24 minimum (~250 GiB predicted), K=25 recommended for first
> boots** (warmup measurement may predate optimizer-state materialization).
> Consequences: dial overlap coverage 37–40% of layers (not 47.5%), dial
> combined prize ~6–10% (not 10–15%), block+K standalone refund ~5–8%; the
> dial A/B memory gate gains a two-stage structure (executor-slope d1 ramp →
> recompute K → d2 canary). The recommendation LOGIC below is unchanged
> (block+K first, dial second, valve composable); only the constants moved.
> — pauli (ex-jacobi)
>
> **2026-08-14 — SECOND CORRECTION (K=25 datapoint; the linear model itself
> is REJECTED).** K=25 measured 265.0 GiB reserved (node-1 binding; leader
> 218.3) vs 244.6 predicted. Reconciliation (full version in my report to
> bayes, W1b revision): (i) the fresh mission-box W1a anchors are d2 162.2 /
> d4 165.0 / d16 182.8 GiB stage-1 reserved — **+20–22 GiB over the old-box
> 143/162 this memo used** (cause unassigned: tree drift / box variance /
> trace-tax on the d16 anchor pair; flagged as a hygiene item); (ii) the
> three points reject any linear stored-set model (3-point fit yields
> garbage constants, residuals ±10 GiB): anchor→K=21 slope 2.75 GiB/layer vs
> K=21→K=25 marginal slope 0.44 GiB/layer, stage gap 11.4→46.7 GiB. Read:
> near the ceiling the stage-1 peak is dominated by K-INDEPENDENT components
> (loss-path stack, cold-pool first-window burst under BT_SKIP_WARMUP=1,
> fragmentation ceiling — "reserved" saturates at ~capacity under pressure
> and stops measuring demand). **Consequences: model-driven K below 25 is
> dead — the dial's memory gate is now a pure empirical ramp (first boot
> K=26–27, walk DOWN under the executor d1 ramp); minimum K today = 25 (fits
> by 2.7 GiB — thin); coverage 35–37.5%; revised dial prize ~12% at d16
> (soft); the valve-stacked-on-dial becomes the coverage-multiplier past
> ~40%.** Outstanding discriminator: the K=25 poller CSV time structure
> (warmup spike vs steady plateau) + the trainer peak-allocated report —
> requested from bayes. — pauli
>
> **2026-08-14 — THIRD NOTE (evidence set landed; the "model rejected" call
> above was itself wrong).** grothendieck's W1b evidence set (CSVs, runlogs,
> the K=25 same-instant triple 253.2 alloc / 256.8 torch-reserved / 265.0
> poller) reconciles everything: (i) the "+20 GiB anchor migration" = ~8 GiB
> metric artifact (driver `/status` = torch max_memory_reserved; poller =
> nvidia-smi; papercut pc_248d82831027) + ~13 GiB REAL uniform base shift
> old-box→mission-box (unassigned; env/tree diff follow-up, not blocking —
> remains pauli's named hygiene item); (ii) the linear model HOLDS where the
> ceiling doesn't bind — stage-0's K=21→K=25 slope is 5.8 GiB/layer =
> 2×(e−c) ✓ — and stage-1 flattens (0.44/layer) because its peak is pinned
> by K-independent components (loss-path fp32 stack + cold-pool burst under
> SKIP_WARMUP + fragmentation at the ceiling). Constants of record (PRIOR
> for the empirical ramp, not a fit): base ≈ 139 torch (mission, d2-class,
> stage-1), S_eager ≈ 2.9–3.1 GiB/layer/mb-set, S_ckpt = 0.19, I=2;
> cold-pool burst ≈ +20 torch; trainer-warmup transient base ≈ +60 (boot#1,
> weight-load unsettled — the trainer warmup is the most dangerous memory
> phase). K=21's OOM points are pre-optimizer-materialization = lower bounds.
> **Floors at the 255 poller line: K=25 fits steady (~251 smi); K=24 razor;
> K=23 needs the burst managed (warm-pool variant boot measures the burst's
> share). Coverage 37.5/40/42.5% at K=25/24/23; dial prize ~12–13.5% at d16
> (soft). Valve-stacked-on-dial load-bearing past K~23. Runbook rule for all
> dial boots: settle-weights-then-warm-pool or graduated driver warmup —
> never cold-pool into a near-ceiling config.** CSV pre-registration of
> record: declining-to-~245–250 = burst theory holds; flat-265 = structural.
> GUARDRAIL (bayes): W1b's DOES-NOT-FIT verdict is CLOSED under tonight's
> protocol — the K=25-fits-steady hypothesis is a NEW experiment (warm-pool
> variant boot inside the dial program's budget), not a retro-appeal; any
> blockK revival gets its own fresh pre-registration after W1c/W2. — pauli
>
> **2026-08-14 — FOURTH NOTE (final, on the complete evidence set).** (a) The
> "anchor migration" hygiene item is CLOSED — it never existed: the W1a
> JSONs carry torch-reserved via `/status`, and the boxes AGREE within ±2.5
> GiB on that metric (mission d2 138.8 / d4 143.0 / d16 163.8 vs old-box
> 141.3 / 143.2 / 161.7). The metric gap at 131k is 8–12 GiB. (b) The REAL
> hygiene finding replacing it: **anchors are step-position-dependent —
> torch-reserved creeps +20.8 GiB across a run's early steps (143.0→163.8
> over d16 steps 6→9) then plateaus flat (d16-ext through step 16). Memory
> A/Bs must compare at matched step positions or at plateau; a window-1 read
> is not a steady read.** (c) The K=25 measurement is **warmup0-ONLY** — the
> run was terminated after warmup0 (canary PASS 12.3242/0.4185; no mains, no
> result JSON), so 265.0/256.8 is the cold-pool first-window peak and steady
> state is unmeasured; the earlier CSV pre-registration is void (no steady
> windows exist). The model fits the warmup0 point exactly (steady 236.7
> torch + 20.1 cold-pool burst = 256.8 = measured). **The warm-pool variant
> boot is the sole remaining discriminator and must run into mains to see
> the plateau; the dial ramp's fit reads must be at plateau or matched
> steps.** — pauli

Author: jacobi (K3), 2026-08-13, for fermi. Source-only Mac-side scouting; no box
contact. Consumes: `pp2cp8ep8/results/OVERLAP_PER_LAYER_RECOMPUTE_DESIGN.md`
(serre), `UPSTREAM_PROPOSAL.md` (serre), NOTEBOOK.md through the lebesgue
hold-for-Jack state, and the vendored trees at `~/Documents/wt-pp2-bringup`
(trainers @ branch `jackrao/lps-1062-pp2cp8ep8`, mcore submodule 57efae08b) —
plus one upstream fetch: TransformerEngine v2.16 `module/grouped_linear.py`
(NVIDIA GitHub), cited below.

## Verdict (one paragraph)

**Build order: (0) block+K partial recompute — zero code, boot it now — then
(1) the per-layer recompute dial (P1) as the overlap program's memory leg;
the offload valve (P2) is demoted from coequal candidate to a composable
follow-up.** The surprise finding of this scout: the dial's memory profile is
reachable TODAY with stock mcore (`recompute: {granularity: full, method:
block, num_layers: K}` — already plumbed through the trainer config), which
both banks the recompute-tax refund (~6–10% step time at d16, no shim, no
flag, no new code) and measures the dial's load-bearing memory constants
before any mcore patch is written. The valve's plumbing is complete but
unbooted, its coverage is too thin to fit 131k standalone (moe_act moves only
~33–50% of the eager bill), and its headline "expert_fc1 coverage gap" is not
a bug — it is the correct consequence of LoRA-frozen routed experts (TE
source-verified). The dial remains the right structural leg: deterministic
fit by construction, ~10–15% combined prize with the shim, composable with
offload later (offload the eager layers, push K lower).

---

## 1. Backpressure-valve plumbing: COMPLETE, NEVER BOOTED

The valve exists as three committed, pushed commits on
`origin/jackrao/lps-1062-pp2cp8ep8` (tip-ward of the reviewed 73c24b00/b8d868ff
state; weierstrass's branch-hygiene note already flags this region as
experimental-P2 — PRs cut at explicit commits, never the tip):

| commit | content |
|---|---|
| `f2407a10` | `ActivationOffloadConfig` (extra=forbid: enabled / modules / min_tensor_size / fraction / delta_bytes_across_pp_ranks) hung off `TrainerControllerConfig`; `_build_config` writes the five mcore fields onto the provider directly; trainer-side legality guardrails re-encoded (mcore validators run at provider construction, before the write); 25/25 Mac stub-harness tests |
| `6d8b22da` | disposable-review follow-ups: validator-timing rationale corrected (bridge `finalize()` re-runs mcore validators on current fields), dead code removed, provider-field mapping pinned by test |
| `4e7d5d3e` | **THE VALVE**: `activation_offload.max_inflight_offloads` → `provider.fine_grained_offloading_max_inflight_offloads` — mcore's only in-tree backpressure: caps pending D2H groups per module name; main stream waits on the oldest event past the cap; 0 = wait on every commit (`fine_grained_activation_offload.py:1110-1121`, `_drain_offload_pending`). PLUS the engagement probe: one `logger.warning` after `get_model` dumping the baked flags (`offload_attn_norm`, `self_attention.offload_core_attention`, `experts.offload_expert_fc1`) — the only pre-step evidence the flag state reached the model. 32/32 tests |

**Critical provenance:** both offload boots tonight (the 131k L0b-mem abort AND
the 32k discriminator) ran tree `6d8b22da` — i.e. **pre-valve**. The valve
commit `4e7d5d3e` (Aug 13 01:32) postdates both. So: the fix designed for the
first-step-burst class has **zero hardware evidence**. The 32k "offload
engages" verdict and the 131k "aborted at >255 GiB" verdict were both measured
without the valve.

Mechanics note for the probe: the valve throttles the *producer* (main stream
waits on D2H events per group name). The first-step burst mechanism is
plausibly pinned-allocation-bound, not copy-bandwidth-bound: MoE offload groups
are hardcoded OFF the CPU pool (`fine_grained_activation_offload.py:362-366` —
"shapes not known in advance… cuda graph compat"), so every layer's offload
calls `torch.empty(..., pin_memory=True)` fresh. At 32k the −30% step cost
decomposes as ~0.6s copies (2×18.7 GiB at ~60 GB/s) vs ~2.1s residual
(alloc+sync) — i.e. **the bulk of the observed stash cost is the pinned-alloc
path, which the pool would absorb after step 1** (pad-to-131k makes shapes
constant — exactly the pool's operating regime; we run no full-iteration cuda
graphs). A ~3-line probe patch (let MoE names use the pool, or gate the
hardcode behind a config) is the cheap follow-up; it touches frozen vendored
mcore → needs the scoped freeze-exception process (precedent: gauss's
cache-clear probe, sanctioned by cauchy with hash-before/restore-after).

## 2. expert_fc1 = 0.00 GB: ROOT-CAUSED — correct behavior, no fix

**Claim: the coverage gap is not a wiring bug and needs no fix. Under
LoRA-frozen routed experts, the fc1 input is genuinely never saved for
backward, so there is nothing to offload.**

Evidence chain (all three links verified):

1. **Our LoRA excludes routed experts.** `lora_targets.py:143-174` (GLM-5.2
   `dsa` branch): targets = attention projections, dense first-k MLP, shared
   experts, LM head; routed `*.mlp.experts.linear_fc{1,2}` deliberately
   EXCLUDED (a ~116k-tensor / ~61 GB adapter that also breaks vLLM serving).
   → routed expert `linear_fc1/linear_fc2` weights have `requires_grad=False`.
2. **TE saves no input when the weight is frozen.** TE v2.16
   `_GroupedLinear.forward` (fetched from NVIDIA/TransformerEngine tag v2.16):
   `weight_requires_grad = weights[0].requires_grad`, then in the
   `is_grad_enabled` block: `if weight_requires_grad: … keep inputmats … else:
   inputmats = [None] * num_gemms` before `ctx.save_for_backward`. Frozen
   weight → the input is replaced by None → **nothing enters autograd's
   saved-tensor set except weights (Parameters, excluded from offload by
   `_can_manage_tensor_for_offload`) and biases**. dgrad needs only the
   weight; wgrad (which would need the input) never fires.
3. **The observed table is exactly this.** mcore's offload group captures only
   what saved-tensors hooks see during the `with expert_fc1_manager:` window
   (`experts.py:701-712`); with nothing saved, the group commits empty → 0.00
   on every rank. `moe_act` fires because activation backward always needs its
   own input — parameter-free — so `fc1_output` (plus the probs-mul operands)
   is saved regardless of frozen weights (15–23 GB/rank at 32k ✓).

Corollaries:
- **Drop `expert_fc1` from future offload module lists** (inert for offload
  bytes; its only residual effect is the `forced_released_tensors` eager
  `resize_(0)` of the fc1 input storage — marginal, safe, not worth a row in
  the config). Not a P2 engineering item; close it.
- **The actual uncovered retained tensors** (nobody offloads them today):
  dispatcher combine-side saves (expert outputs ≈ 0.2 GiB/layer/mb at 131k +
  probs) and attention-side saves outside the core_attn checkpoint (qkv-proj
  input, proj input ≈ 0.2 GiB/layer/mb each). Attention hooks exist but are
  inert on this model: `core_attn` offload is mutually exclusive with
  core_attn recompute by construction (`attention.py:1503-1519` — the
  checkpointed branch never enters the manager), and the `attn_proj` hook
  sites (`attention.py:1577`, `multi_latent_attention.py:461`) are not on
  GLM-5.2's `AbsorbedMLA` forward path. If the valve leg is ever built out,
  the dispatcher-combine hook is the first extension site (single
  `off_interface` wrap in the token dispatcher; correctness-sensitive
  placement, force-release discipline).

## 3. Coverage math: why the valve cannot fit 131k standalone

Measured constants (campaign-measured, cited): `S_eager ≈ 2.25 GiB /
layer / in-flight mb` (E1 delta: 258.8 − 169 over 40 stage-1 layers; upper
bound per serre), `S_ckpt ≈ 0.19 GiB` (16,384 tok × 6144 × bf16), `I = 2`
in-flight (M=N, PP2: min(M,PP)), stage-1 L=40. Fixed-wheel bases: d4 peak 143
GiB, d16 peak 162 GiB (full recompute; stored inputs ≈ 15 GiB included).

moe_act offload coverage (the only firing module): measured 15.1–23.0 GB/rank
at 32k d2 over 80 layer-mb instances → 0.19–0.29 GiB/layer/mb → ×4 tokens at
131k → **0.75–1.15 GiB/layer/mb ≈ 33–51% of S_eager**.

Valve-only at 131k (selective core_attn + full moe_act offload, I=2):
stored = 2×40×2.25 − 2×40×(0.75..1.15) = 180 − (60..92) = **88–120 GiB** vs
~96 GiB headroom → borderline at best, and the offload-margin/fraction
machinery deliberately keeps trailing groups on GPU (reload-block avoidance),
so effective coverage is below the optimistic end. With the two cheap coverage
extensions (dispatcher-combine + attn hooks, ~0.4 GiB/layer/mb more) it fits
(~200–220 GiB peak) — but that is 2–3 new mcore hook sites plus the pool
patch, i.e. the valve stops being cheap exactly when it starts fitting.

Contrast — the dial at K=21: stored = 2×(21×0.19 + 19×2.25) = 93.5 GiB →
**peak ≈ 221 GiB at d4 / ≈ 240 at d16 (fixed wheel)**. Fits by construction;
K is *chosen* to fit. (Old-wheel arithmetic for the record: 272 GiB at d16 —
razor-thin; the 1.27.0 wheel's −32 GiB workspace win is what makes K≈21
comfortable. Nice inversion: the wheel fix didn't move the E1 wall, but it
does buy the dial's headroom.)

> **CORRECTED (W1b fit, see top note): S_eager = 2.94, not 2.25 — this
> paragraph's K=21 OOMed (266.5 GiB). Corrected: stored at K=24 =
> 2×(24×0.19 + 16×2.94) = 103.2 GiB → peak ≈ 250 GiB (d16-class); K=25 →
> ≈245. The dial still fits by construction; the construction's constant was
> wrong.**

## 4. The dial estimate, honestly

serre's design doc is unusually well-built: mechanism verified at source, the
coexistence precedent (dense-layer no-op nodes, unequal f/b plans) is real and
correctly cited, the LoRA silent-zero-grad trap is identified with a
construction-level mitigation, and the verification checklist is written. My
independent read of the same vendored tree found no errors in the citations I
re-checked (`transformer_config.py:2632-2642` asserts, `recompute.py` block
branch, `transformer_block.py:624` dispatch, bridge
`maybe_enable_recompute_inputs_grad` LoRA patch).

Effort: the patch itself is the claimed ~140 LoC mcore + ~20 LoC
bridge/trainer mirror. The honest critical path is validation, not writing:
the grad-equivalence test (opaque layer vs stock checkpoint, adapter grads
nonzero — the LoRA trap) needs a GPU; megatron does not import on Darwin, so
every test iteration is a box/CI round-trip. **Branch-quality prototype:
1.5–2 days** (vs serre's 1d — the delta is test round-trips and the K
calibration loop). **Upstream-grade: 2–3 days** stands. Risks in order:
(§6.1) silent-zero-adapter-grads — designed around, caught by the 1-GPU test;
(§6.2) checkpoint-inside-schedule RNG/stream semantics — `tensor_parallel.
checkpoint` owns RNG; testable at 1 GPU; (§6.3) S_eager mis-set → OOM at ramp
— **retired for free by the block+K probe below** (it measures the true
per-layer stored delta on hardware before the dial boot depends on it).
**Postscript: this is exactly what happened — the probe caught the wrong
constant (2.25 → 2.94) at the cost of one OOM'd boot, before any dial boot
depended on it. The validation-instrument role worked as designed. Note the
memo did flag the right sensitivity (K=21 needed S_eager ≤ ~2.6 at d4); the
central value was wrong, not the bound.**

Dependency precision: the dial's *unit-level* validation (plan structure,
grad equivalence) is shim-free at 1 GPU. Its *purpose* (the 131k overlap A/B)
is gated on lebesgue's contract shim — the fine-grained plan builder only runs
under the big flag, and the flag only runs through the shim. The block+K probe
has NO dependencies at all.

## 5. The third leg: block+K partial recompute (zero code — the build-first)

`recompute: {granularity: "full", method: "block", num_layers: K}` is stock
mcore (`recompute.py` block branch: checkpoints the first K layers of each
stage, remaining L−K eager), already exposed in the trainer config
(`control.py:105-110` → `megatron_config.py:257-264`), and LoRA-safe on the
stock path (bridge `maybe_enable_recompute_inputs_grad` fires for
block/uniform — the silent-zero-grad trap exists only in the fine-grained
executor, which this never enters). No shim, no flag, no patch.

Why it wins twice:

1. **It is a perf lever on the CURRENT stack, today.** Tonight's 918/984
   headlines pay full recompute on all 40 layers. Block+K=21 skips the
   recompute replay on 19/40 layers: ~6% of step time from the compute refund
   alone, plus it deletes the replay's a2a outright on those layers (the L3
   decomposition measured the replay's a2a at 11.7s of the d16 step, 37% of
   the exposed total — ~47% of that dies too). **Honest estimate: +6–10% step
   time at d16, +4–8% at d4, memory-limited not correctness-limited.**
2. **It is the dial's memory validation instrument.** One K sweep measures the
   true per-layer eager stored delta on hardware (S_eager has only the E1
   upper bound today), which sets the dial's K from measurement — retiring the
   dial's risk §6.3 before the dial exists.

Starter artifact (written, this folder's sibling):
`pp2cp8ep8/configs/trainer_pp2cp8ep8_131k_blockK25.json` — the headline config
+ `recompute: {granularity: full, method: block, num_layers: 25}`. ~~K=21~~
(OOMed, see the correction note at top); K=25 predicted peak ≈ 245–249 GiB on
the measured constants. The d1/d2 ramp catches a miss
early (I=2 already at d2, so the d2 read IS the fit read; d4 adds nothing
memory-wise).

### Pre-registered bars — block+K probe (filed before any boot)

- **Fit:** d1 boot → d2: peak reserved ≤ 255 GiB with 2 in-flight. ~~Predicted
  220–240~~ **(K=21 OOMed at 266.5 — see top correction; retry at K=25,
  predicted ≈245–249)**. Miss → recompute K from the measured per-layer delta
  (K' = 40 − (40−21)×(budget−**same-rung full-recompute anchor**)/(Δ =
  peak(21)−anchor) — the base is the K=40 anchor, not the stripped base;
  round UP), one retry.
- **Perf (primary):** d4 vs the fixed-wheel anchor (~878–886 tok/s/GPU):
  within-boot control pairs must agree ±2% (tonight's d4 noise ran 3.0–3.8%
  on two occasions — if controls trip, the run discards and repeats);
  win bar > anchor + 2%. Prediction: +4–8% (≈ 915–955) **— restated at K=25:
  15/40 layers skip the replay → ~+4–7%**. d16 vs 984 as the
  number-of-record rung if d4 passes: prediction +6–10% (≈ 1045–1080)
  **→ restated: ~+5–8% (≈ 1030–1060)**.
- **Canary:** loss 12.2–12.4 band + gn comparable to the fixed-wheel d2
  (0.36–0.49). Recompute-path changes should be numerics-neutral; drift
  >5e-3 = STOP (would indict a checkpoint-RNG or LoRA-patch interaction).
- **Measurement deliverable (feeds the dial):** peak reserved at K=21 (and
  K=18/24 if margin allows) → per-layer eager stored delta table →
  `S_eager` measured, dial K equation calibrated.

### Pre-registered gates — the dial (when built; for lebesgue's A/B design)

- 1-GPU grad-match: opaque layer vs stock `checkpointed_forward`; adapter
  grads nonzero and matching within tolerance (the LoRA trap test, K=1).
- d1 memory ramp: per-layer stored table logged; K set from the block+K
  measurement, not the 2.25 bound. **AMENDED (bayes, ratified): the gate is
  now TWO-STAGE — the d1 ramp measures the per-layer slope UNDER THE EXECUTOR
  (the plain-path 2.94 does not transfer verbatim; the executor's per-node
  detach/free machinery can shift it either way), K is recomputed from the
  executor-measured slope, and only then does the d2 canary boot at that K.
  A single pre-registered K without the executor re-measure is
  known-insufficient evidence.**
- d2 canary: 12.2–12.4 + gn comparability vs the M=N full-recompute baseline.
- d4 A/B vs **block+K** (not vs 918): that pairing isolates the overlap gain
  from the recompute refund both arms share. Win bar: > block+K + 2% with
  ±2% control gate. Trace check: comm-stream dispatch/combine overlap on
  eager layers, idle on checkpointed layers (expected, not a bug).

### Valve probe (only if a box sits idle; NOT the primary leg)

One boot, ~30–45 min, tree = branch tip incl. `4e7d5d3e`: selective
(core_attn) + offload modules `[moe_act]` (drop inert expert_fc1) +
`max_inflight_offloads=1` + NVTE_CPU_OFFLOAD_V1=1 + expandable_segments +
BT_SKIP_WARMUP=1, 131k d2. Reads: (i) survives past the L0b-mem abort point
(>255 GiB line) = valve tames the burst; (ii) engagement table prints at 131k
(the `4e7d5d3e` warning probe gives pre-step evidence now); (iii) steady step
cost. Prediction: fits-but-slow (borderline memory per §3; stash cost until
the pool patch). Value = the offload-stack-on-dial data point, not a
candidate leg. Pool-patch sketch (freeze-exception required): let
`OffloadTensorGroup` use the CPU pool for MoE names when shapes are
constant — `fine_grained_activation_offload.py:365` hardcode, ~3 lines +
config gate.

## 6. Decision summary for fermi

| leg | effort to evidence | prize | risk | verdict |
|---|---|---|---|---|
| **block+K config probe** | zero code; 1 box slot (~30 min) | +6–10% step at d16, standalone, no shim | low (memory ramp-guarded; numerics-neutral class) | **BUILD/RUN FIRST** |
| **per-layer dial (P1)** | 1.5–2d branch-quality (+ shim dependency for the A/B) | ~10–15% combined with shim at d16; composable with offload | medium (LoRA trap — designed around + tested; S_eager — retired by the probe) | **the overlap memory leg** |
| **offload valve (P2)** | plumbing done; probe = 1 boot; fit needs +2–3 mcore hook sites + pool patch | up to ~full a2a coverage IF stacked on the dial; standalone borderline OOM at 131k | medium-high (burst fix unproven; −30% stash cost until pool patch; LoRA+offload upstream-unvalidated) | **composable follow-up, not coequal** |

Recommended sequence: block+K probe at the next box slot (it needs nothing
from anyone) → dial build starts in parallel (K from the probe's measurement)
→ valve probe only in idle box time → endgame = shim + dial, optionally
+ offload on the eager layers.
