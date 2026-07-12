# Nemotron-3-Super memory profiling log

Live profiling of `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` peak GPU
memory vs sequence length, captured with `sft_driver.py --source synthetic`
(`--reset-peak` on → each number is the peak for THAT run, not the cumulative
high-water mark). `peak_alloc` = `torch.cuda.max_memory_allocated` on rank 0,
read from `/status`.

Box: B200, single node, **TP=8, PP=1, CP=1, EP=8** (8 GPUs), bf16 LoRA rank 16,
micro-batch 1. 178 GiB usable HBM/GPU. (PP=1 chosen for profiling because at
PP>1 every forward is padded to `max_seq_len`, so memory wouldn't track seq_len.)

## Experiment 1 — does `recompute_granularity` matter? (answer: no effect)

Same config, toggling the controller's `provider.recompute_granularity`
between `"full"` (committed default) and `None` (disabled).

### recompute = "full" (ON)
| seq_len | peak_alloc (GiB) |
| ------- | ---------------- |
| 4 096   | 40.23 |
| 8 192   | 50.47 |
| 16 384  | 71.31 |
| 24 576  | 92.22 |
| 32 768  | 113.26 |
| 40 960  | 134.32 |

Fit: **y ≈ 0.002562·x + 29.84 GiB** (R²≈1.0), ~2.56 MiB/token/GPU.

### recompute = None (OFF)
| seq_len | peak_alloc (GiB) |
| ------- | ---------------- |
| 1 024   | 32.52 |
| 2 048   | 34.98 |
| 4 096   | 40.25 |
| 6 144   | 45.52 |
| 8 192   | 50.87 |
| 12 288  | 61.41 |
| 20 480  | 82.50 |
| 28 672  | 103.40 |
| 36 864  | 124.70 |

Fit: **y ≈ 0.002572·x + 29.89 GiB** (R²≈1.0), ~2.57 MiB/token/GPU.

### Conclusion
ON and OFF are identical to within 0.4%. **Megatron's `recompute_granularity`
/ `recompute_method` / `recompute_num_layers` provider settings are inert for
this NemotronH hybrid model** — toggling them changes neither the slope nor the
intercept. The ~30 GiB intercept is the TP=8 weight shard; the ~2.57 MiB/token
slope is whatever the model stores per token. Single-node OOM wall ≈ **57.5k**
tokens ((178 − 29.9)/0.002572).

Open question being investigated: is recompute simply not engaging (so 2.57 is
the *un-checkpointed* cost and real checkpointing could cut it severalfold), or
is it forced on internally (so 2.57 is already the floor)? See the recompute
investigation; further experiments appended below.

## Experiment 2 — selective recompute (attention + MoE) WORKS

Root cause of Experiment 1 (via code research): NemotronH runs Megatron's
`HybridStack`, which has **no** full/layer recompute implementation — only
`TransformerBlock` does. Worse, `recompute_granularity="full"` also disables the
selective paths (which require `granularity == "selective"`). So the committed
"full" setting gave **zero** recompute anywhere.

Fix (config-only, no vendor patch): switch to **selective** recompute for the
submodules that DO honor it inside `TransformerLayer`/`MoELayer`:

```python
provider.recompute_granularity = "selective"
provider.recompute_modules = ["core_attn", "moe", "moe_act"]
provider.recompute_method = None
provider.recompute_num_layers = None
```

Same config (TP=8, PP=1, CP=1, EP=8, single node):

| seq_len | peak_alloc (GiB) |
| ------- | ---------------- |
| 8 192   | 39.66 |
| 16 384  | 49.74 |
| 24 576  | 59.61 |
| 32 768  | 69.13 |
| 49 152  | 88.34 |
| 65 536  | 107.21 |
| 81 920  | 125.97 |
| 98 304  | 145.50 |

Fit: **y ≈ 0.001174·x + 30.04 GiB** (R²≈1.0), ~1.17 MiB/token/GPU.

### Result
Selective recompute roughly **halves** the per-token activation slope
(2.57 → 1.17 MiB/token); intercept unchanged (~30 GiB weights). Single-node OOM
wall moves from **~57.5k → ~126k** tokens. So attention+MoE layers were ~half
the per-token activation; the remaining ~1.17 MiB/token is the Mamba/SSM layers,
which have no recompute implementation in this Megatron pin (would need an
upstream `HybridStack`/`MambaLayer` checkpointing patch to reduce further).

### Comparison (per-token slope, single node TP=8)
| recompute setting | slope (MiB/tok) | single-node cap |
| ----------------- | --------------- | --------------- |
| "full" (inert) / None | 2.57 | ~57.5k |
| "selective" (core_attn,moe,moe_act) | 1.17 | ~126k |

### Implication for the 131k golden config
At PP=4 the measured peak was 100.79 GiB with the inert recompute. Selective
recompute should cut the activation portion ~2×, so the PP=4 131k peak should
drop substantially — and **131k may now fit on fewer nodes (PP=2 = 2 nodes)**.
Worth re-profiling PP=2 + selective before finalizing the golden node count.

## Experiment 3 — does adding `layernorm` to recompute_modules help?

Per code research, the only additional valid module that applies to Super's
layer pattern (M/E/* — no dense `-` layers) is `layernorm` (recomputes the MoE
`pre_mlp_layernorm` on the 44 `E` layers). `mlp` is a no-op (no dense layers),
`mla_up_proj` is N/A (no MLA, would raise), `shared_experts` is redundant once
`moe` is on. Tested `["core_attn","moe","moe_act","layernorm"]`:

| seq_len | selective (no LN) | selective + layernorm |
| ------- | ----------------- | --------------------- |
| 8 192   | 39.66 | 39.36 |
| 16 384  | 49.74 | 49.13 |
| 32 768  | 69.13 | 68.21 |
| 65 536  | 107.21 | 105.61 |

Slope 1.174 → **1.155 MiB/token** (~1.6% lower; ~1.6 GiB saved at 65k, ~3 GiB at
131k). Small but free (layernorm recompute is cheap output-discarding). Final
NemotronH recompute set: **`["core_attn","moe","moe_act","layernorm"]`**.

The remaining ~1.155 MiB/token is the **Mamba/SSM layers**, which have no
recompute implementation in this Megatron pin (`HybridStack`/`MambaLayer`) — that
floor can only be lowered by an upstream patch adding Mamba-layer checkpointing.

## Experiment 4 — context parallelism (CP) on 8 GPUs

On a single 8-GPU node, `world = TP×PP×CP×DP = 8`, so raising CP forces TP down
(TP8/CP1 → TP4/CP2). Wired CP properly for this test by adding the missing
`get_batch_on_this_cp_rank` sequence slice in the CE forward step
(`_slice_batch_for_cp`), with `AttnBackend.fused` for CP>1 (FA4 has no CP). The
model builds (Megatron's `MambaContextParallel` handles the SSM via all-to-all).
Selective recompute + layernorm, single node:

| seq_len | TP8/CP1 peak | TP4/CP2 peak |
| ------- | ------------ | ------------ |
| 8 192   | 39.37 | 39.34 |
| 16 384  | 48.96 | 47.34 |
| 24 576  | 58.40 | 55.07 |
| 32 768  | 67.59 | 62.45 |
| 40 960  | 76.75 | 69.85 |
| 65 536  | (—) | 92.46 |
| 98 304  | (—) | 122.98 |
| 131 072 | (—) | **154.85 (fits!)** |

A fine 1k-step TP4/CP2 sweep from 98 304 → 131 072 is perfectly **linear** (122.98
→ 154.85 GiB, no blowup, no OOM). 132 096 > `max_seq_len`=131072 so it's clamped
(bogus 46 GiB) — ignore it. (An earlier one-off "131k connection-refused" was a
**transient**, not an OOM — re-running fits at 154.85.)

- TP8/CP1 fit: **y ≈ 1.14·x(k) + 30.0 GiB**, single-node cap ~130k.
- TP4/CP2 fit (8k–131k): **y ≈ 0.95·x(k) + 31.5 GiB** — slope ~**17% lower** than
  CP=1, baseline only ~1.5 GiB higher (TP4 weight penalty stays small because
  EP=8 shards the expert weights regardless of TP). Single-node cap ~155k.

Findings:
1. **CP=2 fits 131k on a SINGLE 8-GPU node** (154.85 GiB), linearly — vs the
   validated PP path needing TP8/PP4 = 4 nodes. The ~17% lower slope is the
   attention sequence-sharding; Mamba's all-to-all (`pre_conv_ssm`) adds a
   transient but it does NOT blow up (linear to full context).
2. **Correctness gap (the blocker).** On deterministic data the CP=2 loss
   (11.76 @ 8k) doesn't match CP=1 (11.65) to precision (~1% off); a correct CP
   path should match to ~3 decimals. The slice is right but the loss/grad
   normalization isn't CP-aware (chunked LM head + DP-only loss reduction ignore
   the CP token split), so CP>1 training would be subtly mis-scaled.
3. **Costs TP** on a fixed node.

**Verdict:** CP=2 is a genuinely promising *memory* lever — it fits 131k on a
single node (4× fewer GPUs than PP=4) and scales linearly. BUT it is **not a
trustworthy training config yet** because of the ~1% loss/grad correctness gap;
that needs a CP-aware loss normalization fix + numerical parity before use. Until
then, **PP=4 remains the correct, validated lever for 131k**. The experimental
`_slice_batch_for_cp` was NOT committed.

## Context parallelism (CP>1): status & remaining work

**Committed (enough for the memory profiling above):**
- `_slice_batch_for_cp` in `megatron_controller._make_ce_forward_step` — adds the
  `get_batch_on_this_cp_rank` zig-zag slice so each CP rank processes seq/CP.
  No-op at CP==1. `AttnBackend.fused` for CP>1 is already wired (FA4 has no CP).
- This makes CP>1 shard activations correctly (memory numbers in Experiment 4
  are real). It is **NOT** correct for training — committed as EXPERIMENTAL.

**Remaining work to make CP>1 a correct, production training config (TODO):**
1. **CP-aware loss/grad normalization (the blocker).** Loss reporting
   (`_loss_report`) and the grad-scaling token denominator (`loss_tokens` →
   `execute_optim_step` `scale_gradients(1/loss_tokens)`) reduce over the **DP
   group only**. Under CP they must use the **CP-inclusive** group
   `mpu.get_data_parallel_group(with_context_parallel=True)` so the per-token
   mean (and thus grad scale) divides by the global token count across CP ranks.
   Confirm `finalize_model_grads` reduces LoRA grads across CP (it uses the
   with-CP DP group in Megatron, but verify). This is the source of the measured
   ~1% CP=1-vs-CP=2 loss gap.
2. **CP-safe chunked LM head.** `chunked_lm_head.py` drops the last *local*
   position (`hidden_states[:-1]`/`labels[:,:-1]`); under zig-zag CP that drops
   CP positions instead of the single global-final token. Switch to mask-based
   handling (rely on the `-100` label) so it's correct under sharding. (Small
   effect — ~CP/S of positions — but fix for correctness.)
3. **RL forward step.** Apply the same slice in `_make_rl_forward_step` and slice
   `sampling_logprobs`/`advantages`/`temperatures` consistently.
4. **Guardrails.** Validate `max_seq_len % (2*CP) == 0` and
   `nheads % (TP*CP) == 0` (Mamba) at config load with clear errors.
5. **Validation gate.** Add a CP=1 vs CP=2 **loss-parity AND grad-norm/training-
   curve parity** test on a small deterministic input; require <1e-3 match before
   enabling CP>1 in any golden/JSON config.

Until #1–#5 are done, keep CP=1; **PP=4 is the validated lever for 131k**.

## Where the setting lives (persisted)
`RLControllerConfig.recompute` (a `RecomputeConfig` block in the trainer JSON),
default `granularity="full"`. For dense/transformer-MoE models (Qwen, Kimi) the
provider's `TransformerBlock` honours both `full` and `selective` natively. For
hybrid NemotronH models `HybridStack` ignores `recompute_granularity` entirely,
so `megatron_controller._apply_hybrid_layer_recompute` (called right after
`get_model`) emulates it by monkeypatching the decoder layers' `forward` with
non-reentrant `torch.utils.checkpoint`:
- `full` → wraps all 88 layers (40 MambaLayer + 48 TransformerLayer) = true
  full layer recompute. Now actually saves memory (Experiment 6).
- `selective` → native selective handles attn/MoE; set
  `BT_MAMBA_LAYER_RECOMPUTE=1` to additionally wrap the Mamba layers (Route A,
  Experiment 5).
The controller still calls `RecomputeConfig.apply_to_provider` so the provider
flags are set for the transformer layers' native selective path.

## Experiment 5 — how much do the Mamba layers actually cost? (Route A)

Direct measurement on the single-node **TP8/PP1/EP8/ETP1/CP1** box (8×B200, 178
GiB usable, bf16 LoRA r16, micro-batch 1), using a layer-level recompute toggle
as the instrument: store-Mamba vs recompute-Mamba at matched seqlen. The 40
Mamba layers (of 88) are exactly what `HybridStack`/`MambaLayer` never
checkpoint. `BT_MAMBA_LAYER_RECOMPUTE=1` wraps each `MambaLayer.forward` in
non-reentrant `torch.utils.checkpoint` (safe here: dropout=0 → deterministic
recompute, no RNG to desync). peak_alloc/GPU (GiB), `--reset-peak` per run:

| seq | selective (Mamba stored) | selective + Mamba RC | **Mamba activation cost** |
| --- | --- | --- | --- |
| 2048  | 32.10 | 30.75 | 1.35 |
| 4096  | 34.47 | 31.71 | 2.76 |
| 8192  | 39.13 | 33.65 | 5.48 |
| 12288 | 43.87 | 35.49 | 8.38 |
| 16384 | 48.91 | 37.15 | 11.76 |
| 24576 | 59.03 | 41.66 | 17.37 |
| 32768 | 69.13 | 46.15 | 22.98 |

selective+Mamba extends cleanly to long context: 49152→53.86, 65536→61.24,
98304→75.71, **131072→89.61**.

- selective slope **1.236 MiB/tok**, intercept 29.36 GiB.
- selective+Mamba slope **0.472 MiB/tok**, intercept ~30.2 GiB.
- **Mamba activations = 0.764 MiB/tok/GPU, linear (~0 intercept) = 62% of the
  selective per-token slope; ~92 GiB/GPU at 131K.**
- Single-node wall: selective ~123K → selective+Mamba ~320K tokens.
- **131K fits a SINGLE node at TP8/CP1 (89.61 GiB) — no CP, no PP.**

So the Mamba layers are the majority of the per-token activation and the entire
reason 131K didn't fit at TP8/CP1. (The earlier "~1.07 MiB/tok ≈ Mamba" estimate
over-attributed; the clean differential says Mamba is 0.764, the rest is the
selective leftover tail + residual floor — see Experiment 6.)

## Experiment 6 — `full` implemented for the hybrid stack (all layers)

Same box/grid, config `"recompute": {"granularity": "full"}`. The controller now
wraps all 40 MambaLayer + 48 TransformerLayer layers. peak_alloc/GPU (GiB):

| seq | full RC | Route A (ref) | tail reclaimed vs Route A |
| --- | --- | --- | --- |
| 8192   | 32.39 | 33.65 | 1.26 |
| 16384  | 34.91 | 37.15 | 2.24 |
| 32768  | 39.89 | 46.15 | 6.26 |
| 65536  | 48.77 | 61.24 | 12.47 |
| 98304  | 57.64 | 75.71 | 18.07 |
| 131072 | 67.61 | 89.61 | 22.00 |
(full also: 2048→30.41, 4096→31.08, 12288→33.65, 24576→37.40, 49152→44.42)

- full slope **0.292 MiB/tok**, intercept 30.15 GiB. Losses match Route A to ~3
  decimals (9.654 vs 9.653 @2k; 0.535 vs 0.533 @131k).
- **131K fits a single node at 67.61 GiB** (vs 89.6 Route A, vs ~188 OOM
  baseline). Single-node wall ~**519K tokens** (allocated fit; lower in practice
  once reserved + fragmentation are counted).

### Three-way decomposition of the selective 1.236 MiB/tok per-token slope
| component | MiB/tok | share | recomputed by |
| --- | --- | --- | --- |
| Mamba SSM layers (40) | 0.764 | 62% | Route A / full |
| transformer tail on the 48 attn/MoE layers (QKV/out proj, router, dispatch, residual adds) | 0.181 | 15% | full only |
| irreducible floor (88 boundary residuals + 1-layer recompute transient) | 0.292 | 24% | nobody |

### Single-node walls (TP8/PP1/CP1, 178 GiB usable, ~30 GiB weight intercept)
| config | slope MiB/tok | 131K peak | max seqlen |
| --- | --- | --- | --- |
| selective (committed) | 1.236 | ~188 (OOM) | ~123K |
| selective + Mamba (Route A) | 0.472 | 89.61 | ~320K |
| **full (all layers)** | **0.292** | **67.61** | **~519K** |

**Why this matters for Nemotron-3-Ultra (550B, same NemotronH arch, bigger):**
both the weight intercept and the per-token slopes grow with the larger model,
so full layer recompute (now functional for the hybrid stack) is the lever that
keeps long-context SFT on-node. For Super at 131K, selective+Mamba already fits
with headroom; full is the lower-memory / slightly-more-compute option and the
one Ultra will need.

## Experiment 7 — parity test (full recompute vs eager): **PASSES**

Trajectory parity from identical fresh init, seq 16384, lr=1e-4, 8 steps,
deterministic synthetic data. Ground truth = `no recompute`
(`BT_DISABLE_HYBRID_RECOMPUTE=1` + `granularity=full`, which reproduces the
original eager/inert path). Control = two independent no-recompute restarts.

Methodology note (important): the comparison is only valid if every run has the
**same optimizer history**. A frozen-weight (lr=0) loss-average test does NOT
test recompute at all — the reported loss is from the (unchanged) forward pass,
so it matches to 8e-6 regardless; recompute bugs live in the backward/gradients
and only a real-LR **trajectory** exposes them. AND the trajectory runs must all
start from a fresh optimizer: running lr=0 steps first warms Adam's m/v moments
and shifts steps 2+ (bias-correction only fixes step 1), which silently
confounds the comparison.

| step | no-rc #1 | no-rc #2 (control) | full (fresh) |
| --- | --- | --- | --- |
| 1 | 12.2509 | 12.2515 | 12.2533 |
| 2 | 12.2045 | 12.2023 | 12.2023 |
| 3 | 12.0720 | 12.0725 | 12.0733 |
| 4 | 11.9435 | 11.9472 | 11.9440 |
| 5 | 11.8260 | 11.8260 | 11.8239 |
| 6 | 11.6961 | 11.6977 | 11.6974 |
| 7 | 11.5930 | 11.5934 | 11.5939 |
| 8 | 11.5126 | 11.5129 | 11.5145 |

- Two no-recompute restarts agree to a max of **0.0037** (the control noise
  band) → deterministic init + reproducible kernels.
- **full (fresh) vs no-recompute: max deviation 0.0021 — inside the control band
  → parity holds.** Full recompute is numerically equivalent to eager.
- Consistent with the code: the Mamba mixer training forward has no RNG
  (`torch.rand`/`dt_init="random"` are init-only; the `_A_neg_exp_cache` write is
  inference-only; training recomputes `-exp(A_log)` deterministically), so the
  recomputed forward is bit-for-bit the original.

**Retraction of an earlier draft of this section:** a first pass reported `full`
diverging to +0.11 ("FAILS"). That was an **experimental artifact**: that `full`
trainer had ~16 `lr=0` steps (the lr=0 parity + grad_norm probes) run on it
before the trajectory, warming Adam, while the no-rc trajectories were fresh. Re-
running `full` fresh (no lr=0 prefix) removes the confound and it matches eager
(table above). Lesson: never reuse a trainer that has taken optimizer steps for a
trajectory-parity run.

**Conclusion:** the memory wins (Exp 5/6) AND the numerics are sound — `full`
(and Route A) recompute are safe to use for training on this model. (Localizing
to native-selective parity and a longer-horizon run are still cheap nice-to-haves
but no longer blockers.)

## Experiment 8 — full recompute at `max_seq_len=262144` (validate the line to 256k)

Same single node (TP8/PP1/EP8/ETP1/CP1, 8×B200), `granularity=full`, but
`max_seq_len` raised to 262144 so we can *measure* (not extrapolate) the high
end. 8-point sweep, `--reset-peak` per run, peak_alloc/GPU (GiB):

| seq | peak GiB |
| --- | --- |
| 8192   | 32.40 |
| 16384  | 35.00 |
| 32768  | 40.10 |
| 65536  | 49.51 |
| 98304  | 58.51 |
| 131072 | 68.29 |
| 196608 | 87.63 |
| 262144 | **107.84** |

**Fit: y = 0.302·(S/1024) + 30.03 GiB, R² = 0.99980.**

| line | slope MiB/tok | intercept GiB |
| --- | --- | --- |
| full @131k-max config (Exp 6) | 0.292 | 30.15 |
| full @256k-max config (this)  | 0.302 | 30.03 |

- The line is unchanged (~3% slope, ~0.1 GiB intercept) → at PP=1 memory tracks
  the *actual* sequence length, not `max_seq_len`; raising the cap is free until
  you actually feed longer sequences. Residual curvature is negligible (the
  1-layer recompute transient grows slowly with S); R²=0.9998.
- **256K LoRA SFT fits a single node at 107.84 GiB** (measured), with ~70 GiB to
  spare.
- Theoretical single-node max context: **~502K tok @178 GiB / ~519K @183 GiB** —
  and now anchored by measurement out to 256K rather than pure extrapolation.
