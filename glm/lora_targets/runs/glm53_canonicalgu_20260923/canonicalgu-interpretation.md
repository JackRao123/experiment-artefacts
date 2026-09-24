# Canonical gate/up effect: read-only analysis

Compared the 50-step shared-outer run `rrhnpm4p` with canonical-GU + shared-outer run `n2p3guk6`. Raw histories came from the corresponding `validation_50step` directories on `tj-wgpn4vw`. No new GPU training or serving was launched.

## Loss comparison

Both have 50 records. Step, input token count, sequence count, learning rate and loss-weight sum match at every step (tolerance 1e-7). Cookbook configs match except output paths and run names; both use no held-out split and no evaluation loop.

| Metric | Shared outer | Shared outer + canonical GU |
| --- | ---: | ---: |
| First-batch training NLL | 0.3253377974 | 0.3253377974 |
| Mean across all 50 training batches | 0.1759790930 | 0.1764402062 |
| Mean across final 10 training batches | 0.1501876637 | 0.1507565320 |
| Last-batch training NLL | 0.1559137255 | 0.1574802548 |

Canonical is lower on 7/50 steps, with mean signed difference +0.0004611132 and maximum absolute difference 0.0015736222. This is a small worsening in this run, not an established distribution-level effect. The final-ten mean is 0.379% higher.

The earlier no-routed-expert baseline had mean NLL 0.1860654762 and final-ten mean 0.1662819147. Adding shared-outer routed adapters had a much larger observed association with loss than separating gate/up A in the non-routed modules.

## Implementation and artifact checks

Read the current `GateUpCanonicalLoRA.transform`, Megatron `CanonicalLoRA.transform` and `LoRALinearSplitFC1UpGate.forward`. The wrapper separately evaluates gate/up adapters and concatenates their deltas in gate/up order. Routed experts follow the existing LoRA path. Adapter modules are inserted after base weights are frozen; their parameters are trainable by default.

Saved 50-step exports have identical key sets and shapes. Baseline dense/shared gate/up A tensors are identical for all 78 modules; canonical pairs differ for all 78. All 156 canonical gate/up B tensors are nonzero, and all inspected factors are finite. B starts at zero, so these are learned updates. All 75 routed shared gate/up A pairs remain identical as requested.

The configs describe 3 dense layers, 75 MoE layers, hidden size 6144, 256 routed experts per MoE layer, 8 selected per token, and 1 always-active shared expert. Rank is 32. Canonical GU adds one extra 32x6144 A matrix for each of 78 dense/shared modules: 15,335,424 parameters. Counting the exported factors while accounting for gate/up tying gives 3,997,929,472 versus 4,013,264,896 unique adapter parameters (+0.384%). Sparse activation means this parameter fraction is not a measure of functional importance.

## Offline spectral diagnostic

`analyze_canonicalgu.py` uses CPU safetensors reads and small QR/eigendecompositions to analyze the combined update `[B_gate A_gate; B_up A_up]` without loading the base model. Results are saved in `canonicalgu-factor-analysis.json`.

For fused adapters, that combined update has rank at most 32. For canonical adapters, it can have rank up to 64. The best rank-32 approximation loses a median 12.10% (mean 12.49%) of canonical squared Frobenius energy; the fused baseline loses only numerical roundoff. Canonical median rank required to retain 90% of energy is 36, versus 18 for baseline. Canonical therefore uses the additional matrix degrees of freedom; it has not collapsed back into the fused constraint.

This is weight-space energy, not task usefulness, activation energy, or a percentage loss improvement. Fixed-data ablations are needed to measure causal contribution to predictions.

## Interpretation and next discriminators

No evidence of an inactive/ignored canonical path. The experiment only unties non-routed gate/up inputs; it does not change routed-expert sharing, attention, rank per projection, or base weights. Shared A already permits different gate/up updates through different B matrices. Extra capacity need not improve short-run optimization or generalization.

Highest-value next checks:

1. Evaluate both checkpoints on identical held-out examples with per-example paired NLL differences and predictive KL; use deterministic execution or measure repeat noise. Training curves are different batches at each step, not validation curves.
2. On a fixed checkpoint and batch, disable dense/shared gate/up adapters, and then routed adapters, separately. Measure the increase in NLL. This tests which adapter groups the model uses; it does not alone establish that independent A is necessary.
3. Fit an activation-weighted shared-rank-32 approximation to canonical gate/up deltas, substitute it on a copy of the adapter and reevaluate. This directly tests whether the additional input subspace is useful.
4. Record module-level effective adapter activation RMS relative to base projection/output RMS; gate/up A and B gradients, optimizer updates and effective delta-W norms; and cross-branch gradients into shared A. Negative gate/up gradient cosine suggests interference that untying might relieve. Raw A/B norms alone are not gauge-invariant.
5. For causal A/B training comparisons, run multiple seeds on the same code revision. Initialize the independent A copies identically to the shared baseline and keep every unaffected adapter's initialization identical; then train tied versus untied. Historical runs span a target-selector refactor, although the Bridge pin and observed GLM export coverage agree.
