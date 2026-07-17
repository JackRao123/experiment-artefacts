# PR #642: the RL-specific diff

This note assumes the SFT/CE THD context-parallel path already makes sense.
It explains only what PR [#642](https://github.com/basetenlabs/trainers/pull/642)
adds for RL.

## In one sentence

The existing CP path already moved `input_ids`, pre-shifted labels, positions,
and masks through THD packing and zigzag sharding. This PR makes the RL-only
per-token inputs—old logprobs, advantages, and temperatures—move through
exactly the same path, then lets the per-token RL losses run on the shards.

## What was broken before this PR?

An RL loss does not need only the model's new logprob. At each target-token
position, it also needs:

```text
new_logprob  = log π_current(target | context)    # computed by the trainer
old_logprob  = log π_rollout(target | context)    # supplied by the sampler
advantage    = reward-derived signal for that token
temperature  = rollout temperature, if non-default
```

Before #642, the THD CP packer dropped the last three values. In
`_pack_thd_cp_microbatches`, it constructed a CP-local `Batch` like this:

```python
sampling_logprobs=None,
advantages=None,
temperatures=None,
```

So even if the client sent valid RL inputs, the RL forward step received
`None` and raised its existing "`logprobs` and `advantages` are required"
error.

Separately, `execute_forward_backward` rejected every non-CE loss whenever
`context_parallel_size > 1`. Thus RL was blocked twice:

1. the public CP-loss guard rejected it; and
2. the underlying CP batch did not contain the data the RL loss needs.

## The key alignment rule

For one input sequence:

```text
input tokens:  [ t0, t1, t2, t3 ]
target tokens: [ t1, t2, t3, -100 ]
old logprobs:  [ lp0, lp1, lp2, 0 ]
advantages:    [  A0,  A1,  A2, 0 ]
temperatures:  [  T0,  T1,  T2, 1 ]
```

Every column describes the same prediction:

```text
hidden state for t0  → predict t1  → use lp0, A0, T0
hidden state for t1  → predict t2  → use lp1, A1, T1
hidden state for t2  → predict t3  → use lp2, A2, T2
hidden state for t3  → no target   → inactive
```

THD CP zigzag-shards positions. Therefore the three RL tensors must receive
the **same permutation** as `input_ids` and labels. If an advantage were
sharded differently from its corresponding old logprob, the code would still
have valid shapes but would optimize the wrong token/reward pair.

## Diff 1: `packing.py` now packs RL inputs

`ThdCpMicrobatch` gains three optional global packed rows:

```python
sampling_logprobs: Optional[torch.Tensor]
advantages: Optional[torch.Tensor]
temperatures: Optional[torch.Tensor]
```

They are added beside the pre-existing global `input_ids`, labels, positions,
weights, and padding mask. This is not new RL math; it is only preserving the
inputs until the RL loss can consume them.

Inside `pack_thd_cp_microbatch`, the PR now:

1. checks whether every datum in the packed microbatch has each field;
2. appends each datum's per-token values to the global THD row;
3. appends safe values for that datum's THD tail padding; and
4. turns the resulting Python lists into `(1, S_global)` float tensors.

The padding values are intentionally inert:

```text
old logprob: 0.0
advantage:   0.0
temperature: 1.0
```

`advantage == 0` ensures an RL pad cannot contribute loss. `temperature == 1`
does not alter logits. The label is independently padded with `-100`.

### Why `all(...)` matters

The fields are populated only if **every** datum in the THD microbatch has
them. If one datum omits `advantages`, the entire packed `advantages` field is
`None`; it does not build a partly valid tensor.

That is compatible with the existing RL input contract:

- missing old logprobs or advantages causes the RL forward step to fail;
- temperature is optional, so `None` means use temperature one everywhere.

One caveat: do not mix omitted temperatures with non-unit temperatures in one
packed microbatch. Because the packer uses `all(...)`, the entire temperature
row becomes `None`, and the supplied non-unit temperatures are ignored. This
is inherited BSHD behavior, but it is still an easy mistake to make.

## Diff 2: CP now shards those packed RL rows

The PR adds this helper in `_pack_thd_cp_microbatches`:

```python
def _cp_shard_opt(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if t is None else _cp_shard(t)
```

`_cp_shard` already uses Transformer Engine's canonical THD zigzag indices.
The PR changes the local `Batch` construction from discarding the RL values to
using that helper:

```python
sampling_logprobs=_cp_shard_opt(thd.sampling_logprobs),
advantages=_cp_shard_opt(thd.advantages),
temperatures=_cp_shard_opt(thd.temperatures),
```

This is the core data-plumbing change:

```text
before: client RL values → THD packer → discarded before model/loss
after:  client RL values → THD packer → same CP shard as labels → RL loss
```

The same helper is also applied to CE `weights`. That part is merely a small
refactor; its effective behavior does not change.

## Diff 3: the RL forward step now enters the THD model path

The CE forward step already detected THD CP and passed:

```python
packed_seq_params=b.packed_seq_params
padding_mask=b.padding_mask
```

The RL forward step did not. This PR adds the equivalent branch.

Without it, even if the RL input tensors survived sharding, the model forward
would not know it was receiving a packed THD local row. `PackedSeqParams`
contains the global packed-sequence boundaries needed by the THD attention
path; `padding_mask` identifies tail padding.

The PR also passes two facts into the later loss closure:

```python
labels_pre_shifted=thd_cp
thd_cp_seq_params=b.packed_seq_params if thd_cp else None
```

They mean:

- `labels_pre_shifted`: use the THD target alignment, rather than the normal
  dense-row alignment.
- `thd_cp_seq_params`: reconstruct client-visible logprobs after the loss.

## Diff 4: the RL loss stops applying a second shift under THD CP

Before #642, `_rl_loss_from_hidden` always used the dense BSHD convention:

```python
targets = labels[:, :-1]
old_logprobs = sampling_logprobs[:, :-1]
advs = advantages[:, :-1]
```

That is right for the ordinary path. It is wrong for a THD CP local shard,
because labels were already pre-shifted before the zigzag partition.

The PR changes this to:

```text
Normal BSHD:
  use [:, :-1] as before.

THD CP:
  use the complete local labels / old-logprobs / advantages rows.
```

It also passes `targets_pre_shifted=True` to the chunked LM head in the THD
case. That tells the head to score every local position rather than dropping
its final local column.

This does **not** make end-of-document or padding positions active. The mask
is still:

```python
loss_mask = (advs != 0) & (targets != -100)
```

The important distinction is:

```text
"score all local slots"     != "every local slot contributes loss"
```

It means “do not accidentally drop a valid token merely because it happens to
be the last element of this rank's zigzag shard.” The real final token and
padding remain inactive through `-100` and/or zero advantage.

## Diff 5: RL reuses the existing CP logprob reconstruction

The CP SFT branch already did this after its loss:

1. all-gather each rank's local logprob row over the CP group;
2. invert the THD zigzag permutation;
3. remove each document's tail padding; and
4. return logprobs in the client's original datum order.

That machinery was already present in `_stitch_thd_cp_logprobs`,
`unshard_thd_cp_rows`, and `thd_logprobs_to_loss_fn_outputs`.

The PR does **not** invent a second reconstruction mechanism. It changes
`_rl_loss_from_hidden` so that its `out_logprobs` takes the same existing
route:

```python
if thd_cp_seq_params is not None:
    metrics["logprobs"] = _stitch_thd_cp_logprobs(
        out_logprobs, thd_cp_seq_params
    )
```

This part is necessary because the client expects a full per-datum vector,
not a partial vector in CP-rank-local zigzag order.

The gather is for the reported logprobs, not for the gradient computation.
Each CP rank still computes the RL loss for its own tokens.

## Diff 6: expand the CP loss allowlist

The old guard said, in effect:

```text
context_parallel_size > 1 → only cross_entropy
```

The PR defines:

```python
_THD_CP_LOSS_FNS = frozenset(_SUPPORTED_LOSS_FNS) - {"dpo"}
```

So CP now permits:

```text
cross_entropy
dppo
importance_sampling
ppo
cispo
dro
```

`dpo` remains rejected.

## Why these RL losses are allowed but DPO is not

The supported RL objectives are token-separable. Their total loss has this
form:

```text
total_loss = sum over active tokens of
    f(new_logprob_i, old_logprob_i, advantage_i, configuration)
```

Each CP rank can calculate `f(...)` for the tokens it owns. Summing all ranks'
partial losses is identical to computing one sum over the original sequence.
That is why the existing CP reduction/normalization logic can support these
objectives.

DPO is different. It first sums logprobs over a complete chosen sequence and a
complete rejected sequence, then applies a nonlinear preference loss to those
two sequence-level sums:

```text
chosen_score   = sum of chosen-token logprobs
rejected_score = sum of rejected-token logprobs
loss = nonlinear_preference_function(chosen_score, rejected_score)
```

Under CP, each rank has only a partial score for each sequence. The code would
need to reduce the per-document chosen and rejected scores across CP *before*
the nonlinear DPO function. That CP-aware, gradient-carrying per-sequence
reduction does not exist in the chunked LM-head path today.

Computing DPO independently on rank-local partial sums would be wrong:

```text
nonlinear(sum of rank pieces) != sum of nonlinear(rank piece)
```

The PR therefore rejects DPO rather than silently optimizing a different
objective.

## Tests added by this PR

The changed test file is `test_cp_thd_slicing.py`. The new tests verify only
the new packer behavior:

- RL logprobs, advantages, and temperatures are packed in document order.
- Per-document tail padding is `0.0 / 0.0 / 1.0`.
- If a datum omits an RL input, the corresponding packed field is `None`.
- A supplied RL vector with the wrong length raises before GPU execution.

That is useful, but it is not an end-to-end RL CP test.

## What is still unproven

The PR explicitly says GPU execution was not rerun after the rebase. Its test
delta does not verify:

- a CP2/CP32 forward-backward with PPO, DPPO, CISPO, DRO, or IS;
- numerical parity between an RL run at CP1 and the same run at CP>1;
- that the new RL tensors receive the same real Transformer Engine zigzag
  indices as labels in the controller path; or
- non-unit-temperature parity between sampler old logprobs and trainer new
  logprobs.

The highest-value follow-up is a small GPU parity test: use a padded
multi-document batch with nonzero advantages and non-unit temperatures, run
PPO at CP1 and CP2 from identical weights, and compare loss, active-token
count, gradients, and reconstructed client logprobs.

## Bottom line

PR #642 is mostly not new RL algorithm code. The existing RL formulas are
unchanged. It is a correctness-sensitive data-layout change:

```text
Make the rollout-derived per-token values follow their target token through
THD packing and CP zigzag sharding, then use the pre-shifted THD alignment
when evaluating the existing token-local RL loss.
```

That supports the five token-local RL objectives. It correctly refuses DPO
until the trainer implements a distinct CP-aware per-sequence reduction.
