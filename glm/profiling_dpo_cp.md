# GLM-5.2 DPO under THD context parallelism

## Result

DPO now runs under THD context parallelism and matches the established
PP16/EP2 reference topology.

- Debug GLM CP1 versus CP2: **PASS** for loss, returned logprobs, and gradient
  scaling.
- Production GLM-5.2 PP16/EP2/CP1/DP2 versus PP1/EP32/CP32/DP1: **PASS**,
  relative DPO loss delta `5.948e-3`.
- Production CP32 forward/backward plus optimizer step: **PASS**, finite
  gradient norm `0.673794`, with exactly one DPO pair used for normalization.
- Returned per-token policy logprobs reconstruct the scalar DPO loss to below
  `1e-6` absolute error in both production topologies.

Trainer branch: `jackrao/glm-131k-cp-rl-dpo`, based on
`jackrao/glm-131k-cp-rl` at `353f7c8d`.

## Implementation

The THD path now:

1. packs and zigzag-shards `ref_logprobs` with labels and weights;
2. partitions datums in atomic chosen/rejected pairs;
3. computes one local weighted logprob sum per packed document;
4. uses a differentiable CP SUM before applying DPO's nonlinear pair loss;
5. compensates for Megatron's CP loss scaling so gradients and pair counts are
   not multiplied by the CP world size; and
6. stitches policy logprobs for position-aligned client output.

The startup warmup also resets the training metrics tracker after discarding
its gradients. Without that reset, the first real optimizer step was normalized
by the warmup's 63 CE tokens plus the real batch. The production regression
showed `num_loss_tokens=64` before the fix and `num_loss_tokens=1` after it.

## Debug CP1 versus CP2

Harness: `scripts/cp_dpo_parity.py`.

Deterministic three-pair payload, lengths `173/173/96/96/41/41`, run with
`/forward_backward` followed by `/optim_step`.

| loss | CP1 | CP2 | relative delta | mean logprob abs delta | CP2/CP1 grad ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| cross entropy | 7.73883764 | 7.73904994 | 2.743e-5 | 4.142e-3 | 1.00014 |
| DPO | 0.632558723 | 0.632897854 | 5.358e-4 | 4.142e-3 | 1.00067 |

DPO's CP gradient ratio differs from the CE control by only `5.379e-4`
relative. Verdict: **PASS**.

Artifacts:

- `data/dpo_cp_parity/debug_cp1.json`
- `data/dpo_cp_parity/debug_cp2.json`

## Production PP16 versus CP32

Harness: `scripts/pp16_cp32_dpo_parity.py`.

Topologies:

- reference: PP16 / EP2 / CP1 / DP2, `max_seq_len=65536`;
- target: PP1 / EP32 / CP32 / DP1, `max_seq_len=131072`.

The final canonical payload is one chosen/rejected pair with 48,001 tokens per
side. CP32 pads each document to 48,064 tokens, so the THD row is 96,128 tokens.
The completion mask covers the second half, including an explicit active target
at the final position. Both layouts receive identical payload bytes and DPO
configuration (`beta=1e-4`).

| topology | DPO loss | wire reconstruction abs delta | forward time |
| --- | ---: | ---: | ---: |
| PP16/EP2 | 0.691430688 | 3.110e-7 | 104.91 s |
| CP32/EP32 | 0.687318265 | 9.816e-7 | 7.92 s |

Absolute delta: `4.112e-3`; relative delta: `5.948e-3`. Gate:
`rel_tol=1e-2, abs_tol=1e-5`. Verdict: **PASS**.

Artifacts:

- `data/dpo_cp_parity/pp16_48k.json`
- `data/dpo_cp_parity/cp32_48k.json`

### PP16 capacity boundary

A DPO pair must be in one BSHD microbatch on the reference topology. Two
65,521-token rows OOMed one PP stage by roughly 0.9 GiB; two 60,001-token rows
also OOMed on a different PP stage. Two 48,001-token rows are stable.

This is a PP16 B=2 activation-capacity limit, not a CP-DPO correctness failure.
The CP32 target can run the same pair comfortably. The production comparison
therefore validates the 131k-configured CP topology at a 96,128-token packed
row rather than claiming a full 131,072-token chosen/rejected pair fits the
PP16 reference.

## Production CP32 backward

The same 48,001-token pair was run through `/forward_backward` and
`/optim_step` on the production CP32 topology after fixing warmup metric reset:

- DPO loss: `0.686142206`;
- wire reconstruction absolute delta: `1.569e-7`;
- forward/backward: `47.33 s`;
- gradient norm: `0.673794466`;
- optimizer normalization count: exactly `1` pair;
- input tokens: `96,002`;
- total forward/backward plus optimizer time: `52.77 s`.

Artifact: `data/dpo_cp_parity/cp32_48k_backward.json`.

## Verification

- Focused unit suites: `162 passed`.
- Existing DPO GPU integration suite: `5 passed`.
- Debug GLM CP1/CP2 parity: **PASS**.
- Production PP16/CP32 scalar and wire parity: **PASS**.
- Production CP32 backward/optimizer smoke: **PASS**.
- Ruff checks and formatting: **PASS**.

The devbox used for validation was `wlerpv3` (nodes 0-3 of a 5-node B200
cluster). Trainers were stopped and all GPUs drained after the final run.
