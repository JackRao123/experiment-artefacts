# GLM-5.2 CP32 trainer/sampler KL probe

Run metadata
- branch / SHA: jackrao/vllm-0.25-stack / c23d3fa522c55e33f8f1cb029f59eff760f940fc
- trainer image: source server/.venv
- sampler image: source sampler/.venv
- trainer config: /root/.cache/user_artifacts/glm52_cp32_klprobe/c23d3fa5/trainer-config.json
- model: zai-org/GLM-5.2-FP8
- topology: trainer=4x8 B200 (TP1 PP1 CP32 EP32); sampler=1x8 B200 (TP8)
- dataset: PrimeIntellect/Hendrycks-Math/default/train; shuffle seed=999
- rollout: 5 steps; 4 problems/step; group=8; max_tokens=2048; T=1.0; top_p=1.0; sample seed=1234
- gate: k3 < 0.015

Preflight
- version-zero bootstrap: 0->1; lr=0.0; grad_norm=0.0e+00
- adapter policy version: required 1; observed [1]
- teacher-forced aligned tokens: 128
- preflight k3: 0.021106

Per-step parity
| step | k3 | mean_abs | max_abs | ESS/N | clip | tokens | tails (\|r\|>1/2/5/10) | gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 00 | 0.024597 | 0.037995 | 4.1770 | 0.9742 | 0.0523 | 12029 | 80/8/0/0 | FAIL |
| 01 | 0.014777 | 0.037464 | 3.3121 | 0.9786 | 0.0556 | 18782 | 69/6/0/0 | PASS |
| 02 | 0.011368 | 0.033294 | 4.0768 | 0.9773 | 0.0494 | 24897 | 71/5/0/0 | PASS |
| 03 | 0.013015 | 0.034667 | 2.6137 | 0.9787 | 0.0507 | 16616 | 65/6/0/0 | PASS |
| 04 | 0.016451 | 0.030008 | 3.2260 | 0.9766 | 0.0418 | 14354 | 62/11/0/0 | FAIL |

Summary
- steps below 0.015: 3 / 5
- maximum k3: 0.024597
- clean-step mean k3: 0.013054
- token-weighted mean absolute logprob delta: 0.034569
- sampler reloads verified: YES
- overall parity verdict: FAIL

Tail analysis
- behavior/target logprob captures: /root/.cache/user_artifacts/glm52_cp32_klprobe/c23d3fa5/capture
- total scored tokens: 86678
- tails |r|>1/2/5/10: 347/36/0/0
- decoded outliers with |r| > 5: 0
- dangerous positive-r outliers r>1/2/5/10: 78/3/0/0
- interpretation: The literal gate failed on at least one step. Compare each failure with mean_abs, ESS, clip fraction, and signed tail counts to distinguish bulk mismatch from isolated tail domination.
