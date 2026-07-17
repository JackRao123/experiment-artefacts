# GLM-5.2 CP32 trainer/sampler KL probe

Run metadata
- branch / SHA: jackrao/vllm-0.25-stack / c23d3fa522c55e33f8f1cb029f59eff760f940fc
- trainer image: source server/.venv
- sampler image: source sampler/.venv
- trainer config: /root/.cache/user_artifacts/glm52_pp16_klprobe/c23d3fa5/trainer-config.json
- model: zai-org/GLM-5.2-FP8
- topology: trainer=4x8 B200 (TP1 PP16 CP1 EP2); sampler=1x8 B200 (TP8)
- dataset: PrimeIntellect/Hendrycks-Math/default/train; shuffle seed=999
- rollout: 5 steps; 4 problems/step; group=8; max_tokens=2048; T=1.0; top_p=1.0; sample seed=1234
- gate: k3 < 0.015

Preflight
- version-zero bootstrap: 0->1; lr=0.0; grad_norm=0.0e+00
- adapter policy version: required 1; observed [1]
- teacher-forced aligned tokens: 128
- preflight k3: 0.026317

Per-step parity
| step | k3 | mean_abs | max_abs | ESS/N | clip | tokens | tails (\|r\|>1/2/5/10) | gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 00 | 0.016641 | 0.036312 | 3.2660 | 0.9471 | 0.0513 | 12723 | 71/8/0/0 | FAIL |
| 01 | 0.029283 | 0.036771 | 5.6518 | 0.9806 | 0.0540 | 21334 | 78/6/1/0 | FAIL |
| 02 | 0.013752 | 0.033935 | 4.3590 | 0.9850 | 0.0495 | 26763 | 53/5/0/0 | PASS |
| 03 | 0.019260 | 0.033300 | 4.2434 | 0.9767 | 0.0469 | 15285 | 67/7/0/0 | FAIL |
| 04 | 0.030966 | 0.033838 | 4.2127 | 0.9775 | 0.0455 | 14470 | 77/16/0/0 | FAIL |

Summary
- steps below 0.015: 1 / 5
- maximum k3: 0.030966
- clean-step mean k3: 0.013752
- token-weighted mean absolute logprob delta: 0.034814
- sampler reloads verified: YES
- overall parity verdict: FAIL

Tail analysis
- behavior/target logprob captures: /root/.cache/user_artifacts/glm52_pp16_klprobe/c23d3fa5/capture
- total scored tokens: 90575
- tails |r|>1/2/5/10: 346/42/1/0
- decoded outliers with |r| > 5: 1
- dangerous positive-r outliers r>1/2/5/10: 79/5/0/0
- interpretation: The literal gate failed on at least one step. Compare each failure with mean_abs, ESS, clip fraction, and signed tail counts to distinguish bulk mismatch from isolated tail domination.
