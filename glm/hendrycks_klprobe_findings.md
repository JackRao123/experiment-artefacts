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
- step 00: k3=0.024597  mean_abs=0.037995  max_abs=4.1770  ESS/N=0.9742  clip=0.0523  tokens=12029  tails(|r|>1/2/5/10)=80/8/0/0  gate=FAIL
- step 01: k3=0.014777  mean_abs=0.037464  max_abs=3.3121  ESS/N=0.9786  clip=0.0556  tokens=18782  tails(|r|>1/2/5/10)=69/6/0/0  gate=PASS
- step 02: k3=0.011368  mean_abs=0.033294  max_abs=4.0768  ESS/N=0.9773  clip=0.0494  tokens=24897  tails(|r|>1/2/5/10)=71/5/0/0  gate=PASS
- step 03: k3=0.013015  mean_abs=0.034667  max_abs=2.6137  ESS/N=0.9787  clip=0.0507  tokens=16616  tails(|r|>1/2/5/10)=65/6/0/0  gate=PASS
- step 04: k3=0.016451  mean_abs=0.030008  max_abs=3.2260  ESS/N=0.9766  clip=0.0418  tokens=14354  tails(|r|>1/2/5/10)=62/11/0/0  gate=FAIL

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
