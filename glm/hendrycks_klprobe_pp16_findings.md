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
- step 00: k3=0.016641  mean_abs=0.036312  max_abs=3.2660  ESS/N=0.9471  clip=0.0513  tokens=12723  tails(|r|>1/2/5/10)=71/8/0/0  gate=FAIL
- step 01: k3=0.029283  mean_abs=0.036771  max_abs=5.6518  ESS/N=0.9806  clip=0.0540  tokens=21334  tails(|r|>1/2/5/10)=78/6/1/0  gate=FAIL
- step 02: k3=0.013752  mean_abs=0.033935  max_abs=4.3590  ESS/N=0.9850  clip=0.0495  tokens=26763  tails(|r|>1/2/5/10)=53/5/0/0  gate=PASS
- step 03: k3=0.019260  mean_abs=0.033300  max_abs=4.2434  ESS/N=0.9767  clip=0.0469  tokens=15285  tails(|r|>1/2/5/10)=67/7/0/0  gate=FAIL
- step 04: k3=0.030966  mean_abs=0.033838  max_abs=4.2127  ESS/N=0.9775  clip=0.0455  tokens=14470  tails(|r|>1/2/5/10)=77/16/0/0  gate=FAIL

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
