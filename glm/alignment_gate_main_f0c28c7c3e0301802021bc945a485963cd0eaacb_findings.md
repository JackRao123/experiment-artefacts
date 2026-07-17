# GLM-5.2 trainer/sampler KL probe

Scope caveat
- strict `main` at `f0c28c7c3e0301802021bc945a485963cd0eaacb` cannot reload the exported GLM adapter because main's vLLM validation rejects its trained `lm_head`
- the numeric run used the exact main baseline plus upstream compatibility commit `f4131e26` (cherry-picked locally as `8256f79482a5cc976c74f0f4db707dad07253f26`); the gate result below is therefore a diagnostic continuation, not a claim that unpatched main completes the gate

Run metadata
- baseline branch / SHA: main / f0c28c7c3e0301802021bc945a485963cd0eaacb
- numeric-continuation patch: f4131e26 (local cherry-pick 8256f79482a5cc976c74f0f4db707dad07253f26)
- trainer image: source checkout; torch 2.11.0+cu128
- sampler image: source checkout; vLLM 0.25.1, torch 2.11.0+cu130, Marlin
- trainer config: trainer-configs/glm52-b200-cp32-alignment-gate-klprobe.json
- model: /root/.cache/team_artifacts/huggingface/hub/models--zai-org--GLM-5.2-FP8/snapshots/70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1
- topology: 4x8 B200 trainer: TP1 PP1 CP32 EP32; 1x8 B200 sampler: TP8
- dataset: EleutherAI/hendrycks_math; levels=Level 4, Level 5; shuffle seed=16
- thinking: OFF
- rollout: 5 steps; 4 problems/step; group=8; max_tokens=2048; T=1.0; top_p=1.0; sample seed=None
- gate: k3 < 0.015; warmup steps=1

Preflight
- version-zero bootstrap: 0->1; lr=0.0; grad_norm=0.0e+00
- adapter policy version: required 1; observed [1]
- teacher-forced aligned tokens: 128
- preflight k3: 0.000312

Per-step parity
- step 00: k3=0.005176  mean_abs=0.021892  max_abs=2.7644  ESS/N=0.9904  clip=0.0308  tokens=33529  tails(|r|>1/2/5/10)=44/2/0/0  gate=EXCLUDED  step_s=179
- step 01: k3=0.025616  mean_abs=0.033235  max_abs=4.4861  ESS/N=0.9819  clip=0.0446  tokens=20334  tails(|r|>1/2/5/10)=95/16/0/0  gate=FAIL  step_s=127
- step 02: k3=0.047265  mean_abs=0.028030  max_abs=6.9603  ESS/N=0.9821  clip=0.0396  tokens=27056  tails(|r|>1/2/5/10)=65/7/1/0  gate=FAIL  step_s=111
- step 03: k3=0.631781  mean_abs=0.042644  max_abs=9.5568  ESS/N=0.9664  clip=0.0570  tokens=25263  tails(|r|>1/2/5/10)=144/27/3/0  gate=FAIL  step_s=128
- step 04: k3=0.030261  mean_abs=0.042873  max_abs=3.5944  ESS/N=0.8736  clip=0.0584  tokens=17084  tails(|r|>1/2/5/10)=131/21/0/0  gate=FAIL  step_s=94

Summary
- gated steps below 0.015: 0 / 4
- maximum gated k3: 0.631781
- mean gated k3: 0.183731
- token-weighted mean absolute logprob delta: 0.032271
- sampler reloads verified: YES
- overall parity verdict: FAIL

Compressed comparison
| arm | gate | mean mismatch_kl | max | steps >= gate |
|---|---:|---:|---:|---:|
| Alpha | 0.015000 | 0.183731 | 0.631781 | 4/4 |

Tail analysis
- behavior/target logprob captures: runs/alignment_gate_main_f0c28c7c3e0301802021bc945a485963cd0eaacb/capture
- total scored tokens: 123266
- tails |r|>1/2/5/10: 479/73/4/0
- decoded outliers with |r| > 5: 4
- dangerous positive-r outliers r>1/2/5/10: 106/8/0/0

Long-context observation
- prefix + decode: 34,875 + 15,000 tokens
- scored decode tokens: 14,994
- k3: 52.631041
- mean_abs / max_abs: 0.079610 / 13.5686
- tails |r|>1/2/5/10: 137/29/5/1; all five `|r|>5` outliers are negative-r, while positive-r counts above 5 and 10 are zero
- six forced continuation nudges are excluded, leaving 14,994 scored decode tokens

Interpretation
- All four gate-counted steps fail. Step 3 and the long probe are strongly tail-driven: their mean absolute deltas remain 0.042644 and 0.079610 while rare negative-r tokens lift k3 to 0.631781 and 52.631041.
- Reload verification passed at every policy version, so the result is not explained by a stale sampler adapter.
