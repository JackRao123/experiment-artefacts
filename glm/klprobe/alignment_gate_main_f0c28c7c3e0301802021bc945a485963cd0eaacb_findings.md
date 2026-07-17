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
- preflight k3: not recomputable with the trainer formula (the retained `0.000312` value used the reversed offline formula; no per-token preflight capture was retained)

Per-step parity
| step | k3 | mean_abs | max_abs | ESS/N | clip | tokens | tails (\|r\|>1/2/5/10) | gate | step_s |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 00 | 0.004017 | 0.021892 | 2.7644 | 0.9904 | 0.0308 | 33529 | 44/2/0/0 | EXCLUDED | 179 |
| 01 | 0.008871 | 0.033235 | 4.4861 | 0.9819 | 0.0446 | 20334 | 95/16/0/0 | PASS | 127 |
| 02 | 0.006221 | 0.028030 | 6.9603 | 0.9821 | 0.0396 | 27056 | 65/7/1/0 | PASS | 111 |
| 03 | 0.012481 | 0.042644 | 9.5568 | 0.9664 | 0.0570 | 25263 | 144/27/3/0 | PASS | 128 |
| 04 | 0.016849 | 0.042873 | 3.5944 | 0.8736 | 0.0584 | 17084 | 131/21/0/0 | FAIL | 94 |

Summary
- gated steps below 0.015: 3 / 4
- maximum gated k3: 0.016849
- mean gated k3: 0.011105
- token-weighted mean absolute logprob delta: 0.032271
- sampler reloads verified: YES
- overall parity verdict: FAIL

Compressed comparison
| arm | gate | mean mismatch_kl | max | steps >= gate |
|---|---:|---:|---:|---:|
| Alpha | 0.015000 | 0.011105 | 0.016849 | 1/4 |

Tail analysis
- behavior/target logprob captures: runs/alignment_gate_main_f0c28c7c3e0301802021bc945a485963cd0eaacb/capture
- total scored tokens: 123266
- tails |r|>1/2/5/10: 479/73/4/0
- decoded outliers with |r| > 5: 4
- dangerous positive-r outliers r>1/2/5/10: 106/8/0/0

Long-context observation
- prefix + decode: 34,875 + 15,000 tokens
- scored decode tokens: 14,994
- k3: 0.023818
- mean_abs / max_abs: 0.079610 / 13.5686
- tails |r|>1/2/5/10: 137/29/5/1; all five `|r|>5` outliers are negative-r, while positive-r counts above 5 and 10 are zero
- six forced continuation nudges are excluded, leaving 14,994 scored decode tokens

Interpretation
- Three of four gate-counted steps pass; step 4 narrowly fails at 0.016849. The long probe is also above the gate at 0.023818. The extreme negative-r events are real, but they do not contribute exponentially under the trainer's K3 formula.
- Reload verification passed at every policy version, so the result is not explained by a stale sampler adapter.
