# GLM-5.2 trainer/sampler KL probe

Scope caveat
- strict `main` at `f0c28c7c3e0301802021bc945a485963cd0eaacb` bootstrapped policy `0->1`, exported the adapter, and then failed sampler reload because the checkpoint tunes `lm_head` but main's vLLM target-module validation rejected it
- the numeric run used the exact main baseline plus upstream compatibility commit `f4131e26` (cherry-picked locally as `8256f79482a5cc976c74f0f4db707dad07253f26`); this patch only adds `lm_head` to the accepted trainer-adapter modules
- strict-main evidence is preserved in `runs/gsm8k_klprobe_B_main_f0c28c7c3e0301802021bc945a485963cd0eaacb/logs/preflight.strict-main.jsonl` and `trainer.strict-main.log`

Run metadata
- baseline branch / SHA: main / f0c28c7c3e0301802021bc945a485963cd0eaacb
- numeric-continuation patch: f4131e26 (local cherry-pick 8256f79482a5cc976c74f0f4db707dad07253f26)
- trainer image: source checkout; torch 2.11.0+cu128
- sampler image: source checkout; vLLM 0.25.1, torch 2.11.0+cu130, Marlin
- trainer config: trainer-configs/glm52-b200-cp32-gsm8k-klprobe.json
- model: /root/.cache/team_artifacts/huggingface/hub/models--zai-org--GLM-5.2-FP8/snapshots/70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1
- topology: 4x8 B200 trainer: TP1 PP1 CP32 EP32; 1x8 B200 sampler: TP8
- dataset: openai/gsm8k; shuffle seed=999
- thinking: ON
- rollout: 12 steps; 4 problems/step; group=8; max_tokens=2048; T=1.0; top_p=1.0; sample seed=1234
- gate: k3 < 0.015; warmup steps=0

Preflight
- version-zero bootstrap: 0->1; lr=0.0; grad_norm=0.0e+00
- adapter policy version: required 1; observed [1]
- teacher-forced aligned tokens: 128
- preflight k3: not recomputable with the trainer formula (the retained `0.007292` value used the reversed offline formula; no per-token preflight capture was retained)

Per-step parity
| step | k3 | mean_abs | max_abs | ESS/N | clip | tokens | tails (\|r\|>1/2/5/10) | gate | step_s |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 00 | 0.002918 | 0.019769 | 2.5162 | 0.9942 | 0.0253 | 14978 | 8/2/0/0 | PASS | 77 |
| 01 | 0.004509 | 0.031282 | 2.2470 | 0.9899 | 0.0410 | 27727 | 27/3/0/0 | PASS | 149 |
| 02 | 0.003831 | 0.025283 | 2.5596 | 0.9921 | 0.0347 | 17618 | 19/1/0/0 | PASS | 87 |
| 03 | 0.005049 | 0.027451 | 2.7499 | 0.9890 | 0.0359 | 24737 | 45/3/0/0 | PASS | 120 |
| 04 | 0.004756 | 0.035515 | 4.1906 | 0.9907 | 0.0464 | 24608 | 24/3/0/0 | PASS | 98 |
| 05 | 0.003708 | 0.029258 | 2.4322 | 0.9913 | 0.0341 | 28126 | 20/2/0/0 | PASS | 122 |
| 06 | 0.004013 | 0.022399 | 2.8484 | 0.9907 | 0.0290 | 17215 | 25/1/0/0 | PASS | 84 |
| 07 | 0.005244 | 0.032067 | 4.0016 | 0.9887 | 0.0422 | 22739 | 28/2/0/0 | PASS | 101 |
| 08 | 0.005223 | 0.037771 | 2.1512 | 0.9874 | 0.0486 | 34566 | 35/2/0/0 | PASS | 140 |
| 09 | 0.004267 | 0.027708 | 2.3520 | 0.9904 | 0.0359 | 22170 | 18/2/0/0 | PASS | 119 |
| 10 | 0.004782 | 0.024337 | 2.9994 | 0.9875 | 0.0324 | 18125 | 24/1/0/0 | PASS | 87 |
| 11 | 0.005193 | 0.027909 | 3.4422 | 0.9877 | 0.0378 | 17068 | 27/1/0/0 | PASS | 83 |

Summary
- gated steps below 0.015: 12 / 12
- maximum gated k3: 0.005244
- mean gated k3: 0.004458
- token-weighted mean absolute logprob delta: 0.029431
- sampler reloads verified: YES
- overall parity verdict: PASS

Compressed comparison
| arm | gate | mean mismatch_kl | max | steps >= gate |
|---|---:|---:|---:|---:|
| B | 0.015000 | 0.004458 | 0.005244 | 0/12 |

Tail analysis
- behavior/target logprob captures: runs/gsm8k_klprobe_B_main_f0c28c7c3e0301802021bc945a485963cd0eaacb/capture
- total scored tokens: 269677
- tails |r|>1/2/5/10: 300/23/0/0
- decoded outliers with |r| > 5: 0
- dangerous positive-r outliers r>1/2/5/10: 71/1/0/0

Interpretation
- The numeric continuation passes decisively: all 12 steps are below the literal gate, the worst k3 is 0.005244, and there are no `|r|>5` tokens across 269,677 scored tokens.
- This does not make unpatched main runnable: strict main fails earlier at adapter reload, before a numeric parity verdict can be produced.
