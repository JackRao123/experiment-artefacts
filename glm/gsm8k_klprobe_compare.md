# GLM-5.2 trainer/sampler KL comparison

All values are end-to-end trainer/sampler measurements, not single-kernel attribution. B and Alpha use main baseline `f0c28c7c3e0301802021bc945a485963cd0eaacb` plus the minimal `lm_head` adapter-acceptance patch `f4131e26`; strict unpatched main fails adapter reload before numeric probing.

| arm | workload | thinking | gate-counted steps | mean k3 | max k3 | steps >= 0.015 | tail result |
|---|---|---:|---:|---:|---:|---:|---|
| Paras A, Marlin reference | GSM8K, seed / sample seed unknown | on | 12 | 0.0100 | 0.0117 | 0/12 | prior aggregate only |
| Paras A, best reference | GSM8K, seed / sample seed unknown | on | 12 | 0.0062 | 0.0075 | 0/12 | prior aggregate only |
| B | GSM8K, seed 999 / sample seed 1234 | on | 12 | 0.006193 | 0.008919 | 0/12 | 300/23/0/0 tokens at `|r|>1/2/5/10` |
| Alpha | MATH Level 4/5, seed 16 | off | 4 after 1 warmup | 0.183731 | 0.631781 | 4/4 | 479/73/4/0 tokens at `|r|>1/2/5/10` |

Long-context Alpha
- prefix/decode: 34,875 / 15,000 tokens; 14,994 decode tokens scored
- k3 `52.631041`, mean absolute delta `0.079610`, max absolute delta `13.5686`, ESS/N `0.9443`
- tails `137/29/5/1` at `|r|>1/2/5/10`; every `|r|>5` outlier is negative-r

Verdict
- B is better than the Paras A Marlin reference and close to the stated best reference while passing all 12 steps with no `|r|>5` event.
- Alpha fails every gate-counted step and degrades sharply with context length. The captures show rare, large negative-r events rather than broad bulk drift as the dominant k3 failure mode.
- Unpatched main remains blocked independently by `lm_head` adapter validation; the compatibility fix is required before either numeric arm can run.

Artifacts
- B findings: `gsm8k_klprobe_B_main_f0c28c7c3e0301802021bc945a485963cd0eaacb_findings.md`
- Alpha findings: `alignment_gate_main_f0c28c7c3e0301802021bc945a485963cd0eaacb_findings.md`
- Raw logs and captures: `runs/`
