# GLM-5.3 LoRA training throughput and stability

Full GLM-5.3 on 8 B300 GPUs, LoRA rank/alpha 32, TP1/PP1/EP8/CP8/ETP1,
native FP8 expert storage, full uniform recomputation. Each `profile_driver.py`
window uses one 131,072-token synthetic datum. One warmup is excluded from
every measurement; the control count is shown below. TPS/GPU divides total
control tokens by total forward/backward wall time and eight GPUs. Optimizer
time and profiling overhead are excluded. These are **training** TPS, not
sampler output TPS.

| LoRA configuration | Controls | Mean TPS/GPU | Median-window TPS/GPU | FB-time CV | First control vs later median | Mean TPS/GPU after first |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Fused baseline, routed experts excluded](glm53_expert_lora_20260921/glm53-full-baseline-131k.json) | 5 | 1,339.6 | 1,352.9 | 6.4% | 14.5% slower | 1,378.4 |
| [Shared-adapter routed experts](glm53_expert_lora_20260921/glm53-full-shared-adapter-131k.json) | 5 | 1,131.0 | 1,133.8 | 4.8% | 9.4% slower | 1,150.6 |
| [Shared-outer routed experts](glm53_expert_lora_20260921/glm53-full-shared-outer-131k.json) | 5 | 1,057.5 | 1,140.9 | 16.2% | 43.4% slower | 1,149.3 |
| [Canonical gate/up, routed experts excluded](glm53_canonicalgu_20260923/glm53-full-canonicalgu-131k.json) | 3 | 1,265.1 | 1,305.5 | 5.1% | 11.3% slower | 1,312.7 |
| [Canonical gate/up + shared-outer](glm53_canonicalgu_20260923/glm53-full-canonicalgu-sharedouter-131k.json) | 3 | 923.4 | 1,038.9 | 18.3% | 44.4% slower | 1,060.0 |
| [No `q_b_proj` + shared-outer + nonorm, first pass](glm53_noqbproj_20260923/glm53-full-sharedouter-noqbproj-nonorm-131k.json) | 3 | 847.9 | 987.1 | 25.0% | 64.1% slower | 1,028.9 |
| **[No `q_b_proj` + shared-outer + nonorm, warm pass](glm53_noqbproj_20260923/glm53-full-sharedouter-noqbproj-nonorm-131k-5controls.json)** | **5** | **1,087.1** | **1,079.9** | **4.6%** | **2.1% faster** | **1,083.5** |

`FB-time CV` is the population standard deviation divided by the mean of
the observed control forward/backward times. The last column recomputes
aggregate TPS after excluding control 0; it is diagnostic, **not** the
original driver's headline. The 3-control runs have only two later windows,
so their apparent later stability is weak evidence.

## Were they stable?

- **Fused baseline and shared-adapter:** reasonably consistent. Overall
  FB-time CV was 6.4% and 4.8%; after the slightly slow first control,
  the remaining four controls had CV 3.6% and 3.9%.
- **Shared-outer:** not stable across all five controls. Control 0 took
  20.443s; the other four took 14.853, 13.665, 14.361, and 14.146s
  (remaining-window CV 3.0%). Its 1,057.5 headline mean is pulled down
  by that first control; the later-four estimate is 1,149.3 TPS/GPU.
- **Canonical gate/up alone:** modest first-control slowdown (13.889s vs
  12.413 and 12.550s). Three controls are too few to establish a
  run-to-run steady-state confidence interval.
- **Canonical gate/up + shared-outer:** not stable as a three-control set:
  22.315s then 15.771 and 15.142s. The 923.4 headline is pulled down
  by the first; 1,060.0 TPS/GPU from the two later controls is indicative,
  not a stable estimate.
- **No `q_b_proj` initial pass:** likewise unstable (26.125, 16.598,
  15.249s). Do **not** use its 847.9 TPS/GPU mean as the steady estimate.
- **No `q_b_proj` warm five-control pass:** reasonably stable: 14.866,
  15.209, 13.963, 16.144, 15.172s (4.6% CV), with no unusually slow
  first control. **1,087.1 TPS/GPU** is the preferred observed estimate.

The five historical configurations used default token normalization and
were recorded at earlier trainer revisions; the new no-`q_b_proj` profile
sent `skip_token_normalization=true` and checked server acknowledgment on
every window. These measurements share model, topology, sequence length,
rank, and driver, but **do not isolate** the speed effect of removing
`q_b_proj` or skipping normalization. Small differences between the
shared-outer and no-`q_b_proj` headlines should not be treated as a
causal improvement or regression.
