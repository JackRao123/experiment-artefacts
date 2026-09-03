# GLM-5.2/5.3 B200/B300 Rightsizing

GLM-5.2 B300 memory measurements are documented in
`glm52/b300/pp1_cp8_ep8/RESULTS.md`.

## GLM-5.3 measurements

### Method

- Hardware and GPU count are specified per topology below.
- Code: trainers PR 1268 with PR 1257 applied on top.
- PP2 layout: 42 decoder layers on stage 0 and 36 decoder layers on stage 1.
- Model: GLM-5.3, LoRA rank 32, native-FP8 expert weights, full uniform one-layer recompute.
- Each result uses one untraced shape-specific warmup followed by one untraced control window.
- TPS/GPU is computed from forward-backward wall time only.
- Peak allocated and reserved memory are the maximum across all participating ranks.

### 262k, TP1/PP2/CP8/EP8/ETP1

This topology uses 16x B200 across two nodes and has DP1.

| Dispatcher | Microbatches | Tokens/step | Control FB | TPS/GPU | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|
| All-to-all | 4 | 1,048,576 | 82.99s | 789.66 | 156.154 GiB | 161.625 GiB |
| All-to-all | 8 | 2,097,152 | 150.11s | 873.19 | 156.161 GiB | 162.959 GiB |
| HybridEP | 4 | 1,048,576 | 76.54s | 856.25 | 142.034 GiB | 149.846 GiB |
| HybridEP | 8 | 2,097,152 | 139.05s | 942.65 | 142.034 GiB | 151.043 GiB |

#### HybridEP vs all-to-all

| Microbatches | TPS/GPU change | Peak allocated change | Peak reserved change |
|---|---:|---:|---:|
| 4 | +8.43% | -14.120 GiB | -11.779 GiB |
| 8 | +7.95% | -14.128 GiB | -11.916 GiB |

Raw results:

- `glm53/b200/pp2_cp8_ep8/262k/sequence/results/glm53-b200-pp2cp8ep8-pr1268-plus1257-262k-m4.json`
- `glm53/b200/pp2_cp8_ep8/262k/sequence/results/glm53-b200-pp2cp8ep8-pr1268-plus1257-262k-m8.json`
- `glm53/b200/pp2_cp8_ep8/262k/sequence/results/glm53-b200-pp2cp8ep8-hybridep-pr1268-plus1257-262k-m4.json`
- `glm53/b200/pp2_cp8_ep8/262k/sequence/results/glm53-b200-pp2cp8ep8-hybridep-pr1268-plus1257-262k-m8.json`

### 131k, TP1/PP1/CP8/EP8/ETP1

This topology uses one 8xB200 node and has DP1. With no pipeline parallelism,
the benchmark uses one microbatch.

| Dispatcher | GPUs | Microbatches | Tokens/step | Control FB | TPS/GPU | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|
| HybridEP | 8 | 1 | 131,072 | 15.04s | 1089.48 | 160.421 GiB | 164.529 GiB |

Raw result:

- `glm53/b200/pp1_cp8_ep8/131k/results/glm53-b200-8gpu-pp1cp8ep8-hybridep-pr1268-plus1257-131k-m1.json`

### 262k PP1 fit check, TP1/PP1/CP8/EP8/ETP1

This 8xB200 HybridEP topology does **not** fit. The first forward pass OOMed
in routed-expert `linear_fc2` while requesting an 8.02 GiB output buffer. At
failure, PyTorch had 162.92 GiB allocated and the GPU had 6.93 GiB free.

Failure log:

- `glm53/b200/pp1_cp8_ep8/262k/logs/oom-trainer-srun.log`

### 262k, 8xB300, TP1/PP1/CP8/EP8/ETP1

This curiosity run used merged `main`, HybridEP, and one microbatch. It is
recorded locally only and is not included in PR 1257.

| Controls | Tokens/step | Control FB mean | TPS/GPU | Peak allocated | Peak reserved |
|---:|---:|---:|---:|---:|---:|
| 3 | 262,144 | 28.69s | 1142.11 | 199.470 GiB | 206.326 GiB |

| Profile | Forward-backward | Optimizer | Artifact size |
|---|---:|---:|---:|
| Memory | 29.52s | 0.08s | 49,336,191 bytes |
| Runtime | 27.97s | 0.07s | 386,578,812 bytes |

Artifacts:

- `glm53/b300/pp1_cp8_ep8/262k/results/glm53-b300-8gpu-pp1cp8ep8-hybridep-main-262k.json`
- `glm53/b300/pp1_cp8_ep8/262k/memory/memory.rank0.pickle`
- `glm53/b300/pp1_cp8_ep8/262k/runtime/baseten-training-job-q9e90xw-multinode-0_3959.1788379599016053572.pt.trace.json`
