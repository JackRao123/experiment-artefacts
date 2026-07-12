# experiment_artefacts

Working logs, profiling sweeps, and reproducible harnesses for the
Nemotron-3-Super / Nemotron-3-Ultra LoRA SFT + RL work. These survive context
window compaction — read `goal.md` for the mission, then this index for what's
where.

## Top-level notes (cross-cutting)

- `goal.md` — the mission: golden config for Nemotron-3-Super 131k LoRA SFT, no OOM, PR into `jack-nemo3super`.
- `findings.md` — running log of non-obvious pitfalls (architecture, code bugs, devbox/ops, GPU health gate, validated golden config, recompute parity rules). The home for cross-cutting lessons; per-study details live in the study dirs.
- `question_log.md` — open questions.
- `NUMBER_ONE_PP_SAFE_COLLECTIVE.md` — the PP-safe collective export investigation.

## Studies

| dir | model | question | doc |
| --- | --- | --- | --- |
| `super/` | Nemotron-3-Super 120B | how does peak GPU memory scale with seq_len, and which recompute setting fits 131k/256k on one node? | `super/profiling_memory.md` (Exp 1–8) |
| `ultra/` | Nemotron-3-Ultra 550B | how to fit 256k LoRA SFT on 4×8 B200 (MoE expert-capacity, ETP, dropless vs capped)? | `ultra/profiling_memory.md` (Exp U1–U3 + benchmark) |
| `ultra/snapshots/` | Ultra | per-config memory snapshots (open in memory_viz) referenced by Exp U3 | — |
| `ultra/sampler_4vs8/` | Ultra | TRN-1488: NVFP4 sampler on 4×B200 (TP=4) vs 8×B200 (TP=8) — fit, KV pool, concurrency, throughput | `ultra/profiling_4vs8_gpus.md` |
| — (same harness) | Super | Super BF16 sampler quick check: TP=8 vs TP=4 vs TP=2 at 256k | `super/sampler_8vs4vs2.md` |
| `nvfp4_256k_logprob/` | Ultra | NVFP4-vLLM-vs-bf16-Megatron per-token logprob fidelity at long context | `nvfp4_256k_logprob/fidelity_probe.md` |
| `nvfp4_256k_logprob/hendrycks_klprobe/` | Ultra | prime-rl-style per-step mismatch-KL gate: BF16-native vs NVFP4-dequant trainer | `nvfp4_256k_logprob/hendrycks_klprobe/findings.md` |
| `tests/comparison_studies/nvfp4_compare/` | Ultra | can we serve the Ultra base in NVFP4 on the sampler (vs bf16) for ~6× KV headroom? | `tests/comparison_studies/nvfp4_compare/findings.md` |
| `pp_patch_forensics/` | generic | PP>1 adapter-export wedge forensics (tiny NemotronH repro) | `pp_patch_forensics/README.md` |

## Configs / examples

- `examples/ptt-nemotron3-super-sft/` — Super SFT driver scripts + memory report.
- `examples/trainer-configs/` — committed golden trainer JSONs.
- `nemotron_configs/` — base model configs.
- `tests/` — comparison studies (currently `nvfp4_compare/`).

## Reading order if you're new here

1. `goal.md`
2. `findings.md` (skim section headers)
3. `super/profiling_memory.md` (the recompute story — Exp 1→6→7→8 is the arc)
4. `ultra/profiling_memory.md` (the MoE-capacity/ETP story — U1→U2→U3)
5. `nvfp4_256k_logprob/fidelity_probe.md` then `hendrycks_klprobe/findings.md` (the RL-fidelity story)

## Naming convention

- `profiling_memory.md` — peak-GPU-memory-vs-X sweeps.
- `findings.md` — a study's conclusions / verdicts / pitfalls.
- `fidelity_probe.md` / `*_probe.md` — numerical-fidelity comparisons.
- `README.md` — how to run a study's harness.
