# GLM native-FP8 recompute and output-head results

## Shipped draft stack

- Trainers: https://github.com/basetenlabs/trainers/pull/1251
- Bridge: https://github.com/basetenlabs/Megatron-Bridge/pull/63
- MCore: https://github.com/basetenlabs/Megatron-LM/pull/72

The pushed stack contains two independent improvements:

1. Frozen FP8/NVFP4 expert BF16 weights are rematerialized at the grouped-linear boundary for dgrad. Their lifetime no longer requires full/uniform/1 layer recompute.
2. GLM's numerically-required FP32 logits use BF16 tensor-core operands with FP32 GEMM output. Frozen-linear dgrad uses BF16x3 decomposition of the FP32 loss gradient, accumulates in FP32, then casts once to BF16.

Combined-1F1B, recompute-dial, and activation-offload experiments remain uncommitted and are not part of these PRs.

## Final performance

Configuration: full GLM-5.3, one 8xB300 node, seq 131072, D1, TP1/PP1/CP8/EP8/ETP1, native-FP8 expert storage, HybridEP, LoRA r32, full/uniform/1 recompute.

| Metric | FP32 SIMT head | Mixed-output tensor-core head | Delta |
|---|---:|---:|---:|
| Stabilized mean tok/s/GPU | 1,183 | 1,224 | +3.5% |
| Stabilized median tok/s/GPU | 1,189 | 1,231 | +3.5% |
| Profile-run controls | ~1,195 | 1,271 | +6.4% |
| Profiled step wall | 14.99 s | 13.60 s | -9.3% |
| MFU3x (steady) | ~10.8% | ~11.2% | +0.4 pp |
| Transient peak/rank | 48.2 GiB | 38.5 GiB | -9.7 GiB |

The profile-run comparison includes normal run-to-run HybridEP skew; the steady mean/median +3.5% is the conservative headline.

## Runtime trace

- Final: `final_mixed_output_runtime.pt.trace.json` (391 MB)
- Baseline: `../glm53_main_tip_runtime_memory_20260831/glm53_main_tip_131k_runtime.pt.trace.json`

The two FP32 SIMT SGEMMs previously consumed 0.926 s (6.2% of wall). They disappear completely in the final trace. Final GPU wall/busy is 13.60/12.34 s, versus 14.99/13.70 s baseline.

The remaining top buckets are HybridEP (2.82 s sync+movement+permute), DSA attention (2.81 s), ordinary GEMMs (2.68 s), and elementwise/copy (2.28 s).

## Numerical validation

- MCore 8-rank B300 target set: 32/32 tests pass, covering frozen FP8/NVFP4 rematerialization, TE reentrant and non-reentrant checkpoints, enabled quantized-autocast fallback, FP32 TP/SP reduction, and mixed-output frozen dgrad.
- BF16x3 frozen-linear dgrad is bit-exact to full-FP32 dgrad after the contract's final BF16 cast.
- Bridge Kimi MXFP4 import: 17/17 tests pass, including E8M0 byte zero, signed-zero payload groups, and nonstandard NVFP4 E4M3 rejection.
- The MCore adapter supports both the deployed TE grouped-linear call contract and pinned TE 2.17.1's explicit `m_splits` contract.
- Debug 0d1m matched A/B across identical weights/data:
  - loss relative drift: 0.8e-6 to 3.7e-6
  - grad-norm relative drift: 1.2e-4 to 2.6e-4
  - all losses/grad norms finite and nonzero
- Plain BF16 logits were rejected: loss drift reached ~5e-4 relative and grad norms diverged materially.

## Recompute and overlap findings

- Native FP8 itself does not require full recompute. The requirement came from TE autograd retaining temporary BF16 expert weights. Local rematerialization fixes the ownership boundary.
- Selective whole-MoE recompute is +2.7% on the one-layer debug proxy but OOMs on the full model because non-MoE activations are retained.
- Block72 is full-model memory-safe but perf-neutral. Its memory snapshot shows six direct layers add 51.9 GiB:
  - HybridEP dispatch: 9.5 GiB
  - expert activation state: ~13.6 GiB
  - DSA state: ~15.8 GiB
  - frozen grouped-linear outputs and other state: remainder
- Combined-1F1B was made functionally correct with native FP8, D2, chunked CE, and LoRA. K76 is perf-neutral. Exposing enough layers for a useful gain requires offload/recompute whose cost outweighed overlap; K67 achieved only ~1,017 tok/s/GPU. This path was stopped.

## Artifacts

- `final_mixed_output_steady15_result.json`
- `final_mixed_output_profile_result.json`
- `final_mixed_output_runtime.pt.trace.json`
- `final_mixed_output_memory/memory.rank0-7.pickle`
- `full_fp32head_steady10_result.json`
- `debug_mixed_output_head_bf16x3_result.json`
- `block72_memory/`
