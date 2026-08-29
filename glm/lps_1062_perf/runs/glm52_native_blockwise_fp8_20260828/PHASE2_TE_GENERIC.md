# Phase 2: TE Generic BF16 Fallback

## Final Design

Only routed expert FC1 and FC2 weights use native checkpoint storage:

```text
persistent: E4M3 rowwise payload + FP32 128x128 scale grid
forward:    TE dequantizes to BF16 and runs BF16 grouped GEMM
backward:   TE reuses the saved BF16 weight for dgrad
```

FP8 autocast remains disabled, so activations are not quantized. Shared experts,
attention, DSA, routing, residuals, LoRA, and the output head remain unchanged.
Native storage fails closed unless LoRA and full-layer recompute are enabled.

## Exact Pushed Stack

- trainers: `2b94a5ef5f5706ec133cb6afc837b838f02b2ace`
  - PR: https://github.com/basetenlabs/trainers/pull/1222
- Megatron-Bridge: `f5dfc08c1446cdbe8fb9b868ea870f5ea2b131f2`
  - PR: https://github.com/basetenlabs/Megatron-Bridge/pull/54
- Megatron-LM: `8f5ac1e4efe051209ec20a69fd53fd6ef19c27bb`
  - PR: https://github.com/basetenlabs/Megatron-LM/pull/67

## Runtime Validation

The exact pushed stack ran on node 0 of `tj-w5y89m3`, using eight B300 GPUs:

```text
model: GLM-5.2 real native-FP8 0d1m snapshot
sequence length: 131072
topology: TP1 / PP1 / CP8 / EP8 / ETP1 / DP1
steps: one warmup + five controls
```

| Window | FB seconds | tok/s/GPU | Loss | Grad norm |
|---|---:|---:|---:|---:|
| warmup | 29.6 | 553 | 13.5304872931 | 0.0299957097 |
| control 0 | 1.454 | 11,266 | 13.5199920272 | 0.0287385043 |
| control 1 | 1.338 | 12,243 | 13.5253650693 | 0.0299385805 |
| control 2 | 1.314 | 12,467 | 13.5268156190 | 0.0314510539 |
| control 3 | 1.309 | 12,521 | 13.5194112351 | 0.0345148221 |
| control 4 | 1.306 | 12,542 | 13.5269920501 | 0.0378211439 |

All losses and gradient norms were finite; gradient norms were nonzero. The
control mean was 12,188 tok/s/GPU. The trainer reached step 6 and stopped
cleanly, with all eight GPUs returning to zero allocated model memory.

## Safety Gates

Startup validates that every routed grouped-linear parameter has:

- a uint8 rowwise payload;
- FP32 rowwise inverse scales;
- no columnwise/transposed FP8 payload;
- no persistent BF16 parameter.

Native mode rejects selective/no activation recompute because TE retains the
temporary BF16 weight until dgrad. Full-layer recompute bounds that lifetime to
the layer currently being processed during backward.

## Tests

- Repository `make check`: passed through the pre-push hook.
- Controller config: 18 passed.
- Native policy and wiring: 29 passed.
- Bridge direct payload/scale import: 5 passed.
- MCore recipe construction: 7 passed.
- Direct TE BF16 forward/dgrad/LoRA/recompute: 3 passed on B300.
- Real 0d1m forward/backward+optimizer: 6 consecutive steps passed.

## Artifacts

- `phase2_te_generic_final_pushed_steady5.json`
- `phase2_te_generic_final_pushed_trainer.log`
- `phase2_native_te_generic_real_fwdbwd.json`
- `phase2_boundary_comparison.json`
