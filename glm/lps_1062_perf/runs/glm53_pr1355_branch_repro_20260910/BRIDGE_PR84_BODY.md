Experimental source dependency for basetenlabs/trainers#1355; pins Core #76.

- Scope bounded safetensors reader reuse to one immutable checkpoint import.
- Use indexed exact-key lookup while that scope is active; retain glob and
  ordinary uncached behavior.
- Let GLM FP8 checkpoint tensors dequantize on a caller-selected device during
  that import scope, with the original BF16 conversion arithmetic.
- Restore reader/device state and close all cached handles on exit.

These replace the old run-local loader monkeypatches. Trainer #1355 calls
the committed APIs after FSDP construction/sharding. No new dependencies,
test edits, or experiment artifacts are included by these integration commits.
Changed Bridge files pass pre-commit checks. Clean-branch GPU reproduction
passed three debug variants and full GLM-5.3 CP8 at 131072 tokens; this remains
a draft experiment, not a production-readiness claim.
