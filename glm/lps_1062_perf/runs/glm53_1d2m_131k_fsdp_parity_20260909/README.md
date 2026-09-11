# Matched-initialization DDP versus MFSDP numerical experiment

CP8EP1, BF16, 1d2m, 131072 tokens, full one-layer recompute. Identical source
checkpoint and exact constant BF16-representable LoRA A values, zero B.
Learning rate zero so the five control gradient norms should repeat and be
comparable. This is an experiment-only adapter initialization, not the normal
training recipe and not a performance headline. No tests are added or changed.

Purpose: distinguish expected RNG/initialization changes under meta-device
construction from a gradient/reduction error before scaling to the full model.
