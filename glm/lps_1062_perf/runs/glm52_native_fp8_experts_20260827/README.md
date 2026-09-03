# GLM native-FP8 expert campaign

This directory groups the related FP8 expert-storage, loading, memory, and
performance experiments run from 2026-08-27 through 2026-09-01.

Despite the parent directory's historical `glm52` name, the final phase uses
full GLM-5.3. Keep model versions and exact trainer revisions distinct when
comparing results.

## Experiment progression

1. `glm52_fp8_base_b300_20260827/`
   - Established tensorwise FP8 as the GLM-5.2 B300 baseline.
   - Compared FP8 with BF16 and evaluated PP2 versus PP1.
   - Start with `WORKLOG.md` and `ARTIFACTS.md`.

2. `glm52_native_blockwise_fp8_20260828/`
   - Implemented native Hugging Face blockwise-FP8 loading into Transformer
     Engine blockwise parameters.
   - Found the path numerically correct but slower and substantially larger in
     peak memory than tensorwise FP8, so it remained opt-in.
   - Start with `RESULTS.md` and `ARTIFACTS.md`.

3. `glm_native_fp8_selective_moe_20260831/`
   - Investigated GLM-5.3 native-FP8 recompute requirements, expert-weight
     rematerialization, pipeline overlap, and mixed-output LM-head execution.
   - The shipped work removed the artificial full-recompute dependency for
     frozen expert weights and improved the LM-head path.
   - Start with `RESULTS.md` and `WORKLOG.md`.

## Artifact policy

Artifacts remain inside their original experiment directories to preserve
relative paths and provenance. Large traces are not interchangeable: they
cover different model versions, code revisions, topologies, and experimental
arms.
