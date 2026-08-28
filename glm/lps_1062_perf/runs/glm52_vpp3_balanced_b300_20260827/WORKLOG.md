# GLM-5.2 VPP3 stage-balance experiment

## Scope

- Base trainer SHA: `f1f34437d2d9e81eaf42110097bb26ff600c7e1e` (`main`).
- Branch: `jackrao/lps-1062-glm52-vpp-balanced`.
- Local worktree: `/Users/jackrao/Documents/trainers-wt-lps1062-glm52-vpp-balanced`.
- Devbox: `tj-w5y89m3`, 2 nodes x 8 B300 GPUs.
- Target topology: TP1/PP2/VPP3/CP8/EP8/ETP1/DP1.
- Target workload: GLM-5.2-FP8 LoRA SFT, sequence length 131072, four datums.

## Log

20260827 19:43 PDT Created an isolated trainer worktree and branch from the
requested main SHA. Preserved the existing dirty checkout unchanged.

20260827 19:43 PDT Chose an initial six-chunk DSA-valid VPP3 layout of
`[14, 12, 16, 12, 12, 12]` in VPP-major/PP-rank-major order. This assigns 42
decoder layers plus embedding to PP0 and 36 decoder layers plus the vocabulary
head/loss to PP1. Equal 13-layer chunks are invalid because GLM-5.2 shares DSA
top-k indices across four-layer groups; every chunk must start on a layer that
computes its own indices.

20260827 20:18 PDT Stopped before allocating a trainer after the experiment
target changed to FP8 base training. Reverted all VPP implementation commits on
PR #1210. No VPP GPU result was produced.
