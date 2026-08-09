# Upstream fix research (web subagent, 2026-08-08 ~02:00 PT)

Question: is the CP fused-attn forward nondeterminism already known/fixed upstream?

## Verdict
**No matching upstream report found** in TE <= 2.17.1, cuDNN release notes
9.19 -> 9.25, cudnn-frontend <= 1.27.0, or NeMo/Megatron trackers.
If it reproduces on TE main (post-#3186) with cuDNN 9.25 -> novel bug, file it.

## Closest items (and why they don't match)
- **TE PR #3186** "Pass cu_seqlens and token-unit ragged offsets directly to
  cuDNN SDPA fprop" (merged 2026-07-14, MAIN ONLY — not in v2.16/2.17/2.17.1;
  verified old code still present in those tags). Removed a real cross-thread
  race in `cu_seqlens_padded_to_offsets` (fused_attn/utils.cu): for
  quantized-batch tail entries (tid > actual_b), `offsets_v[tid] =
  offsets_k[cu_seqlens_id]` reads another thread's write unsynchronized ->
  nondeterministic V ragged offsets, FORWARD only. Caveat: race exists only
  for interleaved layout groups (3HD/H3D/HD_2HD); our CP path passes
  qkv_layout=thd_thd_thd (HD_HD_HD) whose branch computes offsets_k/offsets_v
  independently. Probably not our bug — rule out empirically (TE main build,
  or NVTE_FUSED_ATTN_DIRECT_SEQLENS on cuDNN >= 9.24).
- **TE issue #2186**: THD+CP NaN at tail of a rank's SECOND chunk — same code
  region as our wobble, but backward, deterministic, Hopper-only, fixed by
  cuDNN 9.18 (older than our 9.19). Cite as region-bug-history when filing.
- **TE issue #3285**: FusedAttention hang under CP on sm103/cuDNN 9.21 — open;
  hang not wrong-results.
- cuDNN fixed issues nearby: 9.23.2 MXFP8 SDPA bwd mismatches on Blackwell;
  9.21.1 SDPA causal fwd hang on Blackwell; 9.19.1 FP8 SDPA hang/incorrect
  (d<64). None match bf16 varlen fwd nondeterminism.
- cudnn-frontend v1.27.0 (2026-08-06): deterministic-BACKWARD workspace on
  Hopper + API additions; no fprop correctness fix.
- TE v2.16.1/v2.17.1 = security fix only; v2.17 fixed-issues has nothing
  CP/attention-determinism related (does blacklist cuDNN 9.23.0/9.23.1 for
  MXFP8 attention). No correctness/race fix to context_parallel.py since
  v2.16 (2026-06-09).

## Version facts
- TE latest: 2.17.1 (PyPI 2026-08-05). PR #3186 in NO release.
- nvidia-cudnn-cu12/cu13 latest stable: 9.25.0.15 (2026-07-20); prev 9.24.0.43;
  cu13 dev builds 9.26.0.x exist. We run 9.19 — six point-releases behind.
  Blackwell SDPA visibly churning across 9.21-9.23 (several undocumented
  fwd fixes + TE blacklisting 9.23.0/1).
- cudnn-frontend latest: 1.27.0.

## Implications for the version matrix (feynman work item C)
Priority order: cuDNN 9.25.0.15, 9.24.0.43, one 9.21.x; TE main (post-#3186)
if buildable, else NVTE_FUSED_ATTN_DIRECT_SEQLENS toggle on cuDNN >= 9.24.
