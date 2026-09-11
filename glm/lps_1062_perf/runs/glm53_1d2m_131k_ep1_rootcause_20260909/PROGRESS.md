# Overnight investigation progress

Completed:

- Three 1d2m BF16 topology comparisons at 131072 tokens, three warmups, five
  controls, twenty follow-up controls, memory and all-rank Kineto captures.
- Core singleton identity-sort elimination and all artifact validation.
- Quantitative explanation of fixed request overhead, GEMM shape efficiency,
  small intra-node dispatch/combine cost, and rank-dependent permutation tuning.
- FSDP+CP8EP1 debug run and matched-initialization DDP/FSDP loss/norm checks.
- Full-model CP8EP1 dynamic-buffer diagnostic: 84.200 ± 4.501 s, 194.58 TPS/GPU.
- Full-model CP8EP1 persistent-buffer result: 11.714 ± 0.244 s, 1398.64 TPS/GPU,
  248.103 GiB peak allocated / 252.785 GiB reserved. Mapping churn eliminated.
- Startup fixes: allocation-aware reclamation, cached root parameter count,
  bounded readers, indexed lookup and GPU dequantization during immutable HF import.
- CP4 with lookahead: startup succeeds after empty-DP alignment fix, but 131k
  warmup OOMs in FP32 LM-head logit addition. Logs retained; no ordinary profiles.
- Scoped cleanup of orphaned run-owned helper groups after the OOM.

Additional completed work:

- No-lookahead CP4 completed at 26.425 seconds / 1240 TPS/GPU but had 16
  allocator retries across 40 rank-controls. It remains diagnostic.
- No-lookahead CP2 OOMed in a 6-GiB expert-output allocation before the LM
  head; live-state OOM snapshots were captured.
- The accepted opt-in FP32 head fusion passed bitwise output/gradient checks
  and lowered full-chunk incremental peak from 8.328 to 5.908 GiB. Rejected
  rounding-changing and higher-memory attempts are preserved, not deployed.

Complete: CP4 with no lookahead, opt-in head fusion and 2048-token head chunks
measured 21.527 ± 0.360 seconds / 1522.15 TPS/GPU, 252.64 GiB peak allocated,
and zero allocator retries/frees/wide syncs across forty rank-controls.
Artifacts passed validation. CP4 is the lowest CP measured successfully.
All owned trainers/helpers are stopped; the devbox remains available.

Final summaries and artifact links are being published to trainers PR #1355.
No tests or original tools/profile_driver.py / tools/mfu.py are changed.
