# GLM-5.2 Post-Merge Optimization Worklog

## Contract

- Date: 2026-08-29
- Target: full GLM-5.2 training throughput
- Iteration proxy: 0d1m debug model
- Hardware: one 8xB300 node from `tj-w5y89m3`
- Shape: sequence length 131072, one datum
- Mesh: TP1 / PP1 / CP8 / EP8 / ETP1 / DP1
- Driver: `profile_driver.py`
- Acceptance metric: unprofiled steady tok/s/GPU and MFU
- Correctness: finite loss, finite nonzero grad norm, completed optimizer steps
- Training semantics: natural routing; no forced load balancing

## Repository State

- Starting trainers base: `b1b28f8850bda62f70c4025b5df30555d4173e80`
- Starting branch: `origin/main`
- #1222 is merged at `b485c642b353d23b15116c574f6de1f018fe0a87`
- Compiled materialization PRs #68 / #55 / #1228 are open drafts, not on main

## Baselines

- Exact pushed SHA: `b1b28f8850bda62f70c4025b5df30555d4173e80`.
- Debug proxy, 10 stabilized controls: **12,559 tok/s/GPU**, driver MFU3x
  **114.3%**, 1.304 s mean FB. Losses were finite and grad norms finite and
  nonzero. The debug MFU exceeds 100% because the driver applies full-model
  FLOP accounting to the one-layer proxy; use it only for matched A/Bs.
- Full model, 10 stabilized controls: **1,119 tok/s/GPU**, MFU3x **10.2%**,
  14.6 s mean FB. Losses were finite and grad norms finite and nonzero. This
  matches the earlier TE-generic 1,117.7 tok/s/GPU measurement within noise.

## Experiments

- Rejected the first debug run because the copied lifecycle script used the
  isolated source path but the reused editable venv still imported Python
  packages from `trainers-native-fp8-final`. No result from that run is used.
- Built a checkout-local venv clone and rewired every editable package to the
  exact `b1b28f885` checkout. Verified imports for trainers, Bridge, MCore, and
  model configs resolve from `trainers-postmerge-opt` before rerunning.
- Candidate 1: BF16-input/FP32-output GLM LM head. Pushed trainers SHA
  `d0295ede0` and opened PR
  `https://github.com/basetenlabs/trainers/pull/1231`. The change uses pinned
  MCore's mixed-output GEMM instead of widening the BF16 hidden state and frozen
  LM-head weight to FP32. Focused tests: 21 passed. `make check`: passed.
- Candidate 1 debug screen: **35,520 tok/s/GPU**, matched proxy MFU3x
  **323.3%**, versus 12,559 and 114.3% at baseline. This 182.8% proxy gain is
  not full-model acceptance evidence because the one-layer model overweights
  the LM head.
- Candidate 1 full model, first 10 controls: **1,162.0 tok/s/GPU**, MFU3x
  **10.58%**, 14.099 s mean FB: +3.86% versus baseline.
- Candidate 1 full model, repeat 10 controls: **1,169.0 tok/s/GPU**, MFU3x
  **10.64%**, 14.015 s mean FB: +4.48% versus baseline.
- Full runtime trace mechanism check: the baseline had four forward SGEMMs
  totaling 467.4 ms and four backward SGEMMs totaling 457.6 ms. The candidate
  trace contains zero SGEMM kernels. This removes the targeted 925.0 ms kernel
  class without claiming all 925.0 ms as exposed wall time.
- Review found that the original FP32-head predicate covered every DSA model,
  while the first mixed-output provider predicate covered only GLM-5.2/5.3.
  Follow-up SHA `98785c05c` aligns those predicates so GLM-5/5.1 retain their
  existing FP32-logit behavior. The measured GLM-5.2 path is unchanged.
  Focused tests after the fix: 22 passed. `make check`: passed.
- PR head `98785c05c` passed GitHub `check` and is ready for review.

## Decisions

- Do not treat inclusive `aten::nonzero` or `cudaStreamSynchronize` duration as wall time.
- Profile and compare one variable at a time.
- Push every profiled revision and record its SHA here.
- Accept candidate 1 as a full-model winner. Keep subsequent screens on the
  debug proxy; do not spend another full-model load on marginal knobs.
- The candidate full trace's largest remaining kernels are HybridEP device sync
  (1.770 s) and DSA backward (1.679 s). HybridEP sync may be cross-rank waiting,
  so do not optimize it from a rank-0 trace alone.
