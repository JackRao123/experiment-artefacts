# PR #875 review cycle + minimal repro + caller-side A/B (2026-08-02 evening → 08-03)

Status snapshot after the review cycle. Supersedes stale references to
`+dsatopk4` elsewhere in these artefacts: **the shipping wheel is
`1.26.0+dsatopk5`** (wheel sha256 `53594ecf…`, `_interface.py` sha256
`7816f1199e3df7a2…`). PR rebased onto main; `/test pr-tests` green (unit +
2xh200 + required 1xh100/1xb200).

## 1. +dsatopk5: review revision of the launch-stream patch

Jack's review call: the `try/except Exception: pass` around caller-stream
identification should fail loudly, not guess. Revision (commit `56a0c9b11`,
rebased `7daf8c64d`):

- try/except removed — an explicitly-passed `current_stream` handle that
  fails `int()` conversion now **raises** out of `indexer_fwd` instead of
  silently event-chaining against torch's current stream.
- Behavior-neutral for every first-party call site: all six
  `dsa_cudnn_kernels.py` wrappers pass `stream=None`, where the branch is
  never taken (`resolve_stream(None)` == torch current stream by
  construction).
- Wheel re-tagged +dsatopk4 → +dsatopk5 per the vendor-README convention
  (bump on any patch revision; two different binaries must not share a
  tag). pyproject/uv.lock repointed; regenerated patch file committed.
- NOTE: the 100-step recipe validation (`FIX_VALIDATION_100STEP.md`) ran on
  the +dsatopk4 content (`bba5b2e7…`). The v4→v5 delta is provably
  unreachable on our call path and the single-GPU A/B below is green on v5;
  Jack accepted without re-running the 100-step campaign.

Regression-test hardening in the same cycle (review items):

- The vendored-wheel import now **hard-fails** (not skips) on any platform
  where pyproject vendors the wheel (linux/x86_64 cp312 — what CI syncs via
  `uv sync --extra worker`); skips only on foreign platforms (darwin).
  The regression guard can no longer vanish silently from CI.
- The GPU test was rewritten from a fingerprint check into a **true repro**
  (below) after A/B showed the original small-geometry version passed even
  on the unfixed wheel.

## 2. Minimal single-GPU standalone repro (new capability)

Found while strengthening the test — the overnight campaign never had a
minimal repro; this confirms `rearm/DETERMINISM_MECHANISM_0801.md`'s
fill-split mechanism in isolation:

**The race manifests iff the `-inf` prefill exceeds 2^31 BYTES** (torch's
TensorIterator splits elementwise kernels into two launches at INT32_MAX
bytes; the *second* fill launch is the one that lands after the DSL
kernel's stores — hence prod's "trailing halves of packed rows").

Recipe: plain `indexer_fwd`, caller on default stream, conn unset,
`torch.cuda.empty_cache()` before the call, run twice, compare bitwise.
Single GPU, no trainer, no multi-rank. Measured on tj-q8x5ky3 (B300),
unfixed wheel, 32 heads × hd128, 2 THD segments:

| out tensor | fill launches | double-exec disagreements |
|---|---|---|
| 1 MiB (512×512) | 1 | 0/8 |
| 0.06–2.00 GiB | 1 | 0/8, 0/6 |
| **2.25 GiB** (8192×73728 fp32) | 2 | **6/6** |
| **4.00 GiB** | 2 | **8/8** |

All disagreements carry the pure erasure signature: finite-mask flips only,
**zero** numeric diff on both-finite positions. Segment count irrelevant
(2.0 GiB @ 4 segs: 0/6).

Productionised as `test_indexer_fwd_bitwise_stable_across_cold_starts`
(`server/tests/unit/dp_worker/test_cudnn_dsa_indexer_launch_stream.py`):
3 reps × (empty_cache → double-exec → cross-rep baseline compare) at the
2.25 GiB geometry, ~7 GiB GPU, sm100+ only. Full A/B on the devbox:
**fix removed → FAILS at rep0; +dsatopk5 → 15/15 across 5 fresh
processes.** The CPU-only arrangement assertion (source inspection of the
installed wheel) remains the everywhere-guard for "wheel lost the patch".

Probe scripts: devbox `/root/lps1003_review/` (`probe_race.py`,
`probe_caller_stream.py`, plus `interface_unfixed.py` /
`interface_fixed_v5.py` / `interface_dsatopk4_inplace.bak` for swapping).

## 3. Caller-side A/B (Paras's review question) — ExternalStream(0) specificity proven

pstefa1707 asked whether the trainer could simply call from a non-default
stream instead of patching the wheel. Measured (pristine wheel, 2.25 GiB
geometry, 6 reps/arm, same session):

| arm | result |
|---|---|
| A control: caller on default stream | **6/6** disagree |
| B ambient: `with torch.cuda.stream(s)` + wait_stream both ways | 0/6 |
| C explicit: `current_stream=CUstream(s.cuda_stream)` | 0/6 |

Confirms the root cause end-to-end: the DSL dispatch honors any handle it's
given; only handle **0** is poisoned, because the wheel glue's
`ExternalStream(int(handle))` wrap silently yields a pool stream for 0.
Any nonzero stream makes the pristine glue correct.

Decision (posted on the PR): wheel patch stays for this PR (defect is
between the wheel's own two ops under its documented-default calling
convention; call sites are also vendored; ambient wrap re-tags multi-GiB
allocations to the side stream = cross-stream lifetime obligations the
probe doesn't exercise; validation sunk). Caller-side helper adopted as the
plan for the **family audit** (FOLLOWUP #10/#11): our GLM path calls six
wheel entry points sharing the glue — indexer fwd (fixed), indexer_top_k
(OOB-patched; stream-pin drafted), indexer_backward,
dense_indexer_backward, sparse/dense score_recompute (score_recompute has
the most torch-side ops under stream contexts: ~24 sm90 / 14 sm100). One
dedicated-stream wrapper in `dsa_cudnn_kernels.py` + per-site double-exec
verification at prod geometry covers the family without five more wheel
patches.

## 4. Prod exposure — corrected framing (per Jack) + live check 08-03

**conn=1 was never deployed as a prod mitigation.** `5wolkzw` was the
investigation's prod TEST deployment only (conn=1 applied via LWS patch to
validate the masking lever; idle at 0 replicas since). Do not describe it
as "prod is masked". PR body corrected accordingly.

Live check (ali cluster, org-99340d71…, 08-03): four running GLM trainers
(`232ln0w`, `4w5yx73`, `v31gz13`, `zq8y7pw`) on image
`trainer-cuda13-sm103-5a4ae4d` — **47 commits before the fix**, no conn=1
env → currently exposed to the cold-start erasure. The three PR commits
are already ported to the `trainer-cuda13-sm103` branch tip (`3ef3b1952`,
carries +dsatopk5), so the close-out path is: build image from tip →
recycle sessions onto it (customer coordination; supersedes the old
"recycle onto 0e0b65a" advice in FOLLOWUP #4).
