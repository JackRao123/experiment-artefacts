# CONTINGENCY — upstream engagement note for cudnn-frontend #543 / #538

Author: serre, 2026-08-13, for cauchy. **Status: contingency only.** To be
sent (by Jack, through whatever vendor channel he prefers) ONLY if the
cudnn-frontend wheel bump (1.26.0+dsatopk1 → 1.27.0) does NOT resolve the
THD-varlen+CP attention corruption. Do not fire autonomously.

---

Draft message:

**Subject: GLM-5.2 DSA at 131k THD+CP — nondeterministic forward corruption;
does the DSA namespace share the FROST THD execute path's layout-keyed
caching?**

Hello — Baseten here (we're the folks behind cudnn-frontend #446/#406 and
TE #3331; the DSA top-k and pad_between_seqs fixes are already in — thanks
for the quick turnarounds).

New issue class on our side, asking for guidance before we burn deeper
debug time:

**Setup.** GLM-5.2 (MLA + DSA + MoE) training on B300 (SM103), torch
2.11/cu13, cudnn-frontend 1.27.0, TE 2.16, Megatron-Core with THD-packed
varlen + context parallelism (CP8/CP16) at 131k tokens. LoRA adapters only
(base frozen), full activation recompute.

**Observed.** In the forward pass, across successive invocations with
*varying packed-segment layouts* (THD partitions differ per call), we see
numerical corruption that (a) appears partway into invocations whose packed
extent is larger than their predecessor's, (b) escalates toward the tail, and
(c) is nondeterministic run-to-run (uninitialized-read / stale-reuse
signature). [Version/wheel details and the exact reproducer geometry get
filled in after the wheel-bump A/B — if we're sending this, the bump didn't
fix it and we'll attach the minimal repro.]

**Why we're writing you.** The signature points at an execution-plan or
workspace cache keyed too coarsely (layout not in the key) — the class your
team is already tracking on the FROST side as #552 (per-total compile keyed
on packed totals) and #538 (cu_seq_len form assumes packed strides), with
#543 (plan-time-only compile keys + launch-stream-bound host prep) in flight.
Our question: **does the `cudnn.DSA` namespace path (indexer forward /
sparse-attention wrappers, the CuTeDSL DSA kernels) share that execute-path
caching and host-prep machinery, or does it key/size its workspaces
differently?** If it shares, we'd like to (1) confirm whether #543 as drafted
covers the DSA call sites or only FROST SDPA, and (2) get your read on
whether a layout-keyed workspace reuse in the DSA path is a known open issue
before we root-cause it ourselves.

**What we've ruled out.** [To fill post-A/B: wheel version, backend version,
stream pinning, the #410 top-k fix, pad_between_seqs handling, hardware.]

Happy to provide the minimal reproducer, traces, and to test any patch on
our B300 capacity. — Jack Rao, Baseten

---

## Notes for whoever sends it

- The bracketed sections fill in from the wheel-bump A/B outcome (if this
  sends, the bump failed — say so explicitly and include both wheels' exact
  versions).
- The ask is deliberately narrow (does DSA share the FROST caching/prep
  path; does #543 cover DSA) — that is the answer that routes our next week,
  and it is cheap for them to answer.
- If gauss's code read lands on a specific mcore-glue cache first (the
  `lru_cache` in dsa_cudnn_kernels.py:329 or the layout caches), this note
  should NOT send — the bug would be ours, not upstream's.
