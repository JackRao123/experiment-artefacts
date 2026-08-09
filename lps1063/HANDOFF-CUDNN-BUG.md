# LPS-1063 overnight handoff: root-cause to full evidence + FIX the CP fused-attn forward nondeterminism

Owner tonight: **feynman** (fresh context, does the work).
Supervisor: **ramanujan** (context-constrained; send CONCISE milestone updates
via `~/.agents/scripts/send-message.sh ramanujan "..."` — findings + numbers,
no log dumps). Jack is away; work autonomously overnight. Directness rules
apply: evidence only, no speculation in conclusions.

## Context (read these first, in order)

1. `NOTEBOOK.md` in this folder — session 3 (bottom) is the current state.
2. `evidence-phase5-determinism-arms/` — all evidence incl. the repro script.
3. Memory: `lps-1063-repro-state.md` (ROOT-CAUSED paragraph at top).

One-paragraph state: forward nondeterminism at CP>1 is pinned to the TE/cuDNN
fused-attention FORWARD under context parallelism (thd + p2p ring). Given
bitwise-identical inputs, output wobbles at EXACTLY one row = tail region of
each rank's SECOND load-balanced CP chunk (local row T_LOCAL-2), intermittent
(3-7 distinct outputs per 20 iters at CP4; rare at CP2; CP ranks whose 2nd
chunk is padding don't show it). Reproduced standalone WITHOUT any model:
`evidence-phase5-determinism-arms/standalone_te_cp4_repro.py` (also
`~/nemotron-repro/4node/`), 4 GPUs, ~2 min. Stack proven: B300 + cu13 +
TE 2.16.0 + cuDNN 9.19 (91900), trainers branch pin 0e0b65a6.
NOT fixed by: NVTE_ALLOW_NONDETERMINISTIC_ALGO=0, CUBLAS_WORKSPACE_CONFIG,
NCCL_ALGO=Ring. attention_backend=flash silently falls back to cuDNN for this
geometry (thd+CP+MQA), so no backend escape.

## Mission (Jack's bar): evidenced, proven, fixed

1. Exact mechanism with evidence at every link — no speculation.
2. Check if already fixed upstream (TE develop / newer cuDNN / release notes).
3. If not: build the fix, prove it works (repro goes quiet over many iters,
   AND the real trainer arm goes bitwise-deterministic), know exactly why.

## Resources

- **B200 1-node box on vul: job `wprr2p3`** (provisioning started 01:31 UTC by
  ramanujan via `devbox-up 1 b200 vul`; alias `tj-wprr2p3` appears in
  ~/.ssh/config when done; if provisioning died, resume: `devbox-up wprr2p3`).
  NOTE vul CPFS ≠ ali CPFS — push files from laptop, verify content per the
  zero-fill gotcha memory.
- B300 boxes: `devbox-up <n> b300 ali` if/when needed (Jack: B300→ali,
  B200→vul). The old 4-node box is torn down. Full trainer re-validation at
  the end needs 4×B300 on ali (TP8/CP4/EP32 driver arm, harness in
  `~/nemotron-repro/4node/`, RUNBOOK-equivalent in NOTEBOOK session 3;
  cudart-12 purge script + setup_on_box.sh are in that dir).
- Laptop harness dir: `~/nemotron-repro/4node/` (standalone script, driver,
  probe hooks, analyzers).
- Venv build gotchas: memories `devbox-multinode-trainer-launch-gotchas`,
  `cpfs-scp-zero-fill-gotcha`, `vllm-on-baseten-jobs` (cu12 driver caveats).

## Work plan (ordered; adjust with judgment, report deviations)

### A. Nightly-stack validation (B200 + cu12 + trainers@4ec66ce3)
On tj-wprr2p3: clone/checkout trainers @4ec66ce3 (the failing nightly's pin;
/root/trainers is a token-backed clone, `git fetch origin 4ec66ce3` works),
build server venv (cu12 lane; needs python3.12-dev ninja pybind11, CUDNN_PATH
per memory). Run standalone repro: CP4, CP2, CP1(=world 1; small script tweak:
skip set_context_parallel_group when world==1 — this also definitively answers
"is CP required"). Record TE + cuDNN + torch versions. Expected: CP4 wobbles →
the nightly's own stack is proven affected end-to-end. If it does NOT wobble:
big finding (cu13/B300-specific) — report immediately.

### B. Ring-step isolation → single-call repro (the decisive narrowing)
TE 2.16's CP attention is Python-orchestrated:
`transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py`
(class/function `attn_forward_func_with_cp`, P2P branch). Instrument it
(sitecustomize monkeypatch or venv-copy edit — NEVER the shared clone):
- Per ring step per iteration: fingerprint (sha) the step's INPUTS (q, kv
  buffer, cu_seqlens views) and OUTPUTS (per-step out + softmax_lse/stats)
  at fixed seed.
- Find the FIRST wobbling tensor: which step, input-side or output-side?
  - If a step's kernel OUTPUT wobbles at bitwise-identical step inputs →
    capture that step's exact tensors (torch.save) and build a
    single-GPU, no-CP replay of that one cuDNN call (same
    fused_attn_fwd args). If the replay wobbles: **minimal single-call
    NVIDIA repro** — the strongest artifact possible.
  - If a step's INPUT (kv double-buffer) wobbles → **TE-side race**
    (p2p buffer overwritten before consumption; look at the send/recv
    stream events around the double-buffered kv exchange, and the
    softmax_lse correction kernels). That would make it OUR fixable bug
    (TE is open source) — likely a missing wait_stream/record_stream on
    a specific step/chunk. The "tail of second chunk" locality is
    consistent with either; let the fingerprints decide.
- The wobble is intermittent (10-30%/call at CP4): fingerprint across ≥30
  iters per config.

### C. Already-fixed-upstream check
ramanujan is running a web-research subagent in parallel and will forward
results. Independently useful on-box: version matrix with the standalone —
swap `nvidia-cudnn-cu13` (and/or cu12) wheel versions in a scratch venv
(9.19 baseline → latest 9.2x; also one older, e.g. 9.14/9.16 if resolvable)
and TE versions (2.16.0 baseline → latest release → git main if buildable;
TE build is heavy, prefer prebuilt wheels). Matrix: version × fires/quiet
(≥60 iters each). This both finds an existing fix AND brackets the
regression window for the NVIDIA report.

### D. The fix, proven
Depending on B's outcome:
- TE-side race → patch TE python (venv copy), rerun standalone ≥100 iters
  × 3 seeds × CP{2,4} → 0 distinct-output violations. Then full-fidelity:
  4×B300 ali box, trainer with patched TE (PYTHONPATH shadow of the TE
  python files works — they're .py), 10-fwd driver arm → expect
  BITWISE_DETERMINISTIC. That's Jack's bar.
- cuDNN-internal + fixed in newer cuDNN → prove by wheel swap (standalone
  quiet ≥100 iters), then trainer arm with swapped wheel if compatible.
- cuDNN-internal + NOT fixed anywhere → mitigation engineering: identify
  the guilty engine via CUDNN_LOGLEVEL_DBG and test engine-exclusion
  (cudnn-frontend errata filter if TE exposes a path; or TE-side kv-layout/
  chunk-size change that dodges the schedule). Plus the gate fix
  (fingerprint-compare) as the product-level mitigation. Document exactly
  what was tried with numbers.

### E. Bookkeeping (non-negotiable)
- Append findings to NOTEBOOK.md (this folder) as you go — session 4 header.
- Evidence files → `evidence-phase6-mechanism/` in this folder.
- Update memory `lps-1063-repro-state.md` at major state changes.
- Tear down boxes you're done with; don't hold idle GPUs.
- Message ramanujan at: (1) box up + A result, (2) B verdict (the decisive
  one), (3) C matrix result, (4) fix candidate + validation numbers,
  (5) any blocker >30 min. Keep messages <10 lines.

## Known traps (save yourself hours)

- ssh relay drops kill foreground remote jobs — always nohup + logfile on
  the box, and never end compound ssh with `| tail`.
- pkill/pgrep self-match; `[b]racket` de-fang only protects that line.
- uv venvs on shared FS dangle across boxes (re-`uv sync` on the consumer).
- The standalone script exits rc=1 when nondeterministic BY DESIGN —
  torchrun then prints a scary ChildFailedError. Read the [seed ...] lines.
- Wobble is in the LAST ~2 rows of the 2nd chunk; padding-region rows only
  wobble if fed non-zero values (trainer pads zeros → CP0 looked clean).
