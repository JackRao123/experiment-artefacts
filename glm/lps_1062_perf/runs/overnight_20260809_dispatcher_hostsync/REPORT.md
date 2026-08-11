# LPS-1062 host-sync elimination ("the aten::nonzero thing") — final report

**Date:** 2026-08-09 (day) · **Box:** qr4ggv3 (2×8 B300, ali) · **Stack:** trainers 0e0b65a6 + Aug-7 patches, vendored mcore d3932e757c · **Team:** pascal (orchestrator), laplace (attribution/verification), ramanujan (implementation), gibbs (validation) · **Model:** GLM-5.2-FP8, golden EP16/CP16, LoRA r32, full recompute, ship NCCL env + TF32 head.

## Result

| | tok/s/GPU steady (131k×d4, 524K tok/step) | step (s) | mfu3x / hfu (LoRA-corrected) |
|---|---:|---:|---:|
| Unpatched (anchor + replicate) | 634–645 | 52.0 | 5.7% / 8.3% |
| **SHIP: B+F gates on** | **~715** | **46.8** | **6.4% / 9.3%** |
| A+B+F (all gates) | ~707 | 46.3 | 6.1% / 8.9% |

- **+11–13% steady throughput at 131k, all from FIX B + FIX F — and +18.5% at the customer-dominant 16k×d32 shape (726 vs 612 unpatched matched-step; step 53.6→45.2s)**: more docs per step ⇒ more host bookkeeping eliminated. 726 is within 1% of the DP-gated CP8/DP2 record (734) with no F2 exposure.
- Loss canary ≤2e-3 on every patched run at 131k; 16k matched-step deconfound **≤5e-4 all windows (BENIGN** — the +35e-3 vs the expB cross-config ref was mesh reduction-order + step-count confound, not the patch). Memory flat. Baseline triple-replicated (633–645).
- **FIX A measured ≈ 0 (−1%, within noise)** by same-code gate-off/on isolation. Parked default-OFF (see below).

## The premise correction (F6)

Aug-7's "26,684 aten::nonzero / 20.6s = MoE dispatcher split bookkeeping, co-dominant lever" was wrong on both counts:

1. **Wrong site:** 93% of the nonzero CPU is one `torch.nonzero(topk_length>0)` per layer-backward in **DSA sparse-attention backward** (`dsa_cudnn_kernels.py`, 156–312 calls/step); the 26.5k small calls are **DSA CP layout builders** (`dsa_layout.py`, per-microbatch-constant, replayed by recompute); the 15s memcpy class is **THD RoPE bookkeeping** (`rope_utils.py` `_apply_rotary_pos_emb_thd` tolist/item, not `sort_chunks_by_idxs` — TE fused chunk_sort is active). The dispatcher itself (mcore 0.19.0) is already side-stream + deferred-event; no per-expert nonzero loops exist.
2. **Wrong impact model:** the blocks ride in comm slack on a single rank (GPU idle inside host-block windows: 0.70s), and total host-block CPU is **M-invariant** (per-block wait ∝ queue depth ∝ 1/M at fixed tokens). The measured win did NOT come from unblocking the GPU directly but from **launch-pipeline decompression** (verdict per pre-registered discriminators: SendRecv flat 25.8→26.0s, GPU-union idle 8.52→1.82s ≈ the wall win, host-block coverage 60%→1%). The starvation *share* is shape-dependent even though block totals are M-invariant — which is why the 2-mb trace under-bounded the 4-mb win.

## The three fixes (all env-gated, default OFF, bitwise-parity-tested)

| gate | site | what | verdict |
|---|---|---|---|
| `BT_DSA_CP_LAYOUT_CACHE` (**B**) | dsa layout builders | cache packed-CP layout per microbatch on the packed_seq_params carrier (26,520→~340 nonzeros, −150k kernel launches/step) | **SHIP** |
| `BT_THD_ROPE_HOST_CACHE` (**F**) | THD RoPE | cached host copy per unique cu_seqlens, keyed (identity, `_version`) — blocking DtoH 587→4, 14.8s→0.16s CPU | **SHIP** |
| `BT_DSA_BWD_ASYNC_NONEMPTY` (**A**) | DSA bwd compaction | async 1-byte nonempty flag (fwd side-stream) replacing the syncing nonzero — works exactly as designed (312 events, p50 5.5µs) | **PARK** — zero win today: the drains it removes are already throttled cheap by the dispatcher's upstream replay eventSync (77ms p50). Revisit AFTER FIX C; v3 sketch (first-pass anchoring) in PATCH_NOTES.md |

Patches + parity tests: `patches/` (final on-box diff `box-applied-qr4ggv3-v2.patch`), tests exercise the production autograd/checkpoint-replay context (the v1 lesson), forced-fallback + mutation-miss cases included. Activation telemetry (WARNING-level armed/disabled lines + per-window counters) is mandatory equipment: v1 shipped inert behind a gate that could never fire (`is_grad_enabled()` inside `Function.forward` is always False) and passed 22/22 parity — only trace counters caught it.

## Verification kit (reusable)

- `check_acceptance.py` — trace_processor counter checker; profiles: baseline-exp05d, baseline-4mb131k (measured-calibrated), post-patch-A/-B/-AB/-ABF(+4mb variant); validated positive+negative on both baseline traces.
- `ATTRIBUTION.md` (+ addendum: corrected win accounting, M-invariance law, eventsync runahead-inflation analysis) · `CANDIDATE_FIXES_ramanujan.md` · `REVIEW_FIXA.md` · `PATCH_NOTES.md` · traces in `traces/` + `~/perf_profiles/lps-1062/`.

## Follow-ups / tickets

1. **Ship B+F**: default the two gates on for GLM THD-CP trainer pods after a soak (they are inert-by-default in the shared checkout today); upstream candidates for the vendored mcore fork.
2. **FIX C — dispatcher replay-metadata reuse** is now the quantified next host-sync lever: 300 replay eventSyncs = 23.7s CPU (runahead-inflated post-B+F), keying design in CANDIDATE_FIXES §C. Then A-v3 becomes worth measuring: re-run the A/B with `BT_DSA_BWD_ASYNC_NONEMPTY` and check the `eventsync_a_*` rows in `check_acceptance.py` — the drains become exposed once the dispatcher throttle is gone (laplace's parting note).
3. Comm (SendRecv ~26s, ~55% of the 46.8s step) is again the dominant lever → DeepEP fabric qualification (Aug-7 ticket) or the F2 fix + CP8/DPn meshes (Aug-9 synthesis).
4. MFU convention: LoRA-corrected accounting landed 2026-08-09 PM (`runs/overnight_20260809_mfu_sweep/mfu_lora_correction.md`) — all published LPS-1062 MFU converted; use `mfu.py`'s `mfu3x(tps, L, lora_rank)` going forward.

## Ops notes

- Boots on one box: unpatched anchor ×2, gated-v1 (accidental BF-only — the inert-A bisection), gated-v2 (ABF), BF-only, gates-off captures, 16k deconfound. No first-step profiled captures (window-1 graph-capture/compile contamination); steady windows only.
- MFU-formula transition raced the bench kit mid-project (re-stage before the freeze warning); resolved by blessing the new driver for the final benches — raw tok/s is driver-independent; MFU cells in NOTEBOOK are single-convention (LoRA-corrected) throughout.
