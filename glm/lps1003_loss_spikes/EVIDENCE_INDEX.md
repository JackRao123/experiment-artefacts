# LPS-1003 evidence index (for fresh-context pickup)

Read order: HANDOFF.md (unified entry point) → INVESTIGATION.md (chronological
findings log, all key numbers) → VERDICT.md (conclusion + fixes) → this file
(claim → raw artifact map).

## Claim → artifact map

| Claim | Raw artifact (this dir unless noted) |
|---|---|
| Bump batches not harder on base weights; wobble noise floor | `phaseA_batches.jsonl` (+ `phaseA_lp/*.npz` per-token), `phaseC_datums.jsonl`; analyze with `analyze_probe.py` |
| Byte-exact training replays flat, twice | `trainF_arm0.jsonl`, `trainF_arm1.jsonl`; compare with `compare_arms.py` |
| Live prod destruction events (per-datum 5–11 nats, partition tails) | `live_prod_probes/event1_midtraining_probe.txt`, `event2_reinit_warm_process.txt` (incl. healing reps), rotation: `event4_rotation_test_fresh_boot.txt`; event3 (hammer cycle 0) numbers transcribed in INVESTIGATION.md (raw jsonl lost to pod deletion) |
| Healing without weight changes | `event2_reinit_warm_process.txt` reps 1–5 (all /forward, no optim) |
| Rotation prediction pre-registered | INVESTIGATION.md "Rotation test (queued…)" section written before `event4` ran |
| Devbox never fires (same image/weights/config/seed) | `devbox_artifacts/phaseR_probe.jsonl`, `devbox_artifacts/es_hammer.jsonl`, `devbox_artifacts/poison_probe.jsonl` (+ logs); `live_prod_probes/devbox_rebuild_clean.txt` |
| Prod-vs-devbox identity checks | INVESTIGATION.md §6 + prod-mimic section (weights sha256, config from live pod 232l69w, env dump) |
| Server-side bump confirmation + metric formula | INVESTIGATION.md §4 (formula: datum_mean = train_mean_nll × num_loss_tokens / 32; validated 0.003037×8052/32 = 0.7642 = client value; prod 0.006095×8052/32 = 1.5335 = W&B) — ClickHouse `baseten.trainer_deployment_logs`, deployments 4w5y6r3 / 8w6y12q |
| B200 same-signature forensics | INVESTIGATION.md "Corrected comparator matrix" (deployment e3m97kq: runs 6wg87jq → 4w5jk73 → 4q9ex6w, hyd cluster) |
| DSA top-k tie degeneracy | `devbox_artifacts/margin_sel*/rank*.jsonl` + INVESTIGATION.md margin summary |

**2026-07-30 ~10:15: full devbox artefact mirror pulled to `devbox_artifacts/`
(74 MB, incl. per-token logprob npz for phaseA + both training arms, all
margin captures, latest boot log). The devbox is no longer required to verify
any claim. Note `devbox_artifacts/lps1003_warmup.patch` + `val_patch*.jsonl`:
the main session's post-rebuild-warmup mitigation and its live validation run.**
| cp>1 required (cp1 minimal config clean 5/5; minimization boundary) | `min64k/` — README.md, exact payload, per-rep full logprobs `min64k_0731_234442/`, `positional_deltas.json.gz` |
| cp16+trigger alone insufficient @32k single-doc rows (clean 3/3; row geometry implicated) | `min32k/` — README.md, payload + meta (incl. real datum b21 i13), full logprobs `min32k_0801_003030/` |
| cp16@32k multi-doc (3×10k, real tail doc) also clean; /forward zero-fills weight-0 logprobs; ambient real-text churn floor 12–18% tokens >1 nat | `min32k/x3x10k_0801_011540/` (window + mud_solo/mud_first/b21 uniform resend), payloads + meta in `min32k/` |
| POSITIVE CONTROL 08-01: partition-1-only fires (rep0 destroys docs 4-6, heals); cutoff chunk 17/32 + flicker band 13-16; negatives validated | `ctrl/` — README.md, payload_b0_part1_uniform.json + meta, full-position dumps `ctrl_p1_0801_013628/` |
| Steady-state persistence 08-01: docs 4-6 visit coherent low-NLL state ~8%/rep, flat rate, 146 reps (conn unset) | `ctrl/soak_0801_015956/` — soak.jsonl (all reps), 17 exemplar full-position dumps, `ctrl/soak.py` |
| Payload/bundle provenance | `probe_bundle.jsonl.gz` (22 labeled batches), `train_bundle_0_31.jsonl.gz` (batches 0-31), builders `build_probe_bundle.py` / `build_train_bundle.py`; batch aggregates bit-match prod ClickHouse submit lines (INVESTIGATION.md §1) |

## Negative/caveat results (do not overclaim)

- cp1 minimal config (2026-08-01, `min64k/`): did NOT fire — but this is a
  BOUNDARY, not an exoneration of anything on the cp>1 path; cp1 never
  reaches THD packing/zigzag. Synthetic data is still untested under cp>1,
  so "synthetic data can't repro" must NOT be claimed from this run.
- 0xFF GPU-memory poison on devbox + fresh boot: did NOT fire
  (`poison_probe.jsonl`, 4 clean reps) — explained post-hoc: 0xFFFFFFFF
  decodes to int32 -1 = the legitimate invalid sentinel. The mechanism is
  localized (top-k output_indices torch.empty under-write); catching the
  exact radix under-write edge in the act is still open (dump test /
  in-range-int poison, spec in experiment_handoff.md).
- `prod_hammer_partial.jsonl` is empty (salvage raced pod deletion); hammer
  cycle-0 numbers are in INVESTIGATION.md text only.
- Secondary bug: 3rd in-process init_trainer_server rebuild deadlocked the
  prod trainer (transcribed in INVESTIGATION.md; pods since deleted).

## Afternoon additions (2026-07-30, branch session)

| Claim | Artifact |
|---|---|
| CP exonerated at model level: Nemotron-3-Ultra (CP4 THD, no DSA) flat — 0 spikes/30+ steps through TWO fresh-init windows, bit-identical step-0 across a rebuild; GLM runs 3-8 spikes in same window | W&B `baseten-training/jackrao-lps1003-compare` (run dm1efc09 live + 6 mirrors); simple 4-run overlay in `baseten-training/jackrao-lps1003` |
| Actual defect localized: cuDNN DSA top-k torch.empty output + trusting glue; CP/packing/allgather Python path fully clean; poison-null explained (0xFF = -1 sentinel) | `CODE_AUDIT_TOPK.md` (incl. eliminated candidates + verification set) |
| Varlen indexer-forward scores ARE pre-filled -inf (candidate 3 eliminated) | wheel read: cudnn/deepseek_sparse_attention/indexer_forward/_interface.py:209 (unconditional fill) |
| PR #843 = window-closing mitigation (full-footprint warmup at boot + after rebuild), buffer-agnostic by design; real fix = fill_(-1) glue hardening + wheel radix write-out fix | github.com/basetenlabs/trainers/pull/843 + `CODE_AUDIT_TOPK.md` |

## Evening additions (2026-07-30, code-verification session)

All wheel cites = unpacked `nvidia_cudnn_frontend-1.26.0+dsatopk1` wheel;
glue cites = Megatron `dsa_cudnn_kernels.py`.

| Claim | Artifact |
|---|---|
| Read-past-scores hypothesis dead, wall 1: scores output pre-filled -inf unconditionally before every launch | `indexer_forward/_interface.py:207-209` (SM100), `_interface_sm90.py:164` |
| Wall 2: kernel n-block loop clamped to per-segment seqlen_k | `indexer_forward/indexer_fwd_sm100.py:1017-1028` (`_causal_num_n_blocks`) |
| Wall 3: boundary tiles per-element masked to -inf in registers pre-store; interior tiles in-bounds by invariant | `indexer_fwd_sm100.py:1168-1177` (invariant comment `:1163-1167`) |
| Wall 4: K input fully materialized, no uninit padding region exists | `dsa_cudnn_kernels.py:796-797` (`index_select` → `segmented_k`) |
| TMA-pad residue bounded: ≤3 uninit output cols, beyond the consumed view, past causal window | `_interface.py:127-138` |
| Multi-CTA / dynamic-multi-CTA compiled OUT on our path; static one-CTA-per-row grid; persistent sched off | `indexer_top_k_decode_varlen.py:655-663` (defaults), `:534-541` (launch), `:611-612`, `:676` |
| Trivial branch (length ≤ K) full-writes incl. explicit -1 padding → under-write must be in long-row radix branch | `indexer_top_k_varlen_util.py:468-487` |
| `large_occupancy = num_rows > 148` is a compile-cache key; shrinks smem candidate buf + enables gmem spill; prod-only variant, uncovered by small tests | `indexer_top_k_decode_varlen.py:611`, `:157-178` |
| decode_varlen is the ONLY top-k impl the wheel ships; public API binds directly to it | `indexer_top_k/api.py:18-19`, `indexer_top_k/__init__.py` |
| Our usage = NVIDIA's announced GLM-5.2 long-context CP-training recipe (not misuse); strengthens upstream report | github.com/NVIDIA-NeMo/Megatron-Bridge/discussions/4957 |
| Allocator size-binning: recycled (rows×2048 int32) block plausibly = previous top-k indices tensor → dump-test expectation = prev-layer echoes | reasoning in `CODE_AUDIT_TOPK.md` (Healing bullet) |

## Night additions (2026-07-30, experiment session — devbox tj-3y0gjkq)

| Claim | Artifact |
|---|---|
| Output-indices under-write NOT reproducible in-situ: 0 unwritten slots / 0 dups / 0 out-of-window over 42,336 top-k calls and 190,506,624 rows (178.8M through the radix branch), 91 shapes staged, staging positive control 1.0000 on every shape | `probe2/runs/stage/SUMMARY.txt` (+ `probe_stage.jsonl/.log`); harness `probe2/sitecustomize.py`, analyzer `probe2/audit_summary.py` |
| Detector is proven, not assumed: staged (rows,k) int32 block is handed to the wheel's `torch.empty` 100% of the time, incl. under expandable_segments | `ctrl=1.0000` / "staging control OK" lines in the same artifacts + `probe2/test_staging.py` |
| Candidate-flood/spill edge refuted: overflow ties spill to gmem when `enable_gmem_store`, and capacity ≥ bucketed cols when not; tested to 98,240 ties vs a 16,384-entry buffer | `probe2/results_flood_sweep.txt` (24 runs) + wheel read varlen_util.py:40-95, :652-690, :805 |
| Real prod geometries clean: odd/unaligned sk (39501/48906/52668/58311), arange + interleaved per-row windows, mixed window<k and window>k launches, rows to 15,360 | `probe2/results_prodgeom.txt` (72 runs) + `probe2/test_prodgeom.py` |
| Scores-path refutation confirmed at RUNTIME: invalid region as delivered to top-k is 100.000% -inf over 1,601,212,824,960 elements (0 finite / 0 NaN / 0 +inf) | `probe2/runs/audit/SUMMARY.txt`; makes the spec's `neginf` control an analytic no-op |
| E0: cuDNN is the sole score producer (cudnn_fw=17,883 vs torch_scores=0); all tfsc calls take the bottom_right_key_start branch; K always even so the torch odd-K fallback never masked anything; launches prod-shaped (rows to 15,360 > 148 → large_occupancy variant) | both `probe2/runs/*/SUMMARY.txt` |
| Harness does not perturb numerics: batch-0 mean_nll 0.7643/0.7651/0.7647/0.7659 (stage) and 0.7681/0.7613/0.7613 (audit), 0 datums > 2.0 | `probe2/runs/*/probe_*.jsonl` |
| Devbox fire count now 0/8 fresh windows (6 prior + 2 here) vs prod 4/4 | this table + earlier devbox rows |

**Devbox-script fix made along the way (tj-3y0gjkq, 2×8 B300):** `.devbox_up/run_trainer_node.sh` derived `master_addr` from the k8s `BT_LEADER_ADDR`, but torchrun's static rendezvous hosts its TCPStore on the Slurm NODEID-0 node, which on this box is *not* the leader pod → rendezvous deadlock (agents dial a host with no store; 3 boots lost). Now derived from `scontrol show hostnames | head -1`, and as an IP rather than a hostname because each pod's `/etc/hosts` carries only the *other* nodes' entries, so the store master cannot resolve its own name. Worth folding into devbox-up itself.

## Bisection additions (2026-07-31, data-vs-script session)

| Claim | Artifact |
|---|---|
| DATA ruled out exhaustively: all 1024 documents (32 batches × 32) probed on frozen weights — 0 datums >2.0 nats, 0 >5.0, worst in corpus = 1.153 nats vs prod destruction 5-11 nats; batch means 0.7196-0.7954 | `probe2/runs/sweep/probe.jsonl` + `.log` |
| Prod-label content correlation is noise: spike/bump vs quiet = +0.0113 nats, Welch t=+1.25 (n=704) | `probe2/data_forensics.py` output |
| POSITION ruled out: tail-of-partition = -0.0070 nats vs mid/head at batch level (t=-0.62); -0.0027 nats at token level over 500,679 supervised tokens (t=-0.39); flat slot profile | `probe2/data_forensics.py`, `probe2/worst_tokens.py` |
| PACKING deterministic: 4 byte-identical reps → exactly 4-periodic top-k launch fingerprints on every rank ⇒ no deterministic (data/script) bug can produce heal-on-replay (6.328→0.427 nats, same payload) | `probe2/runs/stage/PACKING_DETERMINISM.txt`, `probe2/packing_determinism2.py` |
| MASKING/alignment correct: first supervised target is the constant token `{"` (id 4913) in all 2112 datum-dumps, exactly after `...<|assistant|><think></think>`; prefix_len well-formed in all 704 | `probe2/worst_tokens.py` "first supervised token identity" section |
| TOKENIZER healthy: worst tokens decode to unpredictable content (`'C'` in "(Citation 1)" 20.5 nats, `' Mak'` in "Makrgeorgou", `'/J'` in "Flythe/JAMA") at stable positions, wobbling 1-2 nats | same, run on-box with the GLM-5.2-FP8 tokenizer |
| Loss concentration (new): per-token p50=0.0106, p90=2.56, p99=7.82, max=20.5 over 500,679 tokens; supervised fraction 0.0089 ⇒ datum mean carried by ~10% of tokens; destruction needs ~1250 extra nats = wholesale corruption, not bad tokens | same |
| Fresh window with the warmup mitigation OFF (BT_SKIP_FULL_WARMUP=1, warmup 85.9s vs 252.6s) still clean: 6 reps 0.7591-0.7669, max datum 1.006 ⇒ my earlier boots were not confounded by the patch. Devbox 0/9 vs prod 4/4 | `probe2/runs/window/` (probe.jsonl/.log/SUMMARY.txt) |
| Residual search space is the prod IMAGE vs devbox source-built venv (same commit, different artifacts). Devbox versions: torch 2.11.0+cu130, cudnn-frontend 1.26.0+dsatopk1, TE 2.16.0, triton 3.6.0, cudnn-cu13 9.19.0.56, cublas 13.6.0.2, transformers 5.8.1, cutlass-dsl 4.5.2 | INVESTIGATION.md "2026-07-31 — BISECTION" |

## Operational state

Maintained in ONE place: see **HANDOFF.md → "Live operational state"**
(unified 2026-07-30 evening; earlier per-session state sections that lived
here were removed to prevent staleness).
